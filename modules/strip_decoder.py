import torch
import torch.nn as nn
import torch.nn.functional as F


class StripBlock(nn.Module):
    """条状 depthwise 卷积块，替代原 CKABlock 的动态核注意力。"""

    def __init__(self, dim, k1=1, k2=19):
        super().__init__()
        self.dw_3x3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.strip_h = nn.Conv2d(
            dim, dim, kernel_size=(k1, k2), padding=(k1 // 2, k2 // 2), groups=dim
        )
        self.strip_v = nn.Conv2d(
            dim, dim, kernel_size=(k2, k1), padding=(k2 // 2, k1 // 2), groups=dim
        )
        self.pointwise = nn.Conv2d(dim, dim, kernel_size=1)
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.GELU()

    def forward(self, x):
        residual = x
        attn = self.dw_3x3(x)
        attn = self.strip_h(attn)
        attn = self.strip_v(attn)
        attn = self.pointwise(attn)
        attn = self.bn(attn)
        attn = self.act(attn)
        return residual * attn


class DynamicStripBlock(nn.Module):
    """
    动态条状卷积：通过轻量注意力为每个样本/通道生成 (1×k)/(k×1) depthwise 卷积核。
    保留条状感受野，同时引入内容自适应。
    """

    def __init__(self, dim, k=19, attn_dim=64, num_heads=8):
        super().__init__()
        self.dim = dim
        self.k = k
        self.kernel_area = k
        self.num_heads = num_heads
        self.attn = nn.MultiheadAttention(embed_dim=attn_dim, num_heads=num_heads, batch_first=True)
        self.w_proj = nn.Linear(1, attn_dim)
        self.content_mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.GELU(),
            nn.Conv2d(dim // 4, attn_dim * 3, 1),
        )
        # 基础可学习条状核
        self.base_h = nn.Parameter(torch.randn(dim, 1, 1, k) * 0.01)
        self.base_v = nn.Parameter(torch.randn(dim, 1, k, 1) * 0.01)
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.GELU()

    def _dynamic_kernel(self, base_w, qkv_mod):
        # base_w: (C,1,1,k) 或 (C,1,k,1)；qkv_mod: (B,C,3,attn_dim)
        B, C, _, attn_dim = qkv_mod.shape
        # 展平成 (B*C, k, 1)
        w_seq = base_w.view(1, C, -1).repeat(B, 1, 1).view(B * C, self.kernel_area, 1)
        w_seq_proj = self.w_proj(w_seq)  # (B*C, k, attn_dim)

        q_mod, k_mod, v_mod = torch.unbind(qkv_mod, dim=2)  # (B,C,attn_dim) *3
        q_mod = q_mod.reshape(B * C, 1, attn_dim)
        k_mod = k_mod.reshape(B * C, 1, attn_dim)
        v_mod = v_mod.reshape(B * C, 1, attn_dim)

        q = w_seq_proj + q_mod
        k = w_seq_proj + k_mod
        v = w_seq_proj + v_mod

        res, _ = self.attn(query=q, key=k, value=v, need_weights=False)
        res = res + w_seq_proj
        res = res.mean(dim=-1)  # 降维到 (B*C, k)
        res = res.view(B, C, self.kernel_area)  # (B,C,k)
        return res

    def forward(self, x):
        B, C, H, W = x.shape
        # 内容特征 -> qkv
        content = self.content_mlp(x)  # (B, 3*attn_dim, 1, 1)
        content = content.view(B, 3, -1).unsqueeze(1).repeat(1, C, 1, 1)  # (B,C,3,attn_dim)

        dyn_h = self._dynamic_kernel(self.base_h, content)
        dyn_v = self._dynamic_kernel(self.base_v, content)

        x_group = x.view(1, B * C, H, W)
        w_h = dyn_h.view(B * C, 1, 1, self.k)
        w_v = dyn_v.view(B * C, 1, self.k, 1)
        out_h = F.conv2d(x_group, w_h, padding=(0, self.k // 2), groups=B * C).view(B, C, H, W)
        out_v = F.conv2d(x_group, w_v, padding=(self.k // 2, 0), groups=B * C).view(B, C, H, W)
        out = out_h + out_v
        out = self.bn(out)
        out = self.act(out)
        return x * out


class StripDecodeStage(nn.Module):
    def __init__(self, block_cls, block_kwargs, blocks=2):
        super().__init__()
        self.blocks = nn.Sequential(*[block_cls(**block_kwargs) for _ in range(blocks)])

    def forward(self, x):
        return self.blocks(x)


class UpSampleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1),
            nn.GroupNorm(1, out_c),
            nn.GELU(),
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.conv(x)


class StripDecoder(nn.Module):
    """基于条状卷积的解码器，接口与 CKADecoder 保持一致。"""

    def __init__(self, dims, gcn_channels, num_blocks_per_stage=2, k2=15, use_dynamic=False, stage_k2s=None, use_low_gcn=True):
        super().__init__()
        # GCN 融合层
        self.fuse_gcn4 = nn.Conv2d(gcn_channels, dims[1], 1)
        self.fuse_gcn3 = nn.Conv2d(gcn_channels, dims[2], 1)
        self.fuse_gcn2 = nn.Conv2d(gcn_channels, dims[3], 1)
        self.fuse_gcn1 = nn.Conv2d(gcn_channels, dims[3], 1)
        self.use_low_gcn = use_low_gcn

        stage_k2s = stage_k2s or (k2, k2, k2, k2)

        if use_dynamic:
            block_cls = DynamicStripBlock
            def make_kwargs(dim, k_val):
                return {"dim": dim, "k": k_val, "attn_dim": 64, "num_heads": 8}
        else:
            block_cls = StripBlock
            def make_kwargs(dim, k_val):
                return {"dim": dim, "k1": 1, "k2": k_val}

        self.d4 = StripDecodeStage(block_cls, make_kwargs(dims[1], stage_k2s[0]), num_blocks_per_stage)
        self.d3 = StripDecodeStage(block_cls, make_kwargs(dims[2], stage_k2s[1]), num_blocks_per_stage)
        self.d2 = StripDecodeStage(block_cls, make_kwargs(dims[3], stage_k2s[2]), num_blocks_per_stage)
        self.d1 = StripDecodeStage(block_cls, make_kwargs(dims[3], stage_k2s[3]), num_blocks_per_stage)

        self.up4 = UpSampleConv(dims[0], dims[1])
        self.up3 = UpSampleConv(dims[1], dims[2])
        self.up2 = UpSampleConv(dims[2], dims[3])
        self.up1 = UpSampleConv(dims[3], dims[3])

        self.seg_head4 = nn.Conv2d(dims[1], 1, 1)
        self.seg_head3 = nn.Conv2d(dims[2], 1, 1)
        self.seg_head2 = nn.Conv2d(dims[3], 1, 1)
        self.seg_head1 = nn.Conv2d(dims[3], 1, 1)

        def grad_head(c):
            return nn.Sequential(
                nn.Conv2d(c, c // 4, 3, padding=1),
                nn.BatchNorm2d(c // 4),
                nn.GELU(),
                nn.Conv2d(c // 4, 1, 1),
            )

        self.gdt1_head4 = grad_head(dims[1])
        self.gdt2_head4 = grad_head(dims[1])
        self.gdt1_head3 = grad_head(dims[2])
        self.gdt2_head3 = grad_head(dims[2])
        self.gdt1_head2 = grad_head(dims[3])
        self.gdt2_head2 = grad_head(dims[3])
        self.gdt1_head1 = grad_head(dims[3])
        self.gdt2_head1 = grad_head(dims[3])

    def _align(self, feat, target_hw):
        if feat.shape[2:] == target_hw:
            return feat
        return F.interpolate(feat, size=target_hw, mode="bilinear", align_corners=False)

    def forward(self, all_features):
        encoder_features = all_features["encoder"]
        gcn_features = all_features["gcn"]
        x4, x3, x2, x1 = encoder_features

        outputs = []

        y4 = self.up4(x4) + x3
        y4 = y4 + self.fuse_gcn4(self._align(gcn_features[0], y4.shape[2:]))
        y4 = self.d4(y4)
        gdt1_4 = self.gdt1_head4(y4)
        gdt2_4 = self.gdt2_head4(y4)
        mod4 = y4 * torch.sigmoid(gdt1_4) * torch.sigmoid(gdt2_4)
        outputs.append({"seg": self.seg_head4(mod4), "gdt1": gdt1_4, "gdt2": gdt2_4})

        y3 = self.up3(mod4) + x2
        y3 = y3 + self.fuse_gcn3(self._align(gcn_features[1], y3.shape[2:]))
        y3 = self.d3(y3)
        gdt1_3 = self.gdt1_head3(y3)
        gdt2_3 = self.gdt2_head3(y3)
        mod3 = y3 * torch.sigmoid(gdt1_3) * torch.sigmoid(gdt2_3)
        outputs.append({"seg": self.seg_head3(mod3), "gdt1": gdt1_3, "gdt2": gdt2_3})

        y2 = self.up2(mod3) + x1
        if self.use_low_gcn:
            y2 = y2 + self.fuse_gcn2(self._align(gcn_features[2], y2.shape[2:]))
        y2 = self.d2(y2)
        gdt1_2 = self.gdt1_head2(y2)
        gdt2_2 = self.gdt2_head2(y2)
        mod2 = y2 * torch.sigmoid(gdt1_2) * torch.sigmoid(gdt2_2)
        outputs.append({"seg": self.seg_head2(mod2), "gdt1": gdt1_2, "gdt2": gdt2_2})

        y1 = self.up1(mod2)
        if self.use_low_gcn:
            y1 = y1 + self.fuse_gcn1(self._align(gcn_features[3], y1.shape[2:]))
        y1 = self.d1(y1)
        gdt1_1 = self.gdt1_head1(y1)
        gdt2_1 = self.gdt2_head1(y1)
        mod1 = y1 * torch.sigmoid(gdt1_1) * torch.sigmoid(gdt2_1)
        outputs.append({"seg": self.seg_head1(mod1), "gdt1": gdt1_1, "gdt2": gdt2_1})

        return outputs
