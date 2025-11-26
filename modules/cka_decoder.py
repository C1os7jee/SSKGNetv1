import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvWeightAttention(nn.Module):
    def __init__(self, dims, kernel_size=7, attention_dim=64, num_heads=8):
        super().__init__()
        self.dims, self.kernel_size, self.attention_dim, self.num_heads = dims, kernel_size, attention_dim, num_heads
        self.content_mlp = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dims, dims // 4, 1), nn.ReLU(), nn.Conv2d(dims // 4, attention_dim * 3, 1))
        self.w_proj = nn.Linear(1, attention_dim)
        self.attention = nn.MultiheadAttention(embed_dim=attention_dim, num_heads=num_heads, batch_first=True)
        self.ln = nn.LayerNorm([1, kernel_size, kernel_size])
        self.ffn = nn.Sequential(nn.Linear(attention_dim, attention_dim), nn.ReLU(), nn.Linear(attention_dim, 1))

    def forward(self, weight_3x3, x):
        B, C, H, W = x.shape
        kernel_area = self.kernel_size * self.kernel_size

        weight_expanded = weight_3x3.unsqueeze(0).expand(B, -1, -1, -1, -1)
        weight_batched = weight_expanded.reshape(B * C, 1, 3, 3)
        weight_7x7_base = F.interpolate(weight_batched, size=self.kernel_size, mode="bilinear", align_corners=False)
        w_seq = weight_7x7_base.view(B * C, kernel_area, 1)

        content_signal = self.content_mlp(x).view(B, 3, self.attention_dim)
        q_mod, k_mod, v_mod = torch.unbind(content_signal, dim=1)
        repeat = C
        q_mod = q_mod.repeat_interleave(repeat, dim=0)
        k_mod = k_mod.repeat_interleave(repeat, dim=0)
        v_mod = v_mod.repeat_interleave(repeat, dim=0)

        w_seq_proj = self.w_proj(w_seq)
        q = w_seq_proj + q_mod.unsqueeze(1)
        k = w_seq_proj + k_mod.unsqueeze(1)
        v = w_seq_proj + v_mod.unsqueeze(1)

        res, _ = self.attention(query=q, value=v, key=k, need_weights=False)
        res = self.ffn(res) + w_seq
        res = res.squeeze(-1)
        final_weight = res.view(B, C, self.kernel_size, self.kernel_size)
        return final_weight

class CKABlock(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
        self.conv3 = nn.Conv2d(dims, dims, kernel_size=3, padding=1, groups=dims)
        self.weight_generator_7x7 = ConvWeightAttention(dims, kernel_size=7)
        self.conv_fuse = nn.Sequential(nn.BatchNorm2d(2 * dims), nn.Conv2d(2 * dims, 4 * dims, 1), nn.ReLU(inplace=True), nn.Conv2d(4 * dims, dims, 1))

    def forward(self, x):
        r3 = self.conv3(x) + x
        dynamic_w7 = self.weight_generator_7x7(self.conv3.weight, x)
        B, C, H, W = x.shape
        grouped_weight = dynamic_w7.view(B * C, 1, 7, 7)
        x_grouped = x.view(1, B * C, H, W)
        r7 = F.conv2d(x_grouped, grouped_weight, padding=3, groups=B * C).view(B, C, H, W) + x
        r_cat = torch.cat([r3, r7], dim=1)
        return self.conv_fuse(r_cat)

class DWBlocks(nn.Module):
    def __init__(self, dims, num_blocks):
        super().__init__()
        self.blocks = nn.ModuleList([CKABlock(dims) for _ in range(num_blocks)])
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class UpSample(nn.Module):
    def __init__(self, in_c, out_c):
        super(UpSample, self).__init__()
        self.mlp = nn.Sequential(nn.Conv2d(in_c, out_c, 1), nn.GroupNorm(1, out_c), nn.ReLU(inplace=True))
    def forward(self, x):
        return self.mlp(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))

class GradientAttentionHead(nn.Module):
    def __init__(self, in_channels, inter_channels=16):
        super().__init__()
        self.conv_block = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, 1, 1), nn.BatchNorm2d(inter_channels), nn.ReLU(inplace=True))
        self.pred_head = nn.Conv2d(inter_channels, 1, 1, 1, 0)
        self.attn_head = nn.Conv2d(inter_channels, 1, 1, 1, 0)
    def forward(self, x):
        feat = self.conv_block(x)
        return self.pred_head(feat), self.attn_head(feat).sigmoid()

class CKADecoder(nn.Module):
    def __init__(self, dims, gcn_channels, num_blocks_per_stage=2):
        super().__init__()
        # dims = [512, 320, 128, 64]
        
        # 定义用于融合GCN特征的卷积层
        self.fuse_gcn4 = nn.Conv2d(gcn_channels, dims[1], 1)
        self.fuse_gcn3 = nn.Conv2d(gcn_channels, dims[2], 1)
        self.fuse_gcn2 = nn.Conv2d(gcn_channels, dims[3], 1)
        self.fuse_gcn1 = nn.Conv2d(gcn_channels, dims[3], 1)

        # 解码器主干
        self.d4 = DWBlocks(dims[1], num_blocks_per_stage)
        self.d3 = DWBlocks(dims[2], num_blocks_per_stage)
        self.d2 = DWBlocks(dims[3], num_blocks_per_stage)
        self.d1 = DWBlocks(dims[3], num_blocks_per_stage)

        self.up4 = UpSample(dims[0], dims[1])
        self.up3 = UpSample(dims[1], dims[2])
        self.up2 = UpSample(dims[2], dims[3])
        self.up1 = UpSample(dims[3], dims[3])

        # 预测头
        self.seg_head4 = nn.Conv2d(dims[1], 1, 1)
        self.seg_head3 = nn.Conv2d(dims[2], 1, 1)
        self.seg_head2 = nn.Conv2d(dims[3], 1, 1)
        self.seg_head1 = nn.Conv2d(dims[3], 1, 1)

        self.gdt1_head4 = GradientAttentionHead(dims[1])
        self.gdt2_head4 = GradientAttentionHead(dims[1])
        self.gdt1_head3 = GradientAttentionHead(dims[2])
        self.gdt2_head3 = GradientAttentionHead(dims[2])
        self.gdt1_head2 = GradientAttentionHead(dims[3])
        self.gdt2_head2 = GradientAttentionHead(dims[3])
        self.gdt1_head1 = GradientAttentionHead(dims[3])
        self.gdt2_head1 = GradientAttentionHead(dims[3])

    def forward(self, all_features):
        encoder_features = all_features['encoder']
        gcn_features = all_features['gcn']
        if len(gcn_features) != 4:
            raise ValueError(f"Expected 4 GCN feature maps, got {len(gcn_features)}")

        x4, x3, x2, x1 = encoder_features
        all_preds = []

        def align_gcn_feat(gcn_feat, target_hw):
            if gcn_feat.shape[2:] != target_hw:
                return F.interpolate(gcn_feat, size=target_hw, mode='bilinear', align_corners=False)
            return gcn_feat

        # --- Stage 4 ---
        y4_in_skip = self.up4(x4) + x3
        gcn_ref4 = align_gcn_feat(gcn_features[0], y4_in_skip.shape[2:])
        y4_in = y4_in_skip + self.fuse_gcn4(gcn_ref4)
        y4_processed = self.d4(y4_in)
        gdt1_pred4, gdt1_attn4 = self.gdt1_head4(y4_processed)
        gdt2_pred4, gdt2_attn4 = self.gdt2_head4(y4_processed)
        y4_modulated = y4_processed * gdt1_attn4 * gdt2_attn4
        seg_pred4 = self.seg_head4(y4_modulated)
        all_preds.append({'seg': seg_pred4, 'gdt1': gdt1_pred4, 'gdt2': gdt2_pred4})

        # --- Stage 3 ---
        y3_in_skip = self.up3(y4_modulated) + x2
        gcn_ref3 = align_gcn_feat(gcn_features[1], y3_in_skip.shape[2:])
        y3_in = y3_in_skip + self.fuse_gcn3(gcn_ref3)
        y3_processed = self.d3(y3_in)
        gdt1_pred3, gdt1_attn3 = self.gdt1_head3(y3_processed)
        gdt2_pred3, gdt2_attn3 = self.gdt2_head3(y3_processed)
        y3_modulated = y3_processed * gdt1_attn3 * gdt2_attn3
        seg_pred3 = self.seg_head3(y3_modulated)
        all_preds.append({'seg': seg_pred3, 'gdt1': gdt1_pred3, 'gdt2': gdt2_pred3})

        # --- Stage 2 ---
        y2_in_skip = self.up2(y3_modulated) + x1
        gcn_ref2 = align_gcn_feat(gcn_features[2], y2_in_skip.shape[2:])
        y2_in = y2_in_skip + self.fuse_gcn2(gcn_ref2)
        y2_processed = self.d2(y2_in)
        gdt1_pred2, gdt1_attn2 = self.gdt1_head2(y2_processed)
        gdt2_pred2, gdt2_attn2 = self.gdt2_head2(y2_processed)
        y2_modulated = y2_processed * gdt1_attn2 * gdt2_attn2
        seg_pred2 = self.seg_head2(y2_modulated)
        all_preds.append({'seg': seg_pred2, 'gdt1': gdt1_pred2, 'gdt2': gdt2_pred2})

        # --- Stage 1 ---
        y1_in_skip = self.up1(y2_modulated)
        gcn_ref1 = align_gcn_feat(gcn_features[3], y1_in_skip.shape[2:])
        y1_in = y1_in_skip + self.fuse_gcn1(gcn_ref1)
        y1_processed = self.d1(y1_in)
        gdt1_pred1, gdt1_attn1 = self.gdt1_head1(y1_processed)
        gdt2_pred1, gdt2_attn1 = self.gdt2_head1(y1_processed)
        y1_modulated = y1_processed * gdt1_attn1 * gdt2_attn1
        seg_pred1 = self.seg_head1(y1_modulated)
        all_preds.append({'seg': seg_pred1, 'gdt1': gdt1_pred1, 'gdt2': gdt2_pred1})

        return all_preds
