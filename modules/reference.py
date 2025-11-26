import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.color as kcolor
import kornia.filters as kf
from einops import rearrange

# 支持相对导入和绝对导入
try:
    from .gcn import GCN
except ImportError:
    from gcn import GCN

class ExternalReferenceModule(nn.Module):
    """
    外部参考模块
    计算输入图像的一阶和二阶梯度，作为外部结构参考。
    """
    def __init__(self):
        super(ExternalReferenceModule, self).__init__()

    def forward(self, x):
        x_gray = kcolor.rgb_to_grayscale(x)
        gdt1_gt = kf.sobel(x_gray)
        gdt2_gt = kf.laplacian(x_gray, kernel_size=5, border_type='replicate')
        return gdt1_gt, gdt2_gt


class InternalReferenceModule(nn.Module):
    """
    内部参考模块 (最终版):
    1. 借鉴BiRefNet，执行“空间到深度”的分块转换。
    2. 在巨大的通道特征上，应用GCN进行全局建模。
    3. 使用卷积块进行特征提炼和降维。
    """
    def __init__(self, in_channels, feature_dim=64, gcn_layers=2):
        super(InternalReferenceModule, self).__init__()
        self.in_channels = in_channels
        self.feature_dim = feature_dim

        # GCN层。注意：这里的num_state是GCN的输入通道数，num_node是节点数
        # 我们将在forward中动态计算它们
        # self.gcn = GCN(...) # GCN的实例化需要动态尺寸，移到forward中

        # 特征提炼与降维的卷积块 (类似BiRefNet的ipt_blk)
        # 输入通道数会非常大，在forward中动态确定
        self.refine_block = None

    def forward(self, x, ref_feat):
        B, _, H, W = x.shape
        _, _, h, w = ref_feat.shape
        num_patches = h * w


        patch_h, patch_w = H // h, W // w
        patch_channels = self.in_channels * patch_h * patch_w
        # 'b c (h_grid ph) (w_grid pw) -> b (c ph pw) h_grid w_grid'
        patches_batch = rearrange(x, 'b c (h_grid ph) (w_grid pw) -> b (c ph pw) h_grid w_grid', 
                                  h_grid=h, w_grid=w, ph=patch_h, pw=patch_w)


        # 2. GCN全局建模
        gcn_input = patches_batch.view(B, patch_channels, num_patches)
        
        # 动态实例化GCN以匹配维度

        gcn = GCN(num_state=patch_channels, num_node=num_patches).to(x.device)
        gcn_output = gcn(gcn_input)
        
        # 恢复形状
        gcn_enhanced_map = gcn_output.view(B, patch_channels, h, w)


        # 动态实例化降维卷积块
        if self.refine_block is None or self.refine_block[0].in_channels != patch_channels:
            self.refine_block = nn.Sequential(
                nn.Conv2d(patch_channels, self.feature_dim, kernel_size=1),
                nn.ReLU(inplace=True)
            ).to(x.device)
        
        final_ref_map = self.refine_block(gcn_enhanced_map)

        return final_ref_map