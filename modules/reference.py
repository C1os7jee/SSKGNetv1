import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.color as kcolor
import kornia.filters as kf
from einops import rearrange

try:
    from .gcn import GCN
except ImportError:
    from gcn import GCN

class ExternalReferenceModule(nn.Module):
    """
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
 
    def __init__(self, in_channels, feature_dim=64, gcn_layers=2):
        super(InternalReferenceModule, self).__init__()
        self.in_channels = in_channels
        self.feature_dim = feature_dim
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
        
        gcn = GCN(num_state=patch_channels, num_node=num_patches).to(x.device)
        gcn_output = gcn(gcn_input)
        gcn_enhanced_map = gcn_output.view(B, patch_channels, h, w)

        if self.refine_block is None or self.refine_block[0].in_channels != patch_channels:
            self.refine_block = nn.Sequential(
                nn.Conv2d(patch_channels, self.feature_dim, kernel_size=1),
                nn.ReLU(inplace=True)
            ).to(x.device)
        
        final_ref_map = self.refine_block(gcn_enhanced_map)

        return final_ref_map