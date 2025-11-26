import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.filters as kf

# 移除不再需要的导入
# from ..modules.reference import ExternalReferenceModule

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        intersection = torch.sum(pred * target)
        union = torch.sum(pred) + torch.sum(target)
        dice_coeff = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice_coeff

class ComprehensiveLoss(nn.Module):
    def __init__(self, w_main=1.0, w_gdt1=0.3, w_gdt2=0.3, w_dice=0.8, stage_weights=None, w_boundary=0.1):
        super(ComprehensiveLoss, self).__init__()
        self.w_main = w_main
        self.w_gdt1 = w_gdt1
        self.w_gdt2 = w_gdt2
        self.w_dice = w_dice
        self.w_boundary = w_boundary

        self.stage_weights = stage_weights

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.gdt_loss = nn.BCELoss()
        self.l1_loss = nn.L1Loss()


    def set_stage_weights(self, weights):
        self.stage_weights = weights

    def forward(self, predictions, targets):
        total_loss = 0.0

        # 从 targets 字典直接获取所有真值
        seg_gt = targets['seg']
        gdt1_gt = targets['gdt1']
        gdt2_gt = targets['gdt2']

        if self.stage_weights is None:
            sw = [0.1, 0.2, 0.3, 0.4][:len(predictions)]
        else:
            sw = self.stage_weights

        for idx, level_preds in enumerate(predictions):
            seg_pred = level_preds['seg']
            pred_size = seg_pred.shape[2:]

            seg_gt_resized = F.interpolate(seg_gt, size=pred_size, mode='bilinear', align_corners=False)
            
            loss_bce_seg = self.bce_loss(seg_pred, seg_gt_resized)
            loss_dice_seg = self.dice_loss(torch.sigmoid(seg_pred), seg_gt_resized)
            loss_seg = loss_bce_seg + self.w_dice * loss_dice_seg

            seg_sigmoid = torch.sigmoid(seg_pred)
            seg_pred_mask = seg_sigmoid.detach()

            # 边界损失（Sobel 梯度 L1）
            edge_pred = kf.sobel(seg_sigmoid)
            edge_gt = kf.sobel(seg_gt_resized)
            loss_boundary = self.l1_loss(edge_pred, edge_gt)

            gdt1_pred = level_preds['gdt1']
            gdt1_gt_resized = F.interpolate(gdt1_gt, size=pred_size, mode='bilinear', align_corners=False)
            dynamic_gdt1_gt = gdt1_gt_resized * seg_pred_mask
            loss_gdt1 = self.gdt_loss(torch.sigmoid(gdt1_pred), torch.sigmoid(dynamic_gdt1_gt))

            gdt2_pred = level_preds['gdt2']
            gdt2_gt_resized = F.interpolate(gdt2_gt, size=pred_size, mode='bilinear', align_corners=False)
            dynamic_gdt2_gt = gdt2_gt_resized * seg_pred_mask
            loss_gdt2 = self.gdt_loss(torch.sigmoid(gdt2_pred), torch.sigmoid(dynamic_gdt2_gt))
            
            level_loss = (self.w_main * loss_seg) + (self.w_gdt1 * loss_gdt1) + (self.w_gdt2 * loss_gdt2) + (self.w_boundary * loss_boundary)
            level_loss = level_loss * sw[idx] if idx < len(sw) else level_loss
            total_loss += level_loss

        return total_loss / len(predictions)
