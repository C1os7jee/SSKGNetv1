import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.filters as kf

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        intersection = torch.sum(pred * target)
        union = torch.sum(pred) + torch.sum(target)
        dice_coeff = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice_coeff

# --- 新增：Soft Skeletonization 和 clDice Loss ---
class SoftSkeletonize(nn.Module):
    def __init__(self, num_iter=5): # 裂缝通常较细，iter=5 左右通常足够
        super(SoftSkeletonize, self).__init__()
        self.num_iter = num_iter

    def forward(self, x):
        # 使用 Min-Pooling 模拟形态学腐蚀 (Soft Erosion)
        # 腐蚀操作会把白色区域变细
        skel = x
        for _ in range(self.num_iter):
            # max_pool(-x) 等价于 min_pool(x)
            skel = -F.max_pool2d(-skel, kernel_size=3, stride=1, padding=1)
        return skel

class soft_cldice_loss(nn.Module):
    def __init__(self, iter_=3, smooth=1e-5):
        super(soft_cldice_loss, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.soft_skel = SoftSkeletonize(num_iter=iter_)

    def forward(self, y_pred, y_true):
        # y_pred, y_true 必须是 [0,1] 范围的概率图 (Sigmoid 之后)
        
        # 1. 计算骨架
        skel_pred = self.soft_skel(y_pred)
        skel_true = self.soft_skel(y_true)
        
        # 2. 计算 T_prec (拓扑精度): 预测的骨架 落在 真实Mask 里的比例
        # 强迫预测的骨架不要乱跑
        t_prec = (torch.sum(skel_pred * y_true) + self.smooth) / (torch.sum(skel_pred) + self.smooth)
        
        # 3. 计算 T_sens (拓扑召回): 真实的骨架 落在 预测Mask 里的比例
        # 强迫预测Mask覆盖住真实的骨架（这是解决断裂的关键！）
        t_sens = (torch.sum(skel_true * y_pred) + self.smooth) / (torch.sum(skel_true) + self.smooth)
        
        # 4. 计算 clDice
        cl_dice = 2.0 * (t_prec * t_sens) / (t_prec + t_sens + self.smooth)
        
        return 1.0 - cl_dice
# ---------------------------------------------

class ComprehensiveLoss(nn.Module):
    def __init__(self, w_main=1.0, w_gdt1=0.3, w_gdt2=0.3, w_dice=0.8, stage_weights=None, w_boundary=0.1, w_cldice=0.5):
        super(ComprehensiveLoss, self).__init__()
        self.w_main = w_main
        self.w_gdt1 = w_gdt1
        self.w_gdt2 = w_gdt2
        self.w_dice = w_dice
        self.w_boundary = w_boundary
        self.w_cldice = w_cldice # 新增 clDice 权重

        self.stage_weights = stage_weights

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.gdt_loss = nn.BCELoss()
        self.l1_loss = nn.L1Loss()
        
        # 初始化 clDice
        # iter_ 根据你的裂缝宽度调整，如果裂缝很细(1-3像素)，iter=3比较合适；如果较宽，可以设为5或10
        self.cldice_loss_fn = soft_cldice_loss(iter_=3) 


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
            seg_pred = level_preds['seg'] # Logits
            pred_size = seg_pred.shape[2:]

            seg_gt_resized = F.interpolate(seg_gt, size=pred_size, mode='bilinear', align_corners=False)
            
            # --- 基础分割损失 ---
            loss_bce_seg = self.bce_loss(seg_pred, seg_gt_resized)
            seg_sigmoid = torch.sigmoid(seg_pred) # 转换为概率
            loss_dice_seg = self.dice_loss(seg_sigmoid, seg_gt_resized)
            
            loss_seg = loss_bce_seg + self.w_dice * loss_dice_seg

            # --- 新增：clDice Loss ---
            # 只有在权重 > 0 时才计算，节省计算资源
            loss_cldice = 0.0
            if self.w_cldice > 0:
                loss_cldice = self.cldice_loss_fn(seg_sigmoid, seg_gt_resized)
                loss_seg += self.w_cldice * loss_cldice

            seg_pred_mask = seg_sigmoid.detach()

            # --- 边界损失 (Sobel 梯度 L1) ---
            edge_pred = kf.sobel(seg_sigmoid)
            edge_gt = kf.sobel(seg_gt_resized)
            loss_boundary = self.l1_loss(edge_pred, edge_gt)

            # --- 梯度辅助任务损失 ---
            gdt1_pred = level_preds['gdt1']
            gdt1_gt_resized = F.interpolate(gdt1_gt, size=pred_size, mode='bilinear', align_corners=False)
            dynamic_gdt1_gt = gdt1_gt_resized * seg_pred_mask
            loss_gdt1 = self.gdt_loss(torch.sigmoid(gdt1_pred), torch.sigmoid(dynamic_gdt1_gt))

            gdt2_pred = level_preds['gdt2']
            gdt2_gt_resized = F.interpolate(gdt2_gt, size=pred_size, mode='bilinear', align_corners=False)
            dynamic_gdt2_gt = gdt2_gt_resized * seg_pred_mask
            loss_gdt2 = self.gdt_loss(torch.sigmoid(gdt2_pred), torch.sigmoid(dynamic_gdt2_gt))
            
            # 汇总层级损失
            level_loss = (self.w_main * loss_seg) + \
                         (self.w_gdt1 * loss_gdt1) + \
                         (self.w_gdt2 * loss_gdt2) + \
                         (self.w_boundary * loss_boundary)
            
            level_loss = level_loss * sw[idx] if idx < len(sw) else level_loss
            total_loss += level_loss

        return total_loss / len(predictions)