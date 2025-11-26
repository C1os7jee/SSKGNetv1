
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import os
import sys
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import kornia.color as kcolor
import kornia.filters as kf
import numpy as np

# 动态添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from GDR_Net.gdr_net import GDRNet
# 复用训练脚本中的数据集类
from GDR_Net.train import SegmentationDataset

def evaluate_model(args):
    """
    评估单个训练好的模型在测试集上的性能。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 加载模型
    model = GDRNet().to(device)
    if not os.path.exists(args.model_path):
        print(f"错误: 在路径 {args.model_path} 未找到模型权重")
        return
        
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print(f"模型已从 {args.model_path} 加载")

    # 2. 加载测试数据集
    # 注意：这里的data_path应该指向包含'images'和'masks'子文件夹的目录
    test_dataset = SegmentationDataset(data_root=args.data_path, image_size=args.img_size, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    print(f"测试数据已从 {args.data_path} 加载: {len(test_dataset)} 张图片。")

    # 3. 初始化指标计算变量
    tp = [0, 0]  # [background, crack]
    fp = [0, 0]
    fn = [0, 0]

    # 4. 开始评估
    progress_bar = tqdm(test_loader, desc="正在评估")
    with torch.no_grad():
        for images, masks in progress_bar:
            images, masks = images.to(device), masks.to(device)

            # 模型只接收3通道RGB图像
            predictions = model(images)
            
            # 使用最终的、也是分辨率最高的预测进行评估
            final_pred_raw = predictions[-1]['seg']
            final_pred_upsampled = F.interpolate(final_pred_raw, size=masks.shape[2:], mode='bilinear', align_corners=False)
            pred_mask = torch.sigmoid(final_pred_upsampled)
            pred_mask_binary = (pred_mask > 0.5).float()
            
            gt_mask_binary = (masks > 0.5).float()

            # 计算 TP, FP, FN
            # i=0 for background, i=1 for crack
            for i in range(2):
                pred_i = (pred_mask_binary == i)
                gt_i = (gt_mask_binary == i)
                tp[i] += (pred_i & gt_i).sum().item()
                fp[i] += (pred_i & ~gt_i).sum().item()
                fn[i] += (~pred_i & gt_i).sum().item()

    # 5. 计算并打印最终指标
    iou = [tp[i] / (tp[i] + fp[i] + fn[i] + 1e-6) for i in range(2)]
    f1 = [2 * tp[i] / (2 * tp[i] + fp[i] + fn[i] + 1e-6) for i in range(2)]
    precision = [tp[i] / (tp[i] + fp[i] + 1e-6) for i in range(2)]
    recall = [tp[i] / (tp[i] + fn[i] + 1e-6) for i in range(2)]
    
    mIoU = np.mean(iou)
    mF1 = np.mean(f1)

    print("\n--- 评估完成 ---")
    print(f"模型指标: {os.path.basename(args.model_path)}")
    print("---------------------------------")
    print(f"  平均交并比 (mIoU): {mIoU:.4f}")
    print(f"  平均F1分数 (mF1/Dice): {mF1:.4f}")
    print("\n--- 裂缝类别指标 ---")
    print(f"  交并比 (IoU): {iou[1]:.4f}")
    print(f"  F1分数 (F1-Score): {f1[1]:.4f}")
    print(f"  精确率 (Precision): {precision[1]:.4f}")
    print(f"  召回率 (Recall): {recall[1]:.4f}")
    print("\n--- 背景类别指标 ---")
    print(f"  交并比 (IoU): {iou[0]:.4f}")
    print(f"  F1分数 (F1-Score): {f1[0]:.4f}")
    print("---------------------------------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="评估 GDR-Net 模型")
    parser.add_argument('--model_path', type=str, required=True, help='已训练模型的权重路径 (.pth 文件)')
    parser.add_argument('--data_path', type=str, default='/home/ubuntu/CJ-Seg/data/DeepCrack/test', help='测试数据集的根目录 (应包含 images 和 masks 文件夹)')
    parser.add_argument('--img_size', type=int, default=384, help='训练时使用的图像尺寸')
    
    args = parser.parse_args()
    evaluate_model(args)
