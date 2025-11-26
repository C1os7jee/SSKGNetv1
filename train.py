import argparse
import datetime
import os
import random
import sys
import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import kornia.color as kcolor
import kornia.filters as kf

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
# 确保可以绝对导入 GDRNet 相关模块
PROJECT_ROOT = os.path.dirname(__file__)
sys.path.insert(0, PROJECT_ROOT)

from gdr_net import GDRNet
from utils.losses import ComprehensiveLoss


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SegmentationDataset(Dataset):
    """简单的 Crack Seg 数据集封装，支持训练增强。"""

    def __init__(self, data_root, image_size=480, is_train=True):
        self.data_root = data_root
        self.image_size = image_size
        self.is_train = is_train

        image_dir = os.path.join(data_root, "images")
        mask_dir = os.path.join(data_root, "masks")
        self.image_paths = sorted(
            [os.path.join(image_dir, f) for f in os.listdir(image_dir)]
        )
        self.mask_paths = sorted(
            [os.path.join(mask_dir, f) for f in os.listdir(mask_dir)]
        )

        self.base_image_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.base_mask_transform = transforms.Compose(
            [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
        )

        if self.is_train:
            self.color_augment = transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.1
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert("RGB")
            mask = Image.open(self.mask_paths[idx]).convert("L")
        except (IOError, SyntaxError):
            image = Image.new("RGB", (self.image_size, self.image_size), (0, 0, 0))
            mask = Image.new("L", (self.image_size, self.image_size), 0)

        if self.is_train:
            if random.random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
            image = self.color_augment(image)

        image = self.base_image_transform(image)
        mask = self.base_mask_transform(mask)
        return image, mask


def compute_gradient_targets(images):
    """预计算 Sobel / Laplacian 作为附加监督。"""
    with torch.no_grad():
        images_gray = kcolor.rgb_to_grayscale(images)
        gdt1 = kf.sobel(images_gray)
        gdt2 = kf.laplacian(images_gray, kernel_size=5)
    return gdt1, gdt2


def train_one_epoch(model, loader, optimizer, loss_fn, device, epoch, total_epochs, scaler, amp):
    model.train()
    total_loss = 0.0
    progress = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Train]")

    for step, (images, masks) in enumerate(progress, start=1):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        gdt1, gdt2 = compute_gradient_targets(images)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            predictions = model(images)
            targets = {"seg": masks, "gdt1": gdt1, "gdt2": gdt2}
            loss = loss_fn(predictions, targets)

        if amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        progress.set_postfix(loss=f"{total_loss / step:.4f}")

    return total_loss / max(len(loader), 1)


def _binary_stats(pred_mask, gt_mask):
    pred_positive = pred_mask
    pred_negative = 1 - pred_positive
    gt_positive = gt_mask
    gt_negative = 1 - gt_positive

    tp = (pred_positive * gt_positive).sum().item()
    fp = (pred_positive * gt_negative).sum().item()
    fn = (pred_negative * gt_positive).sum().item()
    tn = (pred_negative * gt_negative).sum().item()
    return tp, fp, fn, tn


def validate(model, loader, loss_fn, device, epoch, total_epochs, amp):
    model.eval()
    total_loss = 0.0
    stats_crack = [0.0, 0.0, 0.0]
    stats_bg = [0.0, 0.0, 0.0]

    progress = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Val]")
    with torch.no_grad():
        for step, (images, masks) in enumerate(progress, start=1):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            gdt1, gdt2 = compute_gradient_targets(images)

            with torch.cuda.amp.autocast(enabled=amp):
                predictions = model(images)
                targets = {"seg": masks, "gdt1": gdt1, "gdt2": gdt2}
                loss = loss_fn(predictions, targets)

            total_loss += loss.item()
            progress.set_postfix(loss=f"{total_loss / step:.4f}")

            final_pred = predictions[-1]["seg"]
            final_pred = F.interpolate(
                final_pred, size=masks.shape[2:], mode="bilinear", align_corners=False
            )
            pred_mask = (torch.sigmoid(final_pred) > 0.5).float()
            gt_mask = (masks > 0.5).float()

            tp, fp, fn, tn = _binary_stats(pred_mask, gt_mask)
            stats_crack[0] += tp
            stats_crack[1] += fp
            stats_crack[2] += fn
            stats_bg[0] += tn
            stats_bg[1] += fn  # background FP equals crack FN
            stats_bg[2] += fp  # background FN equals crack FP

    def compute_metrics(tp, fp, fn):
        iou = tp / (tp + fp + fn + 1e-6)
        f1 = (2 * tp) / (2 * tp + fp + fn + 1e-6)
        return iou, f1

    crack_iou, crack_f1 = compute_metrics(*stats_crack)
    bg_iou, bg_f1 = compute_metrics(*stats_bg)
    metrics = {
        "loss": total_loss / max(len(loader), 1),
        "iou_crack": crack_iou,
        "iou_background": bg_iou,
        "f1_crack": crack_f1,
        "f1_background": bg_f1,
        "mIoU": (crack_iou + bg_iou) / 2,
        "mF1": (crack_f1 + bg_f1) / 2,
    }
    return metrics


def build_dataloaders(args, device):
    train_dataset = SegmentationDataset(
        data_root=os.path.join(args.data_path, "train"),
        image_size=args.img_size,
        is_train=True,
    )
    val_dataset = SegmentationDataset(
        data_root=os.path.join(args.data_path, "test"),
        image_size=args.img_size,
        is_train=False,
    )

    common_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(
        train_dataset, shuffle=True, drop_last=False, **common_kwargs
    )
    val_loader = DataLoader(
        val_dataset, shuffle=False, drop_last=False, **common_kwargs
    )
    return train_loader, val_loader


def get_stage_weights(epoch, total_epochs):
    # 深监督权重随训练递减，后期主输出占比更高
    base = [0.4, 0.3, 0.2, 0.1]
    factor = max(0.2, 1 - (epoch - 1) / total_epochs)
    return [w * factor for w in base]


def main():
    parser = argparse.ArgumentParser(description="Train GDR-Net with StripNet backbone")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/ubuntu/CJ-Seg/data/CRACK500_1",
        help="数据集根目录（包含 train/test 子目录）",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="GDR_Net/checkpoints",
        help="保存 checkpoint 的目录",
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=480)
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="warmup 步数占总步数比例")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader worker 数量，受限环境建议设为 0",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="stripnet_s",
        choices=["stripnet_s", "pvt_v2_b2", "swin_v1_l"],
        help="选择主干网络（默认 stripnet_s）",
    )
    parser.add_argument(
        "--decoder",
        type=str,
        default="strip",
        choices=["strip", "cka"],
        help="选择解码器（条状或原 CKA）",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="启用混合精度训练（仅在 CUDA 上生效）",
    )
    parser.add_argument(
        "--boundary_loss_weight",
        type=float,
        default=0.05,
        help="边界损失权重，设为 0 可关闭边界项",
    )
    parser.add_argument(
        "--fixed_stage_weights",
        action="store_true",
        help="使用固定的深监督权重，不随 epoch 递减",
    )
    parser.add_argument(
        "--final_only",
        action="store_true",
        help="仅对最终输出施加损失，关闭中间层深监督",
    )
    parser.add_argument(
        "--strip_k2s",
        type=str,
        default=None,
        help="条状卷积各阶段 k2，格式如 19,15,11,7；留空用默认配置",
    )
    parser.add_argument(
        "--use_low_gcn",
        action="store_true",
        help="在浅层 Stage2/1 也注入 GCN 特征",
    )
    parser.add_argument(
        "--dynamic_strip",
        action="store_true",
        help="启用动态条状卷积核（默认关闭）",
    )
    parser.add_argument(
        "--cldice_weight",
        type=float,
        default=0.5,
        help="clDice 损失权重，用于保持裂缝连通性",
    )
    args = parser.parse_args()

    os.environ["GDR_BACKBONE"] = args.backbone
    os.environ["GDR_DECODER"] = "strip" if args.decoder == "strip" else "cka"
    os.environ["GDR_DYNAMIC_STRIP"] = "1" if args.dynamic_strip else "0"
    if args.strip_k2s:
        os.environ["STRIP_K2S"] = args.strip_k2s
    os.environ["USE_LOW_GCN"] = "1" if args.use_low_gcn else "0"

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = args.amp and device.type == "cuda"

    train_loader, val_loader = build_dataloaders(args, device)
    print(f"Data loaded: {len(train_loader.dataset)} train / {len(val_loader.dataset)} val samples.")

    model = GDRNet().to(device)
    base_stage_weights = [0.1, 0.2, 0.3, 0.4] if not args.final_only else [0.0, 0.0, 0.0, 1.0]

    loss_fn = ComprehensiveLoss(
        w_main=1.0,
        w_gdt1=0.2 if args.final_only else 0.3,
        w_gdt2=0.2 if args.final_only else 0.3,
        w_dice=0.8,
        stage_weights=base_stage_weights if args.fixed_stage_weights else None,
        w_boundary=args.boundary_loss_weight,
        w_cldice=args.cldice_weight, 
    )
    # 分层学习率：backbone 小，解码器/GCN 默认
    enc_params = list(model.encoder.parameters())
    dec_params = [p for n, p in model.named_parameters() if not n.startswith("encoder")]
    optimizer = torch.optim.AdamW(
        [
            {"params": enc_params, "lr": args.lr * 0.5},
            {"params": dec_params, "lr": args.lr},
        ],
        weight_decay=1e-2,
    )
    # warmup + cosine
    def lr_lambda(current_step):
        warmup_steps = max(1, int(args.warmup_ratio * args.epochs * len(train_loader)))
        total_steps = args.epochs * len(train_loader)
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join(args.ckpt_path, args.backbone, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Checkpoints will be stored in: {save_dir}")

    best_mIoU = 0.0
    best_metrics = None

    for epoch in range(1, args.epochs + 1):
        if not args.fixed_stage_weights:
            loss_fn.set_stage_weights(get_stage_weights(epoch, args.epochs))
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            epoch,
            args.epochs,
            scaler,
            amp_enabled,
        )
        val_metrics = validate(
            model, val_loader, loss_fn, device, epoch, args.epochs, amp_enabled
        )
        scheduler.step()

        print(
            f"\nEpoch {epoch}: Train Loss {train_loss:.4f} | "
            f"Val Loss {val_metrics['loss']:.4f} | "
            f"mIoU {val_metrics['mIoU']:.4f} | mF1 {val_metrics['mF1']:.4f}"
        )

        if val_metrics["mIoU"] > best_mIoU:
            best_mIoU = val_metrics["mIoU"]
            best_metrics = val_metrics
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            print(f"  -> New best mIoU {best_mIoU:.4f}, checkpoint saved.")

    if best_metrics:
        print("\n=== Best Validation Metrics ===")
        for k, v in best_metrics.items():
            print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
