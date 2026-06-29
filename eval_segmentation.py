"""
DenseCLIP Evaluation Script
============================
Computes:
  - Single-scale mIoU on ADE20K validation set
  - Per-class IoU breakdown
  - Matplotlib charts (per-class IoU bars, histogram, confusion matrix, summary card)
  - Visual segmentation overlays saved as PNG grids

Usage:
  python eval_segmentation.py \
      --checkpoint /data/checkpoints/denseclip_step_80000.pth \
      --dataset    /data/ADEChallengeData2016 \
      --output_dir /data/eval_results \
      --num_vis    20
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

# =========================================================================
# ADE20K 150-class colour palette
# =========================================================================
ADE20K_PALETTE = [
    [120,120,120],[180,120,120],[6,230,230],[80,50,50],[4,200,3],
    [120,120,80],[140,140,140],[204,5,255],[230,230,230],[4,250,7],
    [224,5,255],[235,255,7],[150,5,61],[120,120,70],[8,255,51],
    [255,6,82],[143,255,140],[204,255,4],[255,51,7],[204,70,3],
    [0,102,200],[61,230,250],[255,6,51],[11,102,255],[255,7,71],
    [255,9,224],[9,7,230],[220,220,220],[255,9,92],[112,9,255],
    [8,255,214],[7,255,224],[255,184,6],[10,255,71],[255,41,10],
    [7,255,255],[224,255,8],[102,8,255],[255,61,6],[255,194,7],
    [255,122,8],[0,255,20],[255,8,41],[255,5,153],[6,51,255],
    [235,12,255],[160,150,20],[0,163,255],[140,140,140],[250,10,15],
    [20,255,0],[31,255,0],[255,31,0],[255,224,0],[153,255,0],
    [0,0,255],[255,71,0],[0,235,255],[0,173,255],[31,0,255],
    [11,200,200],[255,82,0],[0,255,245],[0,61,255],[0,255,112],
    [0,255,133],[255,0,0],[255,163,0],[255,102,0],[194,255,0],
    [0,143,255],[51,255,0],[0,82,255],[0,255,41],[0,255,173],
    [10,0,255],[173,255,0],[0,255,153],[255,92,0],[255,0,255],
    [255,0,245],[255,0,102],[255,173,0],[255,0,20],[255,184,184],
    [0,31,255],[0,255,61],[0,71,255],[255,0,204],[0,255,194],
    [0,255,82],[0,10,255],[0,112,255],[51,0,255],[0,194,255],
    [0,122,255],[0,255,163],[255,153,0],[0,255,10],[255,112,0],
    [143,255,0],[82,0,255],[163,255,0],[255,235,0],[8,184,170],
    [133,0,255],[0,255,92],[184,0,255],[255,0,31],[0,184,255],
    [0,214,255],[255,0,112],[92,255,0],[0,224,255],[112,224,255],
    [70,184,160],[163,0,255],[153,0,255],[71,255,0],[255,0,163],
    [255,204,0],[255,0,143],[0,255,235],[133,255,0],[255,0,235],
    [245,0,255],[255,0,122],[255,245,0],[10,190,212],[214,255,0],
    [0,204,255],[20,0,255],[255,255,0],[0,153,255],[0,41,255],
    [0,255,204],[41,0,255],[41,255,0],[173,0,255],[0,245,255],
    [71,0,255],[122,0,255],[0,255,184],[0,92,255],[184,255,0],
    [0,133,255],[255,214,0],[25,194,194],[102,255,0],[92,0,255],
]

ADE20K_CLASSES = (
    'wall','building','sky','floor','tree','ceiling','road','bed',
    'window pane','grass','cabinet','sidewalk','person','earth','door',
    'table','mountain','plant','curtain','chair','car','water',
    'painting','sofa','shelf','house','sea','mirror','rug','field',
    'armchair','seat','fence','desk','rock','wardrobe','lamp',
    'bathtub','railing','cushion','base','box','column','signboard',
    'chest of drawers','counter','sand','sink','skyscraper','fireplace',
    'refrigerator','grandstand','path','stairs','runway','case',
    'pool table','pillow','screen door','stairway','river','bridge',
    'bookcase','blind','coffee table','toilet','flower','book','hill',
    'bench','countertop','stove','palm','kitchen island','computer',
    'swivel chair','boat','bar','arcade machine','hovel','bus','towel',
    'light','truck','tower','chandelier','awning','street light',
    'booth','television receiver','airplane','dirt track','apparel',
    'pole','land','bannister','escalator','ottoman','bottle','buffet',
    'poster','stage','van','ship','fountain','conveyer belt','canopy',
    'washer','plaything','swimming pool','stool','barrel','basket',
    'waterfall','tent','bag','mini bike','cradle','oven','ball',
    'food','step','tank','trade name','microwave','pot','animal',
    'bicycle','lake','dishwasher','screen','blanket','sculpture',
    'hood','sconce','vase','traffic light','tray','ashcan','fan',
    'pier','crt screen','plate','monitor','bulletin board','shower',
    'radiator','glass','clock','flag'
)


# =========================================================================
# SEMANTIC FPN — must match train_segmentation.py exactly
# =========================================================================
class SemanticFPN(nn.Module):
    def __init__(self, in_channels_list=[256,512,1024,2048], num_classes=150):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Conv2d(c, 256, kernel_size=1) for c in in_channels_list[:-1]
        ])
        self.deepest_projection = nn.Conv2d(
            in_channels_list[-1] + num_classes, 256, kernel_size=1
        )
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(256, 256, kernel_size=3, padding=1) for _ in in_channels_list
        ])
        self.classifier = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, features, score_map):
        deepest_feat = torch.cat([features[-1], score_map], dim=1)
        laterals = [proj(f) for proj, f in zip(self.projections, features[:-1])]
        laterals.append(self.deepest_projection(deepest_feat))
        for i in range(len(laterals)-2, -1, -1):
            laterals[i] = laterals[i] + F.interpolate(
                laterals[i+1], size=laterals[i].shape[2:],
                mode='bilinear', align_corners=False
            )
        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(len(laterals))]
        target_size = fpn_outs[0].shape[2:]
        out = fpn_outs[0]
        for i in range(1, len(fpn_outs)):
            out = out + F.interpolate(fpn_outs[i], size=target_size,
                                      mode='bilinear', align_corners=False)
        return self.classifier(out)


# =========================================================================
# mIoU METRIC
# =========================================================================
class MeanIoU:
    def __init__(self, num_classes=150, ignore_index=255):
        self.num_classes  = num_classes
        self.ignore_index = ignore_index
        self.confusion    = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred, gt):
        mask = gt != self.ignore_index
        p = np.clip(pred[mask].astype(np.int64), 0, self.num_classes-1)
        g = gt[mask].astype(np.int64)
        idx = g * self.num_classes + p
        self.confusion += np.bincount(idx, minlength=self.num_classes**2).reshape(
            self.num_classes, self.num_classes)

    def compute(self):
        tp      = np.diag(self.confusion)
        row_sum = self.confusion.sum(axis=1)
        col_sum = self.confusion.sum(axis=0)
        denom   = tp + row_sum + col_sum - tp
        iou     = np.where(denom > 0, tp / denom, np.nan)
        miou    = float(np.nanmean(iou))
        # Overall pixel accuracy
        aAcc    = float(tp.sum() / self.confusion.sum()) if self.confusion.sum() > 0 else 0.0
        # Mean class accuracy
        acc     = np.where(row_sum > 0, tp / row_sum, np.nan)
        mAcc    = float(np.nanmean(acc))
        return iou, miou, aAcc, mAcc


# =========================================================================
# TRANSFORMS — Albumentations syntax to match dataset.py
# =========================================================================
def get_val_transform():
    # Resize shorter side to 512, preserve aspect ratio. NO CenterCrop.
    return A.Compose([
        A.SmallestMaxSize(max_size=512, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=(0.48145466, 0.4578275,  0.40821073),
                    std= (0.26862954, 0.26130258, 0.27577711)),
        ToTensorV2(),
    ])

def get_vis_transform():
    return A.Compose([
        A.SmallestMaxSize(max_size=512, interpolation=cv2.INTER_LINEAR),
        # This explicitly scales 0-255 pixels down to 0.0-1.0 to perfectly replicate T.ToTensor()
        A.Normalize(mean=(0,0,0), std=(1,1,1), max_pixel_value=255.0),
        ToTensorV2(),
    ])


# =========================================================================
# HELPERS
# =========================================================================
def colourise(seg_map):
    h, w = seg_map.shape
    rgb  = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, colour in enumerate(ADE20K_PALETTE):
        rgb[seg_map == cls_id] = colour
    return rgb

BG = '#0f1117'
PANEL = '#1a1d27'
BORDER = '#2e3247'
FG = '#e8eaf0'
MED = '#c8cdd8'
DIM = '#7a8299'
BLUE = '#5b8cf7'
ORANGE = '#f5a623'
GREEN = '#4caf82'
RED = '#e05c5c'


def dark_ax(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MED, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)


# =========================================================================
# CHART 1 — Per-class IoU bar chart (top-30 / bottom-30)
# =========================================================================
def plot_per_class_iou(per_class_iou, output_dir, step):
    valid   = ~np.isnan(per_class_iou)
    classes = np.array(ADE20K_CLASSES)[valid]
    ious    = per_class_iou[valid] * 100
    order   = np.argsort(ious)[::-1]
    classes, ious = classes[order], ious[order]

    fig, axes = plt.subplots(1, 2, figsize=(28, 13))
    fig.patch.set_facecolor(BG)
    miou_line = float(np.nanmean(per_class_iou)) * 100

    for ax, title, cls_slice, iou_slice in [
        (axes[0], 'Top 30 Classes',    classes[:30],      ious[:30]),
        (axes[1], 'Bottom 30 Classes', classes[-30:][::-1], ious[-30:][::-1]),
    ]:
        dark_ax(ax)
        colors = plt.cm.RdYlGn(iou_slice / 100.0)
        bars   = ax.barh(range(len(cls_slice)), iou_slice,
                         color=colors, edgecolor='none', height=0.7)
        ax.set_yticks(range(len(cls_slice)))
        ax.set_yticklabels(cls_slice, fontsize=8, color=MED)
        ax.set_xlabel('IoU (%)', color=MED, fontsize=10)
        ax.set_title(title, color=FG, fontsize=12, pad=8)
        ax.invert_yaxis()
        ax.axvline(miou_line, color=BLUE, linestyle='--', linewidth=1.5,
                   label=f'mIoU={miou_line:.1f}%')
        ax.axvline(43.5, color=ORANGE, linestyle=':', linewidth=1.5,
                   label='Paper=43.5%')
        ax.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=MED, fontsize=8)
        for bar, v in zip(bars, iou_slice):
            ax.text(v+0.3, bar.get_y()+bar.get_height()/2,
                    f'{v:.1f}', va='center', color=MED, fontsize=7)

    fig.suptitle(f'DenseCLIP — Per-Class IoU on ADE20K  |  Step {step}',
                 color=FG, fontsize=14)
    plt.tight_layout()
    p = os.path.join(output_dir, f'per_class_iou_step{step}.png')
    plt.savefig(p, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'  📊 per_class_iou → {p}')


# =========================================================================
# CHART 2 — IoU histogram
# =========================================================================
def plot_iou_histogram(per_class_iou, output_dir, step):
    valid = per_class_iou[~np.isnan(per_class_iou)] * 100
    miou  = float(np.nanmean(per_class_iou)) * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BG)
    dark_ax(ax)

    n, bins, patches = ax.hist(valid, bins=20, edgecolor=BG, linewidth=0.5)
    for patch, left in zip(patches, bins[:-1]):
        patch.set_facecolor(plt.cm.RdYlGn(left / 100.0))

    ax.axvline(miou,  color=BLUE,   linewidth=2, linestyle='--',
               label=f'mIoU = {miou:.2f}%')
    ax.axvline(43.5,  color=ORANGE, linewidth=2, linestyle=':',
               label='Paper target = 43.5%')
    ax.set_xlabel('IoU (%)', color=MED, fontsize=11)
    ax.set_ylabel('Number of classes', color=MED, fontsize=11)
    ax.set_title(f'IoU Distribution — {len(valid)} valid classes  |  Step {step}',
                 color=FG, fontsize=13)
    ax.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=MED, fontsize=10)

    plt.tight_layout()
    p = os.path.join(output_dir, f'iou_histogram_step{step}.png')
    plt.savefig(p, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'  📊 iou_histogram → {p}')


# =========================================================================
# CHART 3 — Normalised confusion matrix (top-20 classes)
# =========================================================================
def plot_confusion_matrix(confusion, output_dir, step, top_n=20):
    gt_counts = confusion.sum(1)
    top_idx   = np.argsort(gt_counts)[::-1][:top_n]
    sub       = confusion[np.ix_(top_idx, top_idx)].astype(float)
    row_sums  = sub.sum(1, keepdims=True)
    norm      = np.where(row_sums > 0, sub / row_sums, 0)
    labels    = [ADE20K_CLASSES[i][:13] for i in top_idx]

    fig, ax = plt.subplots(figsize=(14, 12))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)
    im = ax.imshow(norm, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.ax.tick_params(colors=MED)
    ax.set_xticks(range(top_n)); ax.set_yticks(range(top_n))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8, color=MED)
    ax.set_yticklabels(labels, fontsize=8, color=MED)
    ax.set_xlabel('Predicted', color=MED, fontsize=11)
    ax.set_ylabel('Ground Truth', color=MED, fontsize=11)
    ax.set_title(f'Normalised Confusion Matrix (Top {top_n} classes)  |  Step {step}',
                 color=FG, fontsize=13)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    plt.tight_layout()
    p = os.path.join(output_dir, f'confusion_matrix_step{step}.png')
    plt.savefig(p, dpi=120, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'  📊 confusion_matrix → {p}')


# =========================================================================
# CHART 4 — Summary card
# =========================================================================
def save_summary_card(miou, aAcc, mAcc, per_class_iou, step, output_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis('off')

    gap      = miou * 100 - 43.5
    gap_str  = f'+{gap:.2f}%' if gap >= 0 else f'{gap:.2f}%'
    gap_col  = GREEN if gap >= 0 else RED
    valid_v  = per_class_iou[~np.isnan(per_class_iou)] * 100

    lines = [
        ('DenseCLIP  |  ResNet-50  |  ADE20K', BLUE,   18, 'bold'),
        (f'Optimizer step: {step}',              DIM,    12, 'normal'),
        ('',                                     '',     8,  'normal'),
        (f'mIoU (single-scale):  {miou*100:.2f}%', FG, 22, 'bold'),
        (f'aAcc (pixel acc):     {aAcc*100:.2f}%', '#9ab3f5', 13, 'normal'),
        (f'mAcc (class acc):     {mAcc*100:.2f}%', '#9ab3f5', 13, 'normal'),
        (f'Paper target (SS):    43.5%',         '#9ab3f5', 13, 'normal'),
        (f'Gap to paper:         {gap_str}',     gap_col,   14, 'bold'),
        ('',                                     '',     8,  'normal'),
        (f'Classes with IoU > 0:    {(valid_v>0).sum()} / {len(valid_v)}',  MED, 12, 'normal'),
        (f'Classes with IoU > 40%:  {(valid_v>40).sum()} / {len(valid_v)}', MED, 12, 'normal'),
        (f'Best:   {ADE20K_CLASSES[int(np.nanargmax(per_class_iou))]}  ({np.nanmax(per_class_iou)*100:.1f}%)', MED, 11, 'normal'),
        (f'Worst:  {ADE20K_CLASSES[int(np.nanargmin(per_class_iou))]}  ({np.nanmin(per_class_iou)*100:.1f}%)', MED, 11, 'normal'),
    ]

    y = 0.95
    for text, color, size, weight in lines:
        if not text:
            y -= 0.03; continue
        ax.text(0.05, y, text, transform=ax.transAxes, color=color,
                fontsize=size, fontweight=weight, va='top', fontfamily='monospace')
        y -= size / 100 * 1.8

    # Progress bar
    BY, BX, BW, BH = 0.10, 0.05, 0.90, 0.05
    fill = min(miou * 100 / 60.0, 1.0)
    targ = 43.5 / 60.0
    ax.add_patch(mpatches.FancyBboxPatch(
        (BX, BY), BW, BH, boxstyle='round,pad=0.005',
        facecolor=PANEL, edgecolor=BORDER, linewidth=1,
        transform=ax.transAxes, clip_on=False))
    ax.add_patch(mpatches.FancyBboxPatch(
        (BX, BY), BW*fill, BH, boxstyle='round,pad=0.005',
        facecolor=BLUE, edgecolor='none',
        transform=ax.transAxes, clip_on=False))
    ax.plot([BX + BW*targ, BX + BW*targ], [BY, BY+BH+0.02],
            color=ORANGE, linewidth=2, transform=ax.transAxes, clip_on=False)
    ax.text(BX + BW*targ, BY-0.03, 'paper',
            transform=ax.transAxes, color=ORANGE, fontsize=8, ha='center')

    plt.tight_layout()
    p = os.path.join(output_dir, f'summary_card_step{step}.png')
    plt.savefig(p, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'  📋 summary_card → {p}')


# =========================================================================
# CHART 5 — Visual segmentation grids (input | GT | prediction)
# =========================================================================
def save_visual_grid(vis_samples, output_dir, step, grid_idx=0):
    n    = len(vis_samples)
    fig  = plt.figure(figsize=(12, n * 4 + 0.8))
    fig.patch.set_facecolor(BG)
    gs   = gridspec.GridSpec(n+1, 3, figure=fig,
                             hspace=0.04, wspace=0.02,
                             top=0.96, bottom=0.01)

    for col, title in enumerate(['Input Image', 'Ground Truth', 'Prediction']):
        ax = fig.add_subplot(gs[0, col])
        ax.text(0.5, 0.5, title, ha='center', va='center',
                color=FG, fontsize=11, fontweight='bold',
                transform=ax.transAxes)
        ax.axis('off')
        ax.set_facecolor(BG)

    for row, s in enumerate(vis_samples):
        gt_rgb   = colourise(s['gt'])
        pred_rgb = colourise(s['pred'])
        for col, rgb in enumerate([s['image'], gt_rgb, pred_rgb]):
            ax = fig.add_subplot(gs[row+1, col])
            ax.imshow(rgb)
            ax.axis('off')
            if col == 2:
                mask = s['gt'] != 255
                if mask.sum() > 0:
                    acc = (s['pred'][mask] == s['gt'][mask]).mean()
                    ax.set_title(f'px-acc {acc*100:.0f}%',
                                 color='#9ab3f5', fontsize=7, pad=2)

    fig.suptitle(f'DenseCLIP Segmentation  |  Step {step}',
                 color=FG, fontsize=13, y=0.99)
    p = os.path.join(output_dir, f'visual_grid_step{step}_{grid_idx}.png')
    plt.savefig(p, dpi=110, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'  🖼️  visual_grid_{grid_idx} → {p}')


# =========================================================================
# MAIN
# =========================================================================
def evaluate(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    print(f"\n{'='*65}")
    print(f"  DenseCLIP Evaluation")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Output dir : {args.output_dir}")
    print(f"  Device     : {device}")
    print(f"{'='*65}\n")

    ckpt     = torch.load(args.checkpoint, map_location=device)
    opt_step = ckpt.get('optimizer_step', 'unknown')
    print(f"✅ Checkpoint loaded  |  optimizer_step = {opt_step}")

    from models import DenseCLIP
    backbone    = DenseCLIP(class_names=ADE20K_CLASSES, context_length=5).to(device)
    decode_head = SemanticFPN(num_classes=150).to(device)
    backbone.load_state_dict(ckpt['backbone'])
    decode_head.load_state_dict(ckpt['decode_head'])
    backbone.eval()
    decode_head.eval()
    print("✅ Weights restored\n")

    from dataset import ADE20KDatasetPurePyTorch
    val_dataset = ADE20KDatasetPurePyTorch(
        args.dataset, split='validation', transform=get_val_transform())
    vis_dataset = ADE20KDatasetPurePyTorch(
        args.dataset, split='validation', transform=get_vis_transform())

    # batch_size=1 required — images have variable width after aspect-ratio resize
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            num_workers=4, pin_memory=True)
    print(f"✅ Validation set: {len(val_dataset)} images\n")

    metric      = MeanIoU(num_classes=150, ignore_index=255)
    vis_samples = []
    vis_indices = set(range(0, min(args.num_vis * 4, len(val_dataset)), 4))

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(val_loader):
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                features, score_map = backbone(images.to(device))
                logits = decode_head(features, score_map)
            logits = F.interpolate(logits.float(), size=masks.shape[1:],
                                   mode='bilinear', align_corners=False)
            preds = logits.argmax(1).cpu().numpy()
            gts   = masks.cpu().numpy()

            for i, (pred, gt) in enumerate(zip(preds, gts)):
                metric.update(pred, gt)
                global_idx = batch_idx   # batch_size=1, so global_idx == batch_idx
                if global_idx in vis_indices and len(vis_samples) < args.num_vis:
                    raw_img, _ = vis_dataset[global_idx]
                    vis_samples.append({
                        'image': (raw_img.permute(1,2,0).numpy()*255).astype(np.uint8),
                        'gt':    gt,
                        'pred':  pred,
                    })

            if (batch_idx+1) % 50 == 0:
                _, m, a, c = metric.compute()
                print(f"  [{batch_idx+1:4d}/{len(val_loader)}]  "
                      f"mIoU: {m*100:.2f}% | aAcc: {a*100:.2f}% | mAcc: {c*100:.2f}%")

    per_class_iou, miou, aAcc, mAcc = metric.compute()

    print(f"\n{'='*65}")
    print(f"  FINAL RESULTS — Step {opt_step}")
    print(f"{'='*65}")
    print(f"  aAcc (Overall Pixel Acc) : {aAcc*100:.2f}%")
    print(f"  mAcc (Mean Class Acc)    : {mAcc*100:.2f}%")
    print(f"  mIoU (single-scale)      : {miou*100:.2f}%")
    print(f"  Paper target (SS)        : 43.5%")
    print(f"  Gap                      : {(miou*100-43.5):+.2f}%")
    print(f"{'='*65}\n")

    valid_mask = ~np.isnan(per_class_iou)
    sorted_idx = np.argsort(np.where(valid_mask, per_class_iou, -1))[::-1]
    print("  Top 15 classes:")
    for i in sorted_idx[:15]:
        print(f"    {ADE20K_CLASSES[i]:28s}  {per_class_iou[i]*100:5.1f}%")
    print("\n  Bottom 15 classes:")
    for i in sorted_idx[-15:]:
        v = per_class_iou[i]
        print(f"    {ADE20K_CLASSES[i]:28s}  {v*100:5.1f}%" if not np.isnan(v)
              else f"    {ADE20K_CLASSES[i]:28s}     nan")

    print(f"\n  Saving charts ...\n")
    plot_per_class_iou(per_class_iou, args.output_dir, opt_step)
    plot_iou_histogram(per_class_iou, args.output_dir, opt_step)
    plot_confusion_matrix(metric.confusion, args.output_dir, opt_step)
    save_summary_card(miou, aAcc, mAcc, per_class_iou, opt_step, args.output_dir)
    for g_i in range(0, len(vis_samples), 5):
        save_visual_grid(vis_samples[g_i:g_i+5], args.output_dir,
                         opt_step, grid_idx=g_i//5)

    miou_pct = miou * 100
    print(f"\n{'='*65}")
    print(f"  THESIS DECISION GUIDE")
    print(f"{'='*65}")
    if miou_pct >= 41.0:
        print(f"  ✅  {miou_pct:.2f}% — COMPLETE. Within thesis range of 43.5%.")
        print(f"      Proceed to writing up results.")
    elif miou_pct >= 37.0:
        print(f"  ⚠️   {miou_pct:.2f}% — BELOW PACE.")
        print(f"      Check weight loading keys. Consider one more run.")
    else:
        print(f"  ❌  {miou_pct:.2f}% — STOP AND FIX.")
        print(f"      Backbone likely has random weights. Fix and rerun.")
    print(f"{'='*65}\n")

    return miou, per_class_iou


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default='/data/checkpoints/denseclip_step_80000.pth')
    parser.add_argument('--dataset',    type=str,
                        default='/data/ADEChallengeData2016')
    parser.add_argument('--output_dir', type=str,
                        default='/data/eval_results')
    parser.add_argument('--num_vis',    type=int, default=20)
    parser.add_argument('--device',     type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    evaluate(parser.parse_args())