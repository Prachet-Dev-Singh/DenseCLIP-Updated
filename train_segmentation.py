import torch
import torch.nn as nn
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import ADE20KDatasetPurePyTorch, train_transform
from models import DenseCLIP
import wandb

# ==========================================
# 1. EXACT SEMANTIC FPN DECODER
# ==========================================
class SemanticFPN(nn.Module):
    def __init__(self, in_channels_list=[256, 512, 1024, 2048], num_classes=150):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Conv2d(c, 256, kernel_size=1) for c in in_channels_list[:-1]
        ])
        self.deepest_projection = nn.Conv2d(in_channels_list[-1] + num_classes, 256, kernel_size=1)
        
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(256, 256, kernel_size=3, padding=1) for _ in in_channels_list
        ])
        
        self.classifier = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, features, score_map):
        deepest_feat = torch.cat([features[-1], score_map], dim=1)
        laterals = [proj(feat) for proj, feat in zip(self.projections, features[:-1])]
        laterals.append(self.deepest_projection(deepest_feat))
        
        # Top-down pathway
        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] = laterals[i] + F.interpolate(laterals[i+1], size=laterals[i].shape[2:], mode='bilinear', align_corners=False)
            
        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(len(laterals))]
        
        # MMSeg SemanticFPN: Upsample all to the 1/4 scale (which is fpn_outs[0]) and SUM them
        target_size = fpn_outs[0].shape[2:]
        out = fpn_outs[0]
        for i in range(1, len(fpn_outs)):
            out = out + F.interpolate(fpn_outs[i], size=target_size, mode='bilinear', align_corners=False)
            
        return self.classifier(out)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing pipeline on device: {device}")
    
    # --- Structural Scaling Params ---
    LOCAL_BATCH_SIZE = 16                 
    TARGET_GLOBAL_BATCH = 32              
    ACCUMULATION_STEPS = TARGET_GLOBAL_BATCH // LOCAL_BATCH_SIZE # = 2
    TOTAL_ITERATIONS = 80000              
    
    DATASET_ROOT = '/data/ADEChallengeData2016'
    CHECKPOINT_DIR = '/data/checkpoints'
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    train_dataset = ADE20KDatasetPurePyTorch(DATASET_ROOT, split='training', transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=LOCAL_BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)
    
    ADE20K_CLASSES = ('wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed', 'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth', 'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car', 'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug', 'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column', 'signboard', 'chestofdrawers', 'counter', 'sand', 'sink', 'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path', 'stairs', 'runway', 'case', 'pooltable', 'pillow', 'screendoor', 'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffeetable', 'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove', 'palm', 'kitchenisland', 'computer', 'swivelchair', 'boat', 'bar', 'arcademachine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower', 'chandelier', 'awning', 'streetlight', 'booth', 'televisionreceiver', 'airplane', 'dirttrack', 'apparel', 'pole', 'land', 'bannister', 'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van', 'ship', 'fountain', 'conveyerbelt', 'canopy', 'washer', 'plaything', 'swimmingpool', 'stool', 'barrel', 'basket', 'waterfall', 'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank', 'tradename', 'microwave', 'pot', 'animal', 'bicycle', 'lake', 'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce', 'vase', 'trafficlight', 'tray', 'ashcan', 'fan', 'pier', 'crtscreen', 'plate', 'monitor', 'bulletinboard', 'shower', 'radiator', 'glass', 'clock', 'flag')
    
    backbone = DenseCLIP(class_names=ADE20K_CLASSES, context_length=5).to(device)
    backbone.load_state_dict(torch.jit.load('/data/weights/RN50.pt', map_location='cpu').state_dict(), strict=False)
    decode_head = SemanticFPN(num_classes=150).to(device)
    
    # Freeze the text encoder completely
    for p in backbone.text_encoder.parameters():
        p.requires_grad = False
    text_enc_ids = {id(p) for p in backbone.text_encoder.parameters()}

    # ZERO WEIGHT DECAY FOR NORM LAYERS
    norm_modules = (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm, nn.SyncBatchNorm)
    decay_params, no_decay_params = [], []
    
    for m in backbone.modules():
        is_norm = isinstance(m, norm_modules)
        for p in m.parameters(recurse=False):
            if not p.requires_grad or id(p) in text_enc_ids: continue
            if is_norm: no_decay_params.append(p)
            else: decay_params.append(p)

    optimizer_grouped_parameters = [
        {'params': no_decay_params, 'weight_decay': 0.0, 'lr': 1e-5},
        {'params': decay_params, 'weight_decay': 0.0001, 'lr': 1e-5},
        {'params': decode_head.parameters(), 'weight_decay': 0.0001, 'lr': 1e-4}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
    
    # EXACT MMSEG WARMUP & POLY DECAY WITH MIN_LR
    def get_lr_multiplier(step):
        warmup_iters = 1500
        total_iters = TOTAL_ITERATIONS
        power = 0.9
        min_lr_ratio = 1e-6 / 1e-4 
        
        if step < warmup_iters:
            return (1e-6/1e-4) + (1.0 - (1e-6/1e-4)) * (step / warmup_iters)
        else:
            progress = (step - warmup_iters) / (total_iters - warmup_iters)
            decay = (1.0 - progress) ** power
            return (1.0 - min_lr_ratio) * decay + min_lr_ratio

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_multiplier)
    
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    scaler = torch.amp.GradScaler('cuda')
    
    current_iter = 0
    optimizer.zero_grad()
    backbone.train()
    decode_head.train()

    print("🚀 Commencing 80k mixed-precision DenseCLIP training run...")
    wandb.init(
        project="denseclip-ade20k-thesis",
        name="denseclip-resnet50-l4-run",
        config={
            "total_iterations": TOTAL_ITERATIONS,
            "global_batch_size": TARGET_GLOBAL_BATCH,
            "initial_lr_head": 1e-4,
            "initial_lr_backbone": 1e-5,
            "gpu": "NVIDIA L4"
        }
    )
    
    while current_iter < TOTAL_ITERATIONS:
        for images, masks in train_loader:
            if current_iter >= TOTAL_ITERATIONS: break
                
            images, masks = images.to(device), masks.to(device)
            
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                features, score_map = backbone(images)
                outputs = decode_head(features, score_map)
                
                outputs = F.interpolate(outputs, size=masks.shape[1:], mode='bilinear', align_corners=False)
                loss_task = criterion(outputs, masks)
                
                # FIX ISSUE 7: score_map is now raw similarity, applying temperature / 0.07 exactly once here
                gt_downsampled = F.interpolate(masks.float().unsqueeze(1), size=score_map.shape[2:], mode='nearest').squeeze(1).long()
                loss_aux = criterion(score_map / 0.07, gt_downsampled)
                
                # FIX ISSUE 5: Using equal weighting (1.0) for the identity pixel-text head loss
                loss = (loss_task + loss_aux) / ACCUMULATION_STEPS
            
            scaler.scale(loss).backward()
            
            if (current_iter + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            # FIX ISSUE 6: Stepping every iteration to accurately align with gradient accumulation steps
            scheduler.step()
            current_iter += 1
            
            if current_iter % 100 == 0:
                print(f"Iter {current_iter}/{TOTAL_ITERATIONS} | Task: {loss_task.item():.4f} | Aux: {loss_aux.item():.4f} | LR (Head): {optimizer.param_groups[2]['lr']:.6f}")
                
                wandb.log({
                    "task_loss": loss_task.item(),
                    "aux_loss": loss_aux.item(),
                    "combined_unscaled_loss": loss.item() * ACCUMULATION_STEPS,
                    "learning_rate_head": optimizer.param_groups[2]['lr']
                }, step=current_iter)
                print(f"Iter {current_iter}/{TOTAL_ITERATIONS} | Task: {loss_task.item():.4f} | Aux: {loss_aux.item():.4f} | LR (Head): {optimizer.param_groups[2]['lr']:.6f}")

            if current_iter % 10000 == 0:
                torch.save({
                    'iteration': current_iter,
                    'backbone': backbone.state_dict(),
                    'decode_head': decode_head.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, os.path.join(CHECKPOINT_DIR, f"denseclip_iter_{current_iter}.pth"))

if __name__ == '__main__':
    main()