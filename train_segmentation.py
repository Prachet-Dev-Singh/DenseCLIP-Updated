import torch
import torch.nn as nn
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, PolynomialLR, SequentialLR
from dataset import ADE20KDatasetPurePyTorch, train_transform
from models import DenseCLIP

# ==========================================
# 1. PURE PYTORCH DENSECLIP FPN DECODER (Issue 7 Fixed)
# ==========================================
class LightweightSemanticFPN(nn.Module):
    def __init__(self, in_channels_list=[256, 512, 1024, 2048], num_classes=150):
        super().__init__()
        
        self.projections = nn.ModuleList([
            nn.Conv2d(c, 256, kernel_size=1) for c in in_channels_list[:-1]
        ])
        
        self.deepest_projection = nn.Conv2d(in_channels_list[-1] + num_classes, 256, kernel_size=1)
        
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(256, 256, kernel_size=3, padding=1) for _ in in_channels_list
        ])
        
        # Output aggregates all 4 scales (256 channels each = 1024)
        self.classifier = nn.Conv2d(256 * 4, num_classes, kernel_size=1)

    def forward(self, features, score_map):
        deepest_feat = torch.cat([features[-1], score_map], dim=1)
        
        laterals = [proj(feat) for proj, feat in zip(self.projections, features[:-1])]
        laterals.append(self.deepest_projection(deepest_feat))
        
        # Top-down pathway
        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] = laterals[i] + F.interpolate(
                laterals[i+1], size=laterals[i].shape[2:], mode='bilinear', align_corners=False
            )
            
        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(len(laterals))]
        
        finest_size = fpn_outs[0].shape[2:]
        for i in range(1, len(fpn_outs)):
            fpn_outs[i] = F.interpolate(fpn_outs[i], size=finest_size, mode='bilinear', align_corners=False)
            
        out = torch.cat(fpn_outs, dim=1)
        return self.classifier(out)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing pipeline on device: {device}")
    
    # --- Structural Scaling Params (Issue 3 Fixed) ---
    LOCAL_BATCH_SIZE = 16                 
    TARGET_GLOBAL_BATCH = 16              
    ACCUMULATION_STEPS = TARGET_GLOBAL_BATCH // LOCAL_BATCH_SIZE 
    TOTAL_ITERATIONS = 80000              
    
    DATASET_ROOT = '/data/ADEChallengeData2016'
    CHECKPOINT_DIR = '/data/checkpoints'
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    train_dataset = ADE20KDatasetPurePyTorch(DATASET_ROOT, split='training', transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=LOCAL_BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)
    
    ADE20K_CLASSES = (
        'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed', 'windowpane', 'grass', 
        'cabinet', 'sidewalk', 'person', 'earth', 'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 
        'car', 'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug', 'field', 
        'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp', 'bathtub', 'railing', 'cushion', 
        'base', 'box', 'column', 'signboard', 'chestofdrawers', 'counter', 'sand', 'sink', 'skyscraper', 'fireplace', 
        'refrigerator', 'grandstand', 'path', 'stairs', 'runway', 'case', 'pooltable', 'pillow', 'screendoor', 'stairway', 
        'river', 'bridge', 'bookcase', 'blind', 'coffeetable', 'toilet', 'flower', 'book', 'hill', 'bench', 
        'countertop', 'stove', 'palm', 'kitchenisland', 'computer', 'swivelchair', 'boat', 'bar', 'arcademachine', 'hovel', 
        'bus', 'towel', 'light', 'truck', 'tower', 'chandelier', 'awning', 'streetlight', 'booth', 'televisionreceiver', 
        'airplane', 'dirttrack', 'apparel', 'pole', 'land', 'bannister', 'escalator', 'ottoman', 'bottle', 'buffet', 
        'poster', 'stage', 'van', 'ship', 'fountain', 'conveyerbelt', 'canopy', 'washer', 'plaything', 'swimmingpool', 
        'stool', 'barrel', 'basket', 'waterfall', 'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball', 
        'food', 'step', 'tank', 'tradename', 'microwave', 'pot', 'animal', 'bicycle', 'lake', 'dishwasher', 
        'screen', 'blanket', 'sculpture', 'hood', 'sconce', 'vase', 'trafficlight', 'tray', 'ashcan', 'fan', 
        'pier', 'crtscreen', 'plate', 'monitor', 'bulletinboard', 'shower', 'radiator', 'glass', 'clock', 'flag'
    )

    # ==========================================
    # 2. MODEL INITIALIZATION (Issue 1 Fixed)
    # ==========================================
    backbone = DenseCLIP(class_names=ADE20K_CLASSES, context_length=5).to(device)
    clip_state_dict = torch.jit.load('/data/weights/RN50.pt', map_location='cpu').state_dict()
    backbone.load_state_dict(clip_state_dict, strict=False)
    
    decode_head = LightweightSemanticFPN(num_classes=150).to(device)
    
    # ==========================================
    # 3. OPTIMIZER & SCHEDULER (Issues 4 & 6 Fixed)
    # ==========================================
    # Freeze the text encoder completely
    for p in backbone.text_encoder.parameters():
        p.requires_grad = False
        
    text_enc_ids = {id(p) for p in backbone.text_encoder.parameters()}
    
    optimizer_grouped_parameters = [
        {'params': [p for p in backbone.parameters() if id(p) not in text_enc_ids], 'lr': 1e-5},   
        {'params': decode_head.parameters(), 'lr': 1e-4}  
    ]
    
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=0.0001)
    
    # Linear Warmup + Poly Decay
    warmup_scheduler = LinearLR(optimizer, start_factor=1e-6/1e-4, end_factor=1.0, total_iters=1500)
    poly_scheduler = PolynomialLR(optimizer, total_iters=TOTAL_ITERATIONS - 1500, power=0.9)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, poly_scheduler], milestones=[1500])
    
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    scaler = torch.amp.GradScaler('cuda')
    
    # ==========================================
    # 4. TRAINING LOOP (Issue 2 Fixed)
    # ==========================================
    current_iter = 0
    optimizer.zero_grad()
    
    print("🚀 Commencing 80k mixed-precision DenseCLIP training run...")
    backbone.train()
    decode_head.train()
    
    while current_iter < TOTAL_ITERATIONS:
        for images, masks in train_loader:
            if current_iter >= TOTAL_ITERATIONS:
                break
                
            images = images.to(device)
            masks = masks.to(device)
            
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                features, score_map = backbone(images)
                outputs = decode_head(features, score_map)
                
                outputs = F.interpolate(outputs, size=masks.shape[1:], mode='bilinear', align_corners=False)
                loss_task = criterion(outputs, masks)
                
                # AUXILIARY PIXEL-TEXT MATCHING LOSS
                gt_downsampled = F.interpolate(
                    masks.float().unsqueeze(1), size=score_map.shape[2:], mode='nearest'
                ).squeeze(1).long()
                loss_aux = criterion(score_map / 0.07, gt_downsampled)
                
                loss = (loss_task + loss_aux) / ACCUMULATION_STEPS
            
            scaler.scale(loss).backward()
            
            if (current_iter + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                
            current_iter += 1
            
            if current_iter % 100 == 0:
                print(f"Iter {current_iter}/{TOTAL_ITERATIONS} | Task Loss: {loss_task.item():.4f} | Aux Loss: {loss_aux.item():.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

            if current_iter % 10000 == 0:
                checkpoint_path = os.path.join(CHECKPOINT_DIR, f"denseclip_iter_{current_iter}.pth")
                torch.save({
                    'iteration': current_iter,
                    'backbone': backbone.state_dict(),
                    'decode_head': decode_head.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, checkpoint_path)

    print("✅ Training Complete.")

if __name__ == '__main__':
    main()