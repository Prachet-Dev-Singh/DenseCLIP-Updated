import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import PolynomialLR
from dataset import ADE20KDatasetPurePyTorch, train_transform
from models import DenseCLIP
from tqdm import tqdm

# ==========================================
# 1. PURE PYTORCH DENSECLIP FPN DECODER
# ==========================================
class LightweightSemanticFPN(nn.Module):
    def __init__(self, in_channels_list=[256, 512, 1024, 2048], num_classes=150):
        super().__init__()
        
        # Standard 1x1 projections for the lower-level features
        self.projections = nn.ModuleList([
            nn.Conv2d(c, 256, kernel_size=1) for c in in_channels_list[:-1]
        ])
        
        # PAPER ALIGNMENT: The deepest feature (2048) gets concatenated with the 
        # language score_map (150 classes) = 2198 input channels
        self.deepest_projection = nn.Conv2d(in_channels_list[-1] + num_classes, 256, kernel_size=1)
        
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, features, score_map):
        # 1. Concatenate Text-Vision Prior to Deepest Features
        deepest_feat = torch.cat([features[-1], score_map], dim=1)
        
        # 2. Build FPN Top-Down Pathway
        out = self.deepest_projection(deepest_feat)
        for i in range(len(features) - 2, -1, -1):
            proj_feat = self.projections[i](features[i])
            out = torch.nn.functional.interpolate(
                out, 
                size=proj_feat.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
            out = out + proj_feat
            
        return self.classifier(out)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing pipeline on device: {device}")
    
    # --- Structural Scaling Params (Aligned with Paper) ---
    LOCAL_BATCH_SIZE = 16                 
    TARGET_GLOBAL_BATCH = 16              
    ACCUMULATION_STEPS = TARGET_GLOBAL_BATCH // LOCAL_BATCH_SIZE 
    TOTAL_ITERATIONS = 160000              
    
    # Paths updated for the Modal attached volume
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
    # 2. MODEL INITIALIZATION & WEIGHT LOADING
    # ==========================================
    backbone = DenseCLIP(class_names=ADE20K_CLASSES, context_length=10).to(device)
    clip_state_dict = torch.jit.load('/data/weights/RN50.pt', map_location='cpu').state_dict()
    backbone.load_state_dict(clip_state_dict, strict=False)
    
    decode_head = LightweightSemanticFPN(num_classes=150).to(device)
    
    # ==========================================
    # 3. OPTIMIZER, SCHEDULER, AND AMP SCALER
    # ==========================================
    optimizer_grouped_parameters = [
        {'params': backbone.parameters(), 'lr': 1e-5},   
        {'params': decode_head.parameters(), 'lr': 1e-4}  
    ]
    
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=0.0001)
    scheduler = PolynomialLR(optimizer, total_iters=TOTAL_ITERATIONS, power=0.9)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    
    scaler = torch.amp.GradScaler('cuda')
    
    # ==========================================
    # 4. MIXED PRECISION TRAINING LOOP
    # ==========================================
    current_iter = 0
    optimizer.zero_grad()
    
    print("🚀 Commencing full mixed-precision training run...")
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
                
                # FIX: Pass BOTH the features and the score_map into the decoder
                outputs = decode_head(features, score_map)
                
                outputs = torch.nn.functional.interpolate(
                    outputs, 
                    size=masks.shape[1:], 
                    mode='bilinear', 
                    align_corners=False
                )
                
                loss = criterion(outputs, masks) / ACCUMULATION_STEPS
            
            scaler.scale(loss).backward()
            
            if (current_iter + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                
            current_iter += 1
            
            # Print update
            if current_iter % 100 == 0:
                print(f"Iteration {current_iter}/{TOTAL_ITERATIONS} | Loss: {loss.item() * ACCUMULATION_STEPS:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

            # FIX: Save Checkpoint every 10,000 iterations to permanent storage
            if current_iter % 10000 == 0:
                checkpoint_path = os.path.join(CHECKPOINT_DIR, f"denseclip_iter_{current_iter}.pth")
                torch.save({
                    'iteration': current_iter,
                    'backbone': backbone.state_dict(),
                    'decode_head': decode_head.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, checkpoint_path)
                print(f"💾 Saved permanent checkpoint to: {checkpoint_path}")

    print("✅ Training Complete. Model is fully optimized.")

if __name__ == '__main__':
    main()