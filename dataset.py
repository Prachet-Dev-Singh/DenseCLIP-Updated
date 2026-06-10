import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ADE20KDatasetPurePyTorch(Dataset):
    def __init__(self, root_dir, split='training', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        self.image_dir = os.path.join(root_dir, 'images', split)
        self.mask_dir = os.path.join(root_dir, 'annotations', split)
        
        self.images = [f for f in sorted(os.listdir(self.image_dir)) if f.endswith('.jpg')]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        mask_name = img_name.replace('.jpg', '.png')
        
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        # Read with OpenCV for rapid Albumentations operations
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Convert mask to numpy array
        mask_np = mask.astype(np.int64)
        
        # Match your logic: Shift labels down by 1 (1-150 becomes 0-149)
        mask_np = mask_np - 1 
        mask_np[mask_np == -1] = 255  # Map background to 255 (ignore_index)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask_np)
            image = augmented['image']
            mask_tensor = augmented['mask'].long()
            
        return image, mask_tensor

# The exact publication pipeline matching standard mmseg configurations
train_transform = A.Compose([
    A.RandomScale(scale_limit=(-0.5, 1.0), p=1.0),
    A.PadIfNeeded(min_height=512, min_width=512, 
                  border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=255),
    A.RandomCrop(height=512, width=512, p=1.0),
    A.HorizontalFlip(p=0.5),
    
    # RESTORED: This accurately replicates MMSegmentation's 'PhotoMetricDistortion'
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    
    A.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    ),
    ToTensorV2()
])

val_transform = A.Compose([
    # Validation uses SmallestMaxSize to resize the shorter side to 512, maintaining aspect ratio
    A.SmallestMaxSize(max_size=512),
    # Pad if the longer side is somehow smaller than 512 to ensure batching works
    A.PadIfNeeded(min_height=512, min_width=512,
                  border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=255),
    A.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    ),
    ToTensorV2()
])