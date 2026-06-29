# DenseCLIP Training Runbook — Institute Server

**Task:** Replicate DenseCLIP (ResNet-50, ADE20K) from the paper:  
*"DenseCLIP: Language-Guided Dense Prediction with Context-Aware Prompting"*  
**Target:** mIoU = 43.5% (single-scale) on ADE20K validation set  
**Estimated time:** ~18–24 hours on a single A100 / V100 32GB GPU

---

## 0. Prerequisites

### Hardware Required
- GPU with ≥ 24GB VRAM (A100 40GB, V100 32GB, or RTX 3090/4090 recommended)
- ≥ 32GB system RAM
- ≥ 100GB free disk space

### Check GPU before starting
```bash
nvidia-smi
# Confirm VRAM ≥ 24GB and driver version ≥ 470
```

---

## 1. Environment Setup

### 1.1 Create conda environment
```bash
conda create -n denseclip python=3.9 -y
conda activate denseclip
```

### 1.2 Install PyTorch (CUDA 11.8 — adjust if server uses different CUDA)
```bash
# Check server CUDA version first:
nvcc --version

# For CUDA 11.8:
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
```

### 1.3 Install remaining dependencies
```bash
pip install \
    albumentations==1.3.1 \
    opencv-python-headless==4.8.0.76 \
    wandb==0.15.12 \
    matplotlib==3.7.2 \
    numpy==1.24.3 \
    Pillow==10.0.0 \
    tqdm==4.65.0 \
    ftfy \
    regex
```

### 1.4 Verify installation
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# Should print: 2.0.1, True, and your GPU name
```

---

## 2. Code Setup

### 2.1 Clone the repository
```bash
cd ~
git clone https://github.com/Prachet-Dev-Singh/DenseCLIP-Updated.git
cd DenseCLIP-Updated
```

### 2.2 Project structure — verify these files exist
```
DenseCLIP-Updated/
├── train_segmentation.py      # Main training script
├── eval_segmentation.py       # Evaluation script
├── dataset.py                 # ADE20K dataset loader
└── models/
    ├── __init__.py
    ├── denseclip.py           # DenseCLIP model
    ├── models.py              # CLIP backbone components
    └── utils.py               # Tokenizer utilities
```

```bash
# Verify all files present
ls -la
ls -la models/
```

---

## 3. Dataset Download

### 3.1 Download ADE20K
```bash
mkdir -p /data/ADEChallengeData2016
cd /data

# Download from MIT
wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
unzip ADEChallengeData2016.zip
rm ADEChallengeData2016.zip
```

### 3.2 Verify dataset structure
```bash
ls /data/ADEChallengeData2016/
# Should show: images/  annotations/

ls /data/ADEChallengeData2016/images/
# Should show: training/  validation/

# Count training images (should be ~20,210)
ls /data/ADEChallengeData2016/images/training/ | wc -l

# Count validation images (should be 2,000)
ls /data/ADEChallengeData2016/images/validation/ | wc -l
```

---

## 4. CLIP Weights Download

### 4.1 Download OpenAI CLIP RN50 weights
```bash
mkdir -p /data/weights
cd /data/weights

# Download CLIP RN50 JIT model
wget -O RN50.pt "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358f22c8e4c2088c2d72ab4e7753/RN50.pt"

# Verify file size (should be ~102MB)
ls -lh /data/weights/RN50.pt
```

### 4.2 Verify weights load correctly
```bash
cd ~/DenseCLIP-Updated
python -c "
import torch
model = torch.jit.load('/data/weights/RN50.pt', map_location='cpu')
keys = list(model.state_dict().keys())
print(f'Total CLIP keys: {len(keys)}')
print(f'Sample keys: {keys[:5]}')
print('✅ CLIP weights loaded successfully')
"
```

---

## 5. Code Changes — Apply Before Training

These are the exact changes needed for paper-faithful training. Edit `train_segmentation.py` directly.

### 5.1 Optimizer Fix (Two-Speed Brain)

Find this block in `train_segmentation.py` (around line 105):
```python
# OLD — DELETE THIS ENTIRE BLOCK:
norm_modules = (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm, nn.SyncBatchNorm)
decay_params, no_decay_params = [], []
for m in backbone.modules():
    ...
optimizer_grouped_parameters = [
    {'params': no_decay_params, 'weight_decay': 0.0, 'lr': 1e-5},
    {'params': decay_params, 'weight_decay': 0.0001, 'lr': 1e-5},
    {'params': decode_head.parameters(), 'weight_decay': 0.0001, 'lr': 1e-4}
]
```

Replace with:
```python
# NEW — Paper-faithful two-speed optimizer
# Reference: official config uses lr_mult=0.1 for backbone, base lr=1e-4 for new layers
norm_modules = (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm, nn.SyncBatchNorm)
lr_1e5_decay, lr_1e5_no_decay = [], []
lr_1e4_decay, lr_1e4_no_decay = [], []

# Identify pre-trained visual backbone parameters
pretrained_visual_ids = {id(p) for p in backbone.backbone.parameters()}

# Sort backbone parameters by whether they are pre-trained or new
for name, m in backbone.named_modules():
    is_norm = isinstance(m, norm_modules)
    for p_name, p in m.named_parameters(recurse=False):
        if not p.requires_grad or id(p) in text_enc_ids:
            continue
        if id(p) in pretrained_visual_ids:
            # Pre-trained visual layers → slow lr=1e-5
            if is_norm or p_name == 'bias': lr_1e5_no_decay.append(p)
            else: lr_1e5_decay.append(p)
        else:
            # New DenseCLIP layers (context_decoder, prompts, gamma) → fast lr=1e-4
            if is_norm or p_name == 'bias': lr_1e4_no_decay.append(p)
            else: lr_1e4_decay.append(p)

# FPN decoder head → fast lr=1e-4
for name, m in decode_head.named_modules():
    is_norm = isinstance(m, norm_modules)
    for p_name, p in m.named_parameters(recurse=False):
        if is_norm or p_name == 'bias': lr_1e4_no_decay.append(p)
        else: lr_1e4_decay.append(p)

optimizer_grouped_parameters = [
    {'params': lr_1e5_no_decay, 'weight_decay': 0.0,    'lr': 1e-5},
    {'params': lr_1e5_decay,    'weight_decay': 0.0001, 'lr': 1e-5},
    {'params': lr_1e4_no_decay, 'weight_decay': 0.0,    'lr': 1e-4},
    {'params': lr_1e4_decay,    'weight_decay': 0.0001, 'lr': 1e-4},
]
optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
```

### 5.2 Update logging (now 4 param groups instead of 3)

Find the print statement in the training loop:
```python
# OLD:
print(f"... | Head LR: {optimizer.param_groups[2]['lr']:.2e}")

# NEW:
print(f"... | Backbone LR: {optimizer.param_groups[0]['lr']:.2e} | Head LR: {optimizer.param_groups[2]['lr']:.2e}")
```

Find the wandb.log call and update:
```python
wandb.log({
    ...
    "lr_backbone": optimizer.param_groups[0]['lr'],
    "lr_head": optimizer.param_groups[2]['lr'],   # was learning_rate_head
    ...
}, step=optimizer_step)
```

### 5.3 Add weight loading diagnostic

Find the weight loading line:
```python
backbone.load_state_dict(
    torch.jit.load('/data/weights/RN50.pt', map_location='cpu').state_dict(),
    strict=False
)
```

Add immediately after:
```python
# Verify CLIP weights actually loaded — critical diagnostic
_clip_state  = torch.jit.load('/data/weights/RN50.pt', map_location='cpu').state_dict()
_model_state = backbone.state_dict()
_matched = sum(1 for k in _clip_state
               if k in _model_state and _clip_state[k].shape == _model_state[k].shape)
print(f"✅ CLIP weights matched: {_matched} / {len(_model_state)} tensors")
print(f"   Sample CLIP key:  {list(_clip_state.keys())[0]}")
print(f"   Sample model key: {list(_model_state.keys())[0]}")
del _clip_state, _model_state  # free memory
```

### 5.4 Update dataset and checkpoint paths

Find these lines and update to match server paths:
```python
DATASET_ROOT = '/data/ADEChallengeData2016'   # ← update if different on server
CHECKPOINT_DIR = '/data/checkpoints'           # ← update if different on server
```

And the weights path:
```python
backbone.load_state_dict(
    torch.jit.load('/data/weights/RN50.pt', map_location='cpu').state_dict(),
    strict=False
)
```

### 5.5 Disable WandB if no internet on server

If the server has no internet access, add this at the top of `main()`:
```python
import os
os.environ['WANDB_MODE'] = 'offline'   # logs locally, sync later
# OR to disable entirely:
os.environ['WANDB_DISABLED'] = 'true'
```

---

## 6. Verify Everything Before Training

### 6.1 Quick sanity check (run this first, takes ~2 minutes)
```bash
cd ~/DenseCLIP-Updated
python -c "
import torch
import torch.nn.functional as F
from models import DenseCLIP
from dataset import ADE20KDatasetPurePyTorch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# Test model initialises
print('Testing model init...')
model = DenseCLIP(class_names=['wall', 'sky', 'floor'], context_length=5)
print(f'  Parameters: {sum(p.numel() for p in model.parameters()):,}')

# Test forward pass
print('Testing forward pass...')
x = torch.randn(1, 3, 512, 512)
with torch.no_grad():
    feats, score_map = model(x)
print(f'  Feature shapes: {[f.shape for f in feats]}')
print(f'  Score map shape: {score_map.shape}')

# Test dataset
print('Testing dataset...')
transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)),
    ToTensorV2(),
])
ds = ADE20KDatasetPurePyTorch('/data/ADEChallengeData2016', split='training', transform=transform)
img, mask = ds[0]
print(f'  Image shape: {img.shape}, Mask shape: {mask.shape}')
print(f'  Mask unique values: {mask.unique()[:5]}')

print()
print('✅ All checks passed — safe to start training')
"
```

### 6.2 Expected output
```
Testing model init...
  Parameters: ~60,000,000
Testing forward pass...
  Feature shapes: [torch.Size([1, 256, 128, 128]), ..., torch.Size([1, 2048, 16, 16])]
  Score map shape: torch.Size([1, 3, 16, 16])
Testing dataset...
  Image shape: torch.Size([3, 512, 512])
  Mask shape: torch.Size([512, 512])
✅ All checks passed — safe to start training
```

---

## 7. Start Training

### 7.1 Create checkpoint directory
```bash
mkdir -p /data/checkpoints
```

### 7.2 Launch training (with logging to file)
```bash
cd ~/DenseCLIP-Updated

# Run training, log to file so you can monitor and it survives terminal disconnect
nohup python train_segmentation.py 2>&1 | tee /data/training_log.txt &

# Save the process ID
echo $! > /data/training_pid.txt
echo "Training started with PID: $(cat /data/training_pid.txt)"
```

### 7.3 Alternative: use tmux (recommended for long runs)
```bash
# Start a tmux session
tmux new-session -s denseclip

# Inside tmux, run:
cd ~/DenseCLIP-Updated
python train_segmentation.py 2>&1 | tee /data/training_log.txt

# Detach from tmux: press Ctrl+B, then D
# Reattach later: tmux attach-session -t denseclip
```

### 7.4 Monitor training
```bash
# Watch live log
tail -f /data/training_log.txt

# Check GPU utilisation
watch -n 5 nvidia-smi

# Check disk space
df -h /data
```

---

## 8. What to Expect During Training

### Training timeline (approximate for A100)
| Step | Time elapsed | Expected task_loss | Expected aux_loss |
|---|---|---|---|
| 1,000 | ~1.5 hrs | ~2.5 | ~3.0 |
| 5,000 | ~7 hrs | ~1.5 | ~1.5 |
| 10,000 | ~14 hrs | ~1.0 | ~1.0 |
| 20,000 | ~28 hrs | ~0.8 | ~0.8 |
| 40,000 | ~56 hrs | ~0.6 | ~0.5 |
| 80,000 | ~112 hrs | ~0.3 | ~0.3 |

> **Note:** Times above assume A100. V100 will be ~1.5× slower. RTX 3090 ~2× slower.

### Healthy training signs ✅
- `task_loss` steadily declining from ~3.5 to ~0.3
- `aux_loss` steadily declining from ~4.0 to ~0.3 (NOT stuck at ~5.01)
- `score_map_max` rising from ~0.3 to ~0.85
- `score_map_std` rising from ~0.04 to ~0.15
- No NaN losses at any point

### Warning signs ⚠️
- `aux_loss` stuck at ~5.01 for more than 5,000 steps → weight loading failed
- `task_loss` not moving after 2,000 steps → LR issue
- NaN loss → reduce batch size or check for data issues

### Stop immediately if ❌
- Any NaN loss values
- GPU OOM error
- `aux_loss` rising instead of falling after step 10k

---

## 9. Checkpoints

Checkpoints are saved automatically at:
```
/data/checkpoints/denseclip_latest.pth          # saved every 1,000 steps (rolling)
/data/checkpoints/denseclip_step_10000.pth      # permanent milestone
/data/checkpoints/denseclip_step_20000.pth
/data/checkpoints/denseclip_step_40000.pth
/data/checkpoints/denseclip_step_80000.pth      # final checkpoint
/data/checkpoints/denseclip_emergency_exit.pth  # saved on any interruption
```

### If training is interrupted, resume from latest checkpoint
Add this block to `train_segmentation.py` after model and optimizer init, before the training loop:

```python
RESUME_CHECKPOINT = '/data/checkpoints/denseclip_latest.pth'

if os.path.exists(RESUME_CHECKPOINT):
    print(f"📂 Resuming from: {RESUME_CHECKPOINT}")
    ckpt = torch.load(RESUME_CHECKPOINT, map_location=device)
    backbone.load_state_dict(ckpt['backbone'])
    decode_head.load_state_dict(ckpt['decode_head'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    optimizer_step = ckpt['optimizer_step']
    current_iter   = optimizer_step * ACCUMULATION_STEPS
    print(f"✅ Resumed at optimizer step {optimizer_step}/{TOTAL_OPT_STEPS}")
else:
    optimizer_step = 0
    current_iter   = 0
    print("🆕 Starting fresh training run")
```

---

## 10. Evaluation

Run after training completes (or at any checkpoint):

```bash
cd ~/DenseCLIP-Updated

python eval_segmentation.py \
    --checkpoint /data/checkpoints/denseclip_step_80000.pth \
    --dataset /data/ADEChallengeData2016 \
    --output_dir /data/eval_results \
    --num_vis 20
```

### Expected evaluation output
```
=================================================================
  FINAL RESULTS — Step 80000
=================================================================
  aAcc (Overall Pixel Acc) : ~80%
  mAcc (Mean Class Acc)    : ~55%
  mIoU (single-scale)      : ~41–44%
  Paper target (SS)        : 43.5%
  Gap                      : ~0 to -2%
=================================================================
```

### Evaluation outputs saved to `/data/eval_results/`
| File | Contents |
|---|---|
| `summary_card_step80000.png` | mIoU, aAcc, mAcc vs paper target |
| `per_class_iou_step80000.png` | Top-30 and bottom-30 classes |
| `iou_histogram_step80000.png` | IoU distribution across all 150 classes |
| `confusion_matrix_step80000.png` | Normalised confusion matrix (top 20 classes) |
| `visual_grid_step80000_0.png` ... | Input / GT / Prediction comparisons |

---

## 11. Troubleshooting

### OOM (Out of Memory) error
```python
# In train_segmentation.py, reduce LOCAL_BATCH_SIZE:
LOCAL_BATCH_SIZE = 8               # was 16
ACCUMULATION_STEPS = 4             # keep effective batch = 32
# Everything else stays the same
```

### Dataset not found error
```bash
# Verify the exact path structure
find /data/ADEChallengeData2016 -name "*.jpg" | head -5
find /data/ADEChallengeData2016 -name "*.png" | head -5
# Update DATASET_ROOT in train_segmentation.py to match
```

### CLIP weights not loading (matched: 0 tensors)
```bash
# Check the key names
python -c "
import torch
m = torch.jit.load('/data/weights/RN50.pt', map_location='cpu')
keys = list(m.state_dict().keys())
print('First 10 CLIP keys:')
for k in keys[:10]: print(f'  {k}')
"
# Share the output — the key prefix mismatch tells us exactly how to remap
```

### WandB login on server
```bash
# If server has internet:
wandb login  # enter your API key when prompted

# If no internet, use offline mode (add to train_segmentation.py):
import os
os.environ['WANDB_MODE'] = 'offline'
```

### Check training is actually running
```bash
# Confirm process is alive
ps aux | grep train_segmentation

# Check GPU is being used
nvidia-smi

# Check log is updating
tail -5 /data/training_log.txt
```

---

## 12. Key Hyperparameters Reference

These match the paper exactly. Do not change without reason.

| Parameter | Value | Source |
|---|---|---|
| Total optimizer steps | 80,000 | Paper config |
| Global batch size | 32 | Paper config |
| Local batch size | 16 | Our impl |
| Gradient accumulation | 2 | Our impl |
| Backbone LR | 1e-5 | `lr_mult=0.1 × 1e-4` |
| New layers LR (context_decoder, FPN) | 1e-4 | Paper base LR |
| Text encoder LR | 0 (frozen) | Paper config |
| Weight decay | 0.0001 | Paper config |
| Optimizer | AdamW | Paper config |
| LR schedule | Poly (power=0.9) | Paper config |
| Warmup steps | 1,500 | Paper config |
| Min LR | 1e-6 | Paper config |
| Aux loss weight | 1.0 | Paper Eq. 7 |
| Temperature τ | 0.07 | Paper Eq. 7 |
| Input size | 512×512 | Paper config |
| Context length (CoOp) | 5 | Paper config |
| Context decoder layers | 3 | Official config |
| Context decoder heads | 4 | Official config |

---

## 13. After Training — Files to Download

When training finishes, download these from the server:

```bash
# From your local machine (replace SERVER with actual address):
scp user@SERVER:/data/checkpoints/denseclip_step_80000.pth ./
scp -r user@SERVER:/data/eval_results/ ./
scp user@SERVER:/data/training_log.txt ./
```

Files needed for thesis:
- `denseclip_step_80000.pth` — final trained model
- `eval_results/summary_card_step80000.png` — main result figure
- `eval_results/per_class_iou_step80000.png` — per-class breakdown
- `eval_results/visual_grid_*.png` — qualitative results
- `training_log.txt` — full training history

---

*Runbook prepared for DenseCLIP thesis replication — LNMIIT Jaipur, 2026*