# Migration Plan: Transitioning to MobileNetV4

This plan details the steps, architectural shifts, and file changes required to replace the heavy **DepthAnythingV2** and **ViT** components in **WAFT-Stereo** with highly-efficient **MobileNetV4** backbones (via `timm`).

---

## Target Goals
1. **Reduce Memory Footprint & FLOPs**: MobileNetV4 is highly optimized for fast inference on hardware/edge platforms.
2. **Replace Feature Encoder**: Replace the DepthAnythingV2 / DINOv3 backbone with `MNv4Encoder`.
3. **Replace Iterative Refinement Decoder**: Replace the ViT-Small + DPT-based updater with a lightweight `MNv4Iter` decoder.

---

## Migration Architecture Comparison

```
Current Architecture (WAFT-Stereo):
[Left/Right Images]
    --> DepthAnythingV2 (ViT-L/B/S) + LoRA + DPT head   [Encoder -- shared weights, run twice]
    --> Classification Module (ViT-S + DPT)               [Initial coarse disparity]
    --> Recurrent Updater (ViT-S + DPT + ResNet blocks)   [Iterative refinement, T-1 steps]
    --> Disparity Prediction (Convex Upsample)

Target MobileNetV4 Architecture:
[Left/Right Images]
    --> MobileNetV4 (Conv/Hybrid) + LoRA + DPT head      [Encoder -- shared weights, run twice]
    --> MNv4Iter Classification Module                    [Initial coarse disparity]
    --> MNv4Iter Recurrent Updater + ResNet blocks        [Iterative refinement, T-1 steps]
    --> Disparity Prediction (Convex Upsample)
```

**Important note on Siamese Encoder**: MobileNetV4 is a single-image backbone. WAFT-Stereo uses shared weights -- the same `MNv4Encoder` is called twice independently for the left and right images (Siamese style). This is required so that the downstream Warping module can match features from both views within the same embedding space.

---

## Correct Channel Dimensions (timm features_only=True)

Always verify with `model.feature_info.channels()` before hardcoding any values. The table below is a reference from the MNv4 paper and timm source.

| Model | Stage 1 (/4) | Stage 2 (/8) | Stage 3 (/16) | Stage 4 (/32) | Params |
|-------|-------------|-------------|--------------|--------------|--------|
| mobilenetv4_conv_small | 32 | 32 | 64 | 96 | 3.8M |
| mobilenetv4_conv_medium | 48 | 80 | 160 | 256 | 9.7M |
| mobilenetv4_conv_large | 48 | 96 | 192 | 512 | 32.6M |
| mobilenetv4_hybrid_medium | 48 | 80 | 160 | 256 | 11M |
| mobilenetv4_hybrid_large | 48 | 96 | 192 | 512 | 37.8M |

```python
# Run this before hardcoding any channel dimensions
import timm
for name in ['mobilenetv4_conv_small', 'mobilenetv4_conv_medium', 'mobilenetv4_hybrid_medium']:
    m = timm.create_model(name, features_only=True, out_indices=(1, 2, 3, 4))
    print(f"{name}: {m.feature_info.channels()}")
```

---

## Step-by-Step Implementation

### Step 1: Create the MobileNetV4 Feature Encoder (MNv4Encoder)

Create `model/encoder/mnv4.py`. This wraps timm's MobileNetV4 backbone as a drop-in replacement for `DAv2Encoder` / `DINOv3Encoder`.

- **Supported models**: `mobilenetv4_conv_small`, `mobilenetv4_conv_medium`, `mobilenetv4_conv_large`, `mobilenetv4_hybrid_medium`, `mobilenetv4_hybrid_large`.
- **Multi-scale features**: Use `features_only=True` with `out_indices=(1, 2, 3, 4)` to extract hierarchical feature maps at strides /4, /8, /16, /32.
- **LoRA**: Only applicable to `hybrid` variants that contain attention layers. For pure `conv` variants, freeze the encoder and train only the DPT head instead.
- **Grayscale input**: Broadcast 1-channel input to 3 channels inside `forward()` before passing to the pretrained backbone.

```python
import torch
import torch.nn as nn
import timm
from peft import LoraConfig, get_peft_model
from einops import rearrange
from model.layers.dpt import UpsampleFeats, ProjFeats


class MNv4Encoder(nn.Module):
    SUPPORTED = {
        'mobilenetv4_conv_small',
        'mobilenetv4_conv_medium',
        'mobilenetv4_conv_large',
        'mobilenetv4_hybrid_medium',
        'mobilenetv4_hybrid_large',
    }

    def __init__(self, model_name='mobilenetv4_conv_medium', alpha=None, r=None):
        super().__init__()
        assert model_name in self.SUPPORTED, f"Unknown model: {model_name}"

        backbone = timm.create_model(
            model_name,
            pretrained=True,
            features_only=True,
            out_indices=(1, 2, 3, 4),
        )

        dims = backbone.feature_info.channels()  # auto-detect, do NOT hardcode
        self.output_dim = dims[0]

        if r is not None:
            lora_config = LoraConfig(
                r=r,
                lora_alpha=alpha if alpha is not None else r * 2,
                target_modules=["query", "key", "value", "proj"],  # hybrid variants only
            )
            self.encoder = get_peft_model(backbone, lora_config)
        else:
            self.encoder = backbone

        self.fmap_proj     = ProjFeats(self.output_dim, dims, lvl=-2)
        self.fmap_upsample = UpsampleFeats(self.output_dim, dims)

        self.hidden_proj     = ProjFeats(self.output_dim * 2, [d * 2 for d in dims], lvl=-2)
        self.hidden_upsample = UpsampleFeats(self.output_dim * 2, [d * 2 for d in dims])

    def forward(self, imgs):
        B, N, C, H, W = imgs.shape

        if C == 1:
            imgs = imgs.expand(-1, -1, 3, -1, -1)

        imgs_flat = imgs.reshape(B * N, 3, H, W)
        feats = self.encoder(imgs_flat)

        fmap_feats = self.fmap_proj(feats)
        fmap_feats = self.fmap_upsample(fmap_feats)
        fmaps = rearrange(fmap_feats[0], '(b n) c h w -> b n c h w', n=2)
        fmap1, fmap2 = fmaps[:, 0], fmaps[:, 1]

        hidden_feats = [rearrange(x, '(b n) c h w -> b (n c) h w', n=2) for x in feats]
        hidden_feats = self.hidden_proj(hidden_feats)
        hidden_feats = self.hidden_upsample(hidden_feats)
        hidden = hidden_feats[0]

        return fmap1, fmap2, hidden
```

---

### Step 2: Register the New Feature Encoder

Update `model/encoder/__init__.py`:

```python
from .mnv4 import MNv4Encoder

def fetch_feature_encoder(cfg):
    ...
    elif cfg.TYPE == 'mnv4':
        factor = 8  # WAFT warps at H/2; feature stride before DPT upsample is 8
        encoder = MNv4Encoder(
            model_name=cfg.ARCH,
            alpha=cfg.get('LORA_ALPHA', None),
            r=cfg.get('LORA_RANK', None),
        )
        encoder_dim = encoder.output_dim
    ...
```

---

### Step 3: Implement MobileNetV4 Iterative Decoder (MNv4Iter)

Create `model/iterative/mnv4.py` to replace the ViT-S + DPT recurrent updater.

`MNv4Iter` uses **UIB blocks** (the Universal Inverted Bottleneck building block native to MNv4) paired with a **ConvGRU cell** as the recurrent state update mechanism. Running the full MobileNetV4 backbone inside the iterative loop would be prohibitively expensive over T iterations. High-resolution ResNet blocks are retained from the original WAFT-Stereo design.

```python
import torch
import torch.nn as nn
from model.layers.block import resconv


class UIBBlock(nn.Module):
    def __init__(self, dim, expand_ratio=4):
        super().__init__()
        hidden = int(dim * expand_ratio)
        self.dw      = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.expand  = nn.Conv2d(dim, hidden, 1)
        self.act     = nn.GELU()
        self.project = nn.Conv2d(hidden, dim, 1)
        self.norm    = nn.BatchNorm2d(dim)

    def forward(self, x):
        return x + self.project(self.act(self.expand(self.act(self.dw(self.norm(x))))))


class ConvGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.reset_gate  = nn.Conv2d(input_dim + hidden_dim, hidden_dim, 3, padding=1)
        self.update_gate = nn.Conv2d(input_dim + hidden_dim, hidden_dim, 3, padding=1)
        self.out_gate    = nn.Conv2d(input_dim + hidden_dim, hidden_dim, 3, padding=1)

    def forward(self, x, h):
        combined = torch.cat([x, h], dim=1)
        r = torch.sigmoid(self.reset_gate(combined))
        z = torch.sigmoid(self.update_gate(combined))
        o = torch.tanh(self.out_gate(torch.cat([x, r * h], dim=1)))
        return (1 - z) * h + z * o


class MNv4Iter(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_uib_blocks=4, res_layers=4):
        super().__init__()
        self.input_proj = nn.Conv2d(input_dim * 2, hidden_dim, 1)
        self.uib_blocks = nn.Sequential(*[UIBBlock(hidden_dim) for _ in range(num_uib_blocks)])
        self.gru        = ConvGRUCell(hidden_dim, hidden_dim)
        self.res_convs  = nn.Sequential(*[resconv(hidden_dim, hidden_dim, k=3, s=1) for _ in range(res_layers)])
        self.output_dim = hidden_dim

    def forward(self, fmap_left, warped_fmap_right, hidden):
        x      = torch.cat([fmap_left, warped_fmap_right], dim=1)
        x      = self.input_proj(x)
        x      = self.uib_blocks(x)
        hidden = self.gru(x, hidden)
        hidden = self.res_convs(hidden)
        return hidden
```

---

### Step 4: Register the New Decoder

Update `model/iterative/__init__.py`:

```python
from .mnv4 import MNv4Iter

def fetch_iterative_module(cfg, input_dim):
    ...
    elif cfg.TYPE == 'mnv4':
        iter_decoder = MNv4Iter(
            input_dim=input_dim,
            hidden_dim=cfg.get('HIDDEN_DIM', 128),
            num_uib_blocks=cfg.get('NUM_UIB_BLOCKS', 4),
            res_layers=cfg.get('RES_LAYERS', 4),
        )
    ...
```

---

### Step 5: Update Default Parameters & Config Files

Update `bridgedepth/config/default.py`:

```python
_CN.WAFT.FEATURE_ENCODER.TYPE       = "mnv4"
_CN.WAFT.FEATURE_ENCODER.ARCH       = "mobilenetv4_conv_medium"
_CN.WAFT.FEATURE_ENCODER.LORA_RANK  = 8
_CN.WAFT.FEATURE_ENCODER.LORA_ALPHA = 16

_CN.WAFT.ITERATIVE.TYPE             = "mnv4"
_CN.WAFT.ITERATIVE.HIDDEN_DIM       = 128
_CN.WAFT.ITERATIVE.NUM_UIB_BLOCKS   = 4
_CN.WAFT.ITERATIVE.RES_LAYERS       = 4

_CN.WAFT.DISP_BINS                  = 40
_CN.WAFT.MAX_DISP                   = 800
```

Create `configs/custom/MNv4M-grayscale.yaml`:

```yaml
WAFT:
  FEATURE_ENCODER:
    TYPE: "mnv4"
    ARCH: "mobilenetv4_conv_medium"
    LORA_RANK: 8
    LORA_ALPHA: 16
  ITERATIVE:
    TYPE: "mnv4"
    HIDDEN_DIM: 128
    NUM_UIB_BLOCKS: 4
    RES_LAYERS: 4
  DISP_BINS: 40
  MAX_DISP: 800

TRAIN:
  DATASETS: ("sanpo_synthetic",)
  BATCH_SIZE: 8
  LR: 1.0e-4
  CROP_SIZE: [320, 240]
  STEPS: 50000
```

---

### Step 6: Integrate SANPO Datasets

SANPO stores depth in meters (float), not disparity in pixels. Convert using:

    disparity = (focal_length_px * baseline_m) / depth_m

Update `bridgedepth/dataloader/datasets.py`:

```python
import numpy as np
import os
import glob
import json


class SanpoSynthetic(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/sanpo_synthetic'):
        super().__init__(aug_params, sparse=False, reader=self._read_sanpo_depth)
        assert os.path.exists(root), f"Dataset root not found: {root}"
        for session in sorted(glob.glob(os.path.join(root, 'session_*'))):
            lefts  = sorted(glob.glob(os.path.join(session, 'left',  '*.png')))
            rights = sorted(glob.glob(os.path.join(session, 'right', '*.png')))
            depths = sorted(glob.glob(os.path.join(session, 'depth', '*.npy')))
            calib  = os.path.join(session, 'calib.json')
            assert os.path.exists(calib), f"Missing calib: {calib}"
            for l, r, d in zip(lefts, rights, depths):
                self.image_list.append([l, r])
                self.disparity_list.append([d, calib])

    @staticmethod
    def _read_sanpo_depth(path_pair):
        depth_path, calib_path = path_pair
        depth = np.load(depth_path).astype(np.float32)
        with open(calib_path) as f:
            calib = json.load(f)
        depth = np.clip(depth, 0.1, 100.0)
        return (calib['focal_length_px'] * calib['baseline_m']) / depth


class SanpoReal(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/sanpo_real'):
        super().__init__(aug_params, sparse=False, reader=self._read_sanpo_depth)
        assert os.path.exists(root), f"Dataset root not found: {root}"
        for session in sorted(glob.glob(os.path.join(root, 'session_*'))):
            lefts  = sorted(glob.glob(os.path.join(session, 'left',     '*.png')))
            rights = sorted(glob.glob(os.path.join(session, 'right',    '*.png')))
            depths = sorted(glob.glob(os.path.join(session, 'depth_ml', '*.npy')))
            calib  = os.path.join(session, 'calib.json')
            for l, r, d in zip(lefts, rights, depths):
                self.image_list.append([l, r])
                self.disparity_list.append([d, calib])

    @staticmethod
    def _read_sanpo_depth(path_pair):
        depth_path, calib_path = path_pair
        depth = np.load(depth_path).astype(np.float32)
        with open(calib_path) as f:
            calib = json.load(f)
        valid = depth > 0.1
        disparity = np.zeros_like(depth)
        disparity[valid] = (calib['focal_length_px'] * calib['baseline_m']) / depth[valid]
        return disparity
```

Register in `build_train_loader()`:

```python
elif dataset_name == 'sanpo_synthetic':
    new_dataset = SanpoSynthetic(aug_params)
    logger.info(f"{len(new_dataset)} samples from SanpoSynthetic")
elif dataset_name == 'sanpo_real':
    new_dataset = SanpoReal(aug_params)
    logger.info(f"{len(new_dataset)} samples from SanpoReal")
```

---

### Step 7: Grayscale Input Handling

SANPO captures mono (1-channel) video. MobileNetV4 is pretrained on RGB (3 channels). Handle inside `MNv4Encoder.forward()` -- no DataLoader changes required:

```python
# Already included in MNv4Encoder.forward():
if C == 1:
    imgs = imgs.expand(-1, -1, 3, -1, -1)  # broadcast 1ch -> 3ch
```

Alternatively, handle at load time in the DataLoader:

```python
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
img = np.stack([img] * 3, axis=2)          # [H, W, 3]
```

---

### Step 8: Two-Stage Fine-Tuning Schedule

#### Stage 1 -- SANPO-Synthetic (Dense Geometry Learning)

Objective: Adapt MNv4 encoder and UIB-based iterative updater to egocentric outdoor depth using clean renderer supervision.

```yaml
# configs/custom/stage1-sanpo-synthetic.yaml
TRAIN:
  DATASETS: ("sanpo_synthetic",)
  BATCH_SIZE: 8
  LR: 1.0e-4
  CROP_SIZE: [320, 240]
  STEPS: 50000
  OPTIMIZER: AdamW
  SCHEDULER: OneCycle
  WARMUP_STEPS: 2000
```

Validation: EPE on SANPO-Synthetic validation split should decrease steadily. If EPE > 3px after 10k steps, reduce LR to `5e-5`.

#### Stage 2 -- SANPO-Real (Sim-to-Real Adaptation)

Objective: Adapt to the real egocentric domain -- motion blur, ML depth noise, long-tail objects.

```yaml
# configs/custom/stage2-sanpo-real.yaml
TRAIN:
  DATASETS: ("sanpo_real",)
  BATCH_SIZE: 4
  LR: 5.0e-5
  CROP_SIZE: [320, 240]
  STEPS: 20000
  OPTIMIZER: AdamW
  SCHEDULER: Cosine
  WARMUP_STEPS: 500

CHECKPOINT:
  LOAD: "checkpoints/stage1_sanpo_synthetic_final.pth"
  FREEZE_ENCODER: false
```

#### Pipeline Validation (Overfit One Batch)

Run locally (RTX 3050 6GB) before uploading to Kaggle:

```python
import torch
from torch.cuda.amp import autocast, GradScaler
from waft_model import WAFTStereo

model = WAFTStereo(encoder_type='mnv4', encoder_arch='mobilenetv4_conv_medium').cuda()
model.train()

dummy_left  = torch.randn(1, 1, 240, 320).cuda()
dummy_right = torch.randn(1, 1, 240, 320).cuda()
dummy_disp  = torch.rand(1, 240, 320).cuda() * 100

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scaler    = GradScaler()

for step in range(50):
    optimizer.zero_grad()
    with autocast():
        _, loss = model(dummy_left, dummy_right, dummy_disp)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    if step % 10 == 0:
        print(f"Step {step:>3d} | Loss: {loss.item():.4f}")

# Expected: loss decreases from ~20 to ~0.1 within 50 steps.
# If loss does not decrease: check for shape mismatch in MNv4Encoder.forward().
# If CUDA OOM: reduce input to 128x96 or enable gradient checkpointing.
```

---

## Files Changed Summary

| File | Action | Notes |
|------|--------|-------|
| `model/encoder/mnv4.py` | Create | MNv4Encoder with Siamese design, LoRA, DPT projection |
| `model/encoder/__init__.py` | Modify | Register `mnv4` type; set `factor = 8` |
| `model/iterative/mnv4.py` | Create | MNv4Iter with UIBBlock + ConvGRU (no full backbone inside iter) |
| `model/iterative/__init__.py` | Modify | Register `mnv4` type with `hidden_dim`, `num_uib_blocks` |
| `bridgedepth/config/default.py` | Modify | Add config keys for MNv4 encoder and decoder |
| `bridgedepth/dataloader/datasets.py` | Modify | Add `SanpoSynthetic`, `SanpoReal` with depth-to-disparity conversion |
| `configs/custom/MNv4M-grayscale.yaml` | Create | Main config for SentienVision (320x240, grayscale) |
| `configs/custom/stage1-sanpo-synthetic.yaml` | Create | Stage 1 fine-tuning config |
| `configs/custom/stage2-sanpo-real.yaml` | Create | Stage 2 fine-tuning config |

---

## Known Issues and Open Questions

1. **LoRA on Conv2d**: PEFT does not natively support LoRA on `Conv2d`. For `mobilenetv4_conv_*` variants, either freeze the encoder and train only the DPT head, or switch to `mobilenetv4_hybrid_medium` to enable LoRA on attention projections.

2. **SANPO depth file format**: Verify the actual storage format (`.npy` float32, 16-bit PNG, or `.pfm`) against the official README at `google-research-datasets/sanpo_dataset` before finalizing the reader.

3. **SANPO calibration metadata**: SANPO-Real uses a ZED 2 stereo camera with ~12 cm baseline. Confirm focal length per-session from the released metadata. SANPO-Synthetic may use a fixed calibration across all sessions.

4. **`factor` variable scope**: Verify where `factor` is consumed in `algorithms/` or `model/` before changing the value to confirm `factor = 8` is correct end-to-end.
