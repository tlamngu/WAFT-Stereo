# WAFT-Stereo Data Preparation Guide

This guide describes how to structure, format, and prepare the dataset files for training and evaluating WAFT-Stereo models, with specific focus on the newly integrated **Sanpo-Synthetic** and **Sanpo-Real** datasets.

---

## 📂 1. Directory Structure

All datasets are expected to reside inside a root-level `datasets/` directory in the repository:

```
WAFT-Stereo/
├── datasets/
│   ├── sceneflow/
│   ├── KITTI/
│   ├── middlebury/
│   ├── sanpo_synthetic/  <-- New Synthetic Dataset
│   └── sanpo_real/       <-- New Real Dataset
```

---

## 🏙️ 2. Sanpo-Synthetic & Sanpo-Real Preparation

Because SANPO captures metric depth instead of pixel disparity, the loader converts depth maps dynamically. Ensure your files are structured exactly as shown below:

### Directory Layout
```
datasets/
├── sanpo_synthetic/
│   ├── session_0001/
│   │   ├── left/
│   │   │   ├── 000000.png
│   │   │   └── 000001.png
│   │   ├── right/
│   │   │   ├── 000000.png
│   │   │   └── 000001.png
│   │   ├── depth/
│   │   │   ├── 000000.npy  (float32 depth in meters)
│   │   │   └── 000001.npy
│   │   └── calib.json
│   └── session_0002/
│       └── ...
└── sanpo_real/
    ├── session_1001/
    │   ├── left/
    │   ├── right/
    │   ├── depth_ml/       (float32 ML depth in meters)
    │   └── calib.json
    └── ...
```

### Calibration JSON Format (`calib.json`)
Every session folder must contain a `calib.json` containing the camera parameters:
```json
{
  "focal_length_px": 532.5,
  "baseline_m": 0.12
}
```

### Depth-to-Disparity Conversion
The loader converts the metric depth maps using:
$$\text{disparity} = \frac{\text{focal\_length\_px} \times \text{baseline\_m}}{\text{depth\_m}}$$

* **SanpoSynthetic**: Depth values are clipped to `[0.1m, 100.0m]` to avoid zero-division.
* **SanpoReal**: Depths under `0.1m` are marked as invalid (disparity = 0, excluded from loss).

---

## 🚂 3. Standard Datasets Layout

### KITTI (2012 / 2015)
Expects the official stereo benchmark files:
```
datasets/KITTI/
├── 2012/
│   ├── training/
│   │   ├── colored_0/ (left pngs)
│   │   ├── colored_1/ (right pngs)
│   │   └── disp_noc/ (ground truth)
│   └── testing/
└── 2015/
    ├── training/
    │   ├── image_2/ (left pngs)
    │   ├── image_3/ (right pngs)
    │   └── disp_occ_0/ (ground truth)
    └── testing/
```

### SceneFlow
Expects finalpass/cleanpass files and PFM disparity maps:
```
datasets/sceneflow/
├── FlyingThings3D/
│   ├── frames_finalpass/
│   └── disparity/
├── Monkaa/
└── driving/
```
