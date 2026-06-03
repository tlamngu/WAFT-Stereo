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

The official SANPO dataset is hosted on Google Cloud Storage. Since the full dataset is roughly 6 TB, it is recommended to selectively download specific sessions.

### 📥 How to Download the Dataset
You need to install the Google Cloud CLI (`gcloud`) on your machine. Follow the [Official gcloud CLI Installation Guide](https://cloud.google.com/sdk/docs/install).

Once installed, use the `gcloud storage cp` command to download specific session directories:

1. **Download a Synthetic Session**:
   ```bash
   # Download a synthetic session folder (containing left, right, depth (.npz), and calib.json)
   gcloud storage cp -r "gs://gresearch/sanpo_dataset/v0/synthetic/session_0001" datasets/sanpo_synthetic/
   ```

2. **Download a Real Session**:
   ```bash
   # Download a real session folder (containing left, right, depth_ml (.npz), and calib.json)
   gcloud storage cp -r "gs://gresearch/sanpo_dataset/v0/real/session_1001" datasets/sanpo_real/
   ```

3. **Check/List Official Train/Test Splits**:
   ```bash
   gcloud storage ls gs://gresearch/sanpo_dataset/v0/synthetic/splits/
   gcloud storage ls gs://gresearch/sanpo_dataset/v0/real/splits/
   ```

Because SANPO captures metric depth instead of pixel disparity, the loader converts depth maps dynamically. Our dataloader dynamically supports reading both standard `.npy` and compressed `.npz` NumPy depth formats. Ensure your downloaded files are structured exactly as shown below:

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
The SceneFlow dataset consists of three subsets: `FlyingThings3D`, `Monkaa`, and `driving`. The default `sceneflow` dataloader dynamically aggregates all three subsets for training. Ensure your folders are structured as follows:

```
datasets/sceneflow/
├── FlyingThings3D/
│   ├── frames_finalpass/
│   │   └── TRAIN/ (containing left/right image folders)
│   └── disparity/
│       └── TRAIN/ (containing left/right disparity PFM folders)
├── Monkaa/
│   ├── frames_finalpass/ (containing left/right image folders)
│   └── disparity/ (containing left/right disparity PFM folders)
└── driving/
    ├── frames_finalpass/
    │   └── [focal_length_param]/[scene_direction]/[scene_speed]/
    │       ├── left/  (e.g., 15mm_focallength/scene_backwards/fast/left/*.png)
    │       └── right/ (e.g., 15mm_focallength/scene_backwards/fast/right/*.png)
    └── disparity/
        └── [focal_length_param]/[scene_direction]/[scene_speed]/
            └── left/  (e.g., 15mm_focallength/scene_backwards/fast/left/*.pfm)
```
