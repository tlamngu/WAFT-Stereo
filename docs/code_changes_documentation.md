# MobileNetV4 Migration - Code Changes Documentation

This document provides a comprehensive log of every new code block, class, and helper function implemented during the migration of the WAFT-Stereo framework to MobileNetV4.

---

## 💾 1. Feature Encoder: `MNv4Encoder` & `MNv4ProjFeats`
* **File Location**: [model/encoder/mnv4.py](file:///C:/Users/tlamn/Documents/AI/WAFT-Stereo/model/encoder/mnv4.py)

### `MNv4ProjFeats` (Projection Layer)
* **Functionality**: A module list of 1x1 convolutions mapping the different channel counts at each of MobileNetV4's hierarchical stages to standard DPT features.
* **Why it was created**: The native `ProjFeats` in `model/layers/dpt.py` expects a single scalar representing the input channels of all feature maps (since ViT layer tokens always have uniform channel dimensions). MobileNetV4 is a hierarchical CNN and has varying channels per stage (e.g. `[48, 80, 160, 960]`).
* **Stride Scaling (Upsampling by 2)**: Stride projection uses `nn.functional.interpolate(..., scale_factor=2)` to map the hierarchical strides of MobileNetV4 (`/4, /8, /16, /32`) to the target strides (`/2, /4, /8, /16`) required by the downstream warping modules.

### `MNv4Encoder` (Siamese Wrapper)
* **Functionality**: Instantiates the selected MobileNetV4 backbone from `timm` (e.g. `mobilenetv4_conv_medium`), runs input images through it, and manages Siamese forward calls.
* **Grayscale Input Support**: Captures grayscale (1-channel) input tensors and broadcasts/expands them to 3 channels using `expand(-1, -1, 3, -1, -1)` to maintain compatibility with pretrained ImageNet weights.
* **LoRA PEFT support**: Dynamically injects LoRA adapters into the backbone if `LORA_RANK` is configured.

---

## 🔄 2. Recurrent Decoder: `MNv4Iter` & Components
* **File Location**: [model/iterative/mnv4.py](file:///C:/Users/tlamn/Documents/AI/WAFT-Stereo/model/iterative/mnv4.py)

### `UIBBlock` (Universal Inverted Bottleneck Block)
* **Functionality**: A core convolutional block native to MobileNetV4, consisting of a depthwise convolution, expansion, activation, projection, and batch normalization with skip connection.
* **Why it was created**: Replaces the expensive Vision Transformer self-attention blocks in the update loop with a fast, edge-optimized feed-forward block.

### `ConvGRUCell` (Convolutional GRU Cell)
* **Functionality**: Performs gated recurrent updates over spatial grid maps using standard resets and updates.
* **Why it was created**: Manages the hidden/context state updates over recurrent iterations.

### `MNv4Iter`
* **Functionality**: Decodes iterative visual/disparity updates using a single-input signature compatible with `algorithms/waft.py` without modifying the core matching logic. It maps the input channels to 128, runs UIB Blocks and ConvGRUCell updates, and projects channels back.

---

## ⚙️ 3. Registry & Factory Registrations

### `model/encoder/__init__.py`
* **Changes**: Registered the `mnv4` type inside `fetch_feature_encoder()` and set its padding `factor = 16`.

### `model/iterative/__init__.py`
* **Changes**: Registered the `mnv4` type inside `fetch_iterative_module()` to dynamically extract parameters like `HIDDEN_DIM`, `NUM_UIB_BLOCKS`, and `RES_LAYERS`.

### `bridgedepth/config/default.py`
* **Changes**: Added default configuration keys under YACS configuration node for `mnv4` feature encoder and decoders.

---

## 🗄️ 4. Sanpo Dataset Loaders
* **File Location**: [bridgedepth/dataloader/datasets.py](file:///C:/Users/tlamn/Documents/AI/WAFT-Stereo/bridgedepth/dataloader/datasets.py)

### `SanpoSynthetic` & `SanpoReal`
* **Functionality**: Parses and loads dataset sessions (left/right/depth pairs) for simulated and real egocentric stereo datasets.
* **Reader Conversion (`_read_sanpo_depth`)**: Maps metric depth files (`.npy` containing distance in meters) to pixel-level disparities using focal length and baseline configuration:
  $$\text{disparity} = \frac{\text{focal\_length\_px} \times \text{baseline\_m}}{\text{depth\_m}}$$

---

## ☁️ 5. Cloud Checkpoint Uploads
* **File Location**: [main.py](file:///C:/Users/tlamn/Documents/AI/WAFT-Stereo/main.py)

### `upload_checkpoint_to_r2`
* **Functionality**: Checks for configuration credentials stored in the local `.env` file, initializes a `boto3` client pointing to a Cloudflare R2 bucket endpoint, and uploads step/latest checkpoints asynchronously on the main training processes.

---

## 💻 6. Device-Agnostic Execution & CPU Dry-Run Support
* **File Locations**: [main.py](file:///C:/Users/tlamn/Documents/AI/WAFT-Stereo/main.py), [bridgedepth/utils/misc.py](file:///C:/Users/tlamn/Documents/AI/WAFT-Stereo/bridgedepth/utils/misc.py), [bridgedepth/utils/eval_disp.py](file:///C:/Users/tlamn/Documents/AI/WAFT-Stereo/bridgedepth/utils/eval_disp.py), [model/encoder/mnv4.py](file:///C:/Users/tlamn/Documents/AI/WAFT-Stereo/model/encoder/mnv4.py)

### Dynamic CUDA Checks
* **Functionality**: Replaced all hardcoded `.cuda()` and `device="cuda"` targets with dynamic device bindings (`device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`).
* **Profiler, Autocast & BatchNorm**: Conditioned `SyncBatchNorm` conversion to run only when CUDA is active. Restructured profiling activities and disabled mixed-precision `autocast` during CPU dry-runs to prevent runtime crashes.
* **Process Metric Synchronization**: Restructured distributed process reductions in `misc.py` and evaluation data movements in `eval_disp.py` to dynamically target the active device (CPU or GPU), resolving backend-mismatch exceptions.

### Dynamic LoRA & Parameter Freezing
* **Functionality**: Checked if the selected backbone model is a hybrid variant (`"hybrid" in model_name`). If a pure convolutional variant is selected (e.g. `mobilenetv4_conv_medium`), the training code logs a warning, skips PEFT/LoRA wrapping, and freezes the backbone parameters by setting `requires_grad = False` (as specified in the migration plan).
