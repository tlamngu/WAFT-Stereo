# WAFT-Stereo Architectural Analysis & Model Split

This document details the architectural breakdown of the **WAFT-Stereo** repository, showcasing the split of responsibilities and structure across three main components: **DepthAnythingV2 (Feature Encoder)**, **WAFT-Stereo (Overall Pipeline/Stereo matching logic)**, and **ViT (Iterative refinement modules & transformer backbones)**.

---

## 🏛️ Architectural Overview

At a high level, WAFT-Stereo operates as an iterative stereo-matching pipeline that extracts dense feature maps from left/right image pairs, computes an initial disparity distribution, and refines it iteratively. The framework leverages pre-trained foundation models (ViT/DINOv2/DepthAnythingV2) adapted via parameter-efficient fine-tuning (PEFT/LoRA) to execute high-quality stereo matching.

```mermaid
graph TD
    subgraph Input
        L[Left Image]
        R[Right Image]
    end

    subgraph Feature Encoder [DepthAnythingV2 / ViT]
        direction TB
        DA2[DAv2Encoder or DINOv3Encoder]
        LoRA1[LoRA Fine-tuning]
        DA2 --> LoRA1
    end

    subgraph WAFT-Stereo Pipeline
        direction TB
        Init[Proposal Initializer]
        Warp[Disparity Warping]
        Iter[Iterative Refinement Task Loop]
    end

    subgraph ViT Iterative Decoders
        direction TB
        PropDec[Proposal Decoder: VitIter]
        DeltaDec[Delta Decoder: VitIter]
    end

    L --> Feature Encoder
    R --> Feature Encoder
    Feature Encoder -->|Left/Right Feature Maps| WAFT-Stereo Pipeline
    WAFT-Stereo Pipeline -->|Feature Maps| PropDec
    PropDec -->|Initial Disparity| Warp
    Warp -->|Warped Features| DeltaDec
    DeltaDec -->|Disparity Delta| Iter
    Iter -->|Refined Disparity Output| Final[Final Predicted Disparity]
```

---

## 🔍 Detailed Component Split

### 1. DepthAnythingV2 (`DepthAnythingV2`)
* **Role**: Primary robust feature extractor backend.
* **Implementation Location**:
  * [model/encoder/dav2.py](file:///C:/Users/tlamn/Documents/AI/WAFT-Stereo/model/encoder/dav2.py) (Wrapper module `DAv2Encoder`)
  * [thirdparty/DepthAnythingV2/depth_anything_v2/dpt.py](file:///C:/Users/tlamn/Documents/AI/WAFT-Stereo/thirdparty/DepthAnythingV2/depth_anything_v2/dpt.py) (Pretrained model definition)
* **Key Mechanics**:
  * Loads pretrained DPT checkpoints from `depth-anything-ckpts/depth_anything_v2_{model_name}.pth` supporting `vits`, `vitb`, and `vitl` backbones.
  * Extrapolates intermediate layer activations using `get_intermediate_layers()` at layer indices spaced evenly throughout the backbone depth.
  * Injects parameter-efficient fine-tuning (PEFT) using **LoRA** (Low-Rank Adaptation) targets on attention modules (`["qkv", "proj"]`) to keep the primary backbone frozen while adapting feature representation specifically for stereo matching:
    ```python
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["qkv", "proj"]
    )
    self.encoder = get_peft_model(depth_anything.pretrained, lora_config)
    ```
  * Performs projection and bilinear upsampling of intermediate features to prepare them for the disparity heads.

### 2. Waft-Stereo (`Waft-stereo`)
* **Role**: Pipeline manager, matcher, cost-aggregator, warping, and coordinator.
* **Implementation Location**:
  * [algorithms/waft.py](file:///C:/Users/tlamn/Documents/AI/WAFT-Stereo/algorithms/waft.py) (Main `WAFT` PyTorch module)
* **Key Mechanics**:
  * Normalizes the stereo inputs and pads them to match the factor of the feature encoder.
  * Extracts representations from left and right images simultaneously via the configured feature encoder:
    ```python
    fmap1, fmap2, net = self.encoder(torch.stack([image1, image2], dim=1))
    ```
  * Utilizes a **Proposal Module** (`prop_decoder` using `VitIter`) to predict initial probability distributions over a set of disparity bins (`n_bins`). Computes expected initial disparity using soft-argmax:
    ```python
    prob_bins = F.softmax(prob_bins, dim=1)
    disp = torch.sum(prob_bins * idx_bins_2x, dim=1, keepdim=True)
    ```
  * Loops through refinement iterations (`iters`). In each iteration:
    1. Warps the right image feature map to match the current left image disparity estimation (`disp_warp`).
    2. Feeds a concatenation of left feature, warped right feature, context features, and current disparity into the **Delta Module** (`delta_decoder` using `VitIter`).
    3. Outputs a delta updates (`delta_disp`) and bilinear convex upsampling weights to upscale predictions back to original resolution.

### 3. ViT (`ViT`)
* **Role**: Refinement decoding engine and alternative feature backbones.
* **Implementation Location**:
  * [model/iterative/vit.py](file:///C:/Users/tlamn/Documents/AI/WAFT-Stereo/model/iterative/vit.py) (`VitIter` PyTorch module)
  * [model/encoder/dinov3.py](file:///C:/Users/tlamn/Documents/AI/WAFT-Stereo/model/encoder/dinov3.py) (DINOv3-based feature encoder)
* **Key Mechanics**:
  * **Iterative Decoder (`VitIter`)**: Instead of relying on traditional GRUs (like RAFT) for iterative refinement, WAFT-Stereo leverages a lightweight Vision Transformer (`VitIter`) from `timm` (supporting configurations `vit_large`, `vit_base`, `vit_small`, `vit_tiny`).
    * Input features are downsampled via convolutional residual blocks (`patch_embed`).
    * Converted to tokens via sequence rearrangement (`einops.rearrange`) and processed by transformer blocks:
      ```python
      for i in range(len(self.blks)):
          vit_x = self.blks[i](vit_x)
          if i in self.idx:
              vit_feats.append(rearrange(vit_x, 'b (h w) c -> b c h w', h=h, w=w))
      ```
    * The transformer blocks are fine-tuned using LoRA targeting standard projection/query-key-value modules (`"qkv"`, `"proj"`).
    * Upsampled and combined with residual skip connections.
  * **DINOv3 Feature Encoder**: Configurable option (`dinov3` in configs) that utilizes pre-trained DINOv3 backbones from `timm` (`vit_7b_patch16_dinov3`, `vit_huge_plus_patch16_dinov3`, etc.) with LoRA adapters as the primary feature extraction tool instead of DepthAnythingV2.

---

## 🛠️ Config Example: Bringing it Together

In typical configs such as [configs/SynLarge/DAv2L-5.yaml](file:///C:/Users/tlamn/Documents/AI/WAFT-Stereo/configs/SynLarge/DAv2L-5.yaml), the components are combined:

* **Feature Encoder**: Type `dav2` (DepthAnythingV2) using `vitl` (ViT Large) architecture, adapted with LoRA parameters `r=8` and `alpha=16`.
* **Iterative Modules**: Both proposal and delta modules are configured as `vit` (ViT decoders) using the `vits` (ViT Small) architecture with a `patch_size` of `8`.
* **Refinement Loop**: Iterates 4 times (`TASK: ['delta', 'delta', 'delta', 'delta']`).
