# WAFT-Stereo Dataflow Analysis

This document details the step-by-step dataflow of the **WAFT-Stereo** model, showcasing how inputs flow through preprocessing, feature encoding, proposal initialization, warping, iterative refinement, and final upsampling/unpadding operations.

---

## 📊 Overview Dataflow Diagram

The diagram below maps out the main stages of tensor shapes and functions as they transition from raw stereo input images to the final sub-pixel disparity maps.

```mermaid
graph TD
    %% Inputs
    subgraph Inputs
        L["Left Image (img1)<br/>Shape: (B, 3, H, W)"]
        R["Right Image (img2)<br/>Shape: (B, 3, H, W)"]
    end

    %% Preprocessing
    subgraph Preprocessing [1. Normalize & Pad]
        NormL["Normalize (ImageNet stats)"]
        NormR["Normalize (ImageNet stats)"]
        PadL["Padder.pad()<br/>Shape: (B, 3, Hp, Wp)"]
        PadR["Padder.pad()<br/>Shape: (B, 3, Hp, Wp)"]
        
        L --> NormL --> PadL
        R --> NormR --> PadR
    end

    %% Feature Extraction
    subgraph FeatureExtraction [2. Feature Encoder]
        Stack["Stack Images<br/>Shape: (B, 2, 3, Hp, Wp)"]
        Encoder["DAv2Encoder or DINOv3Encoder<br/>(w/ LoRA fine-tuning)"]
        
        F1["fmap1 (Left features)<br/>Shape: (B, C, Hc, Wc)<br/>Hc = Hp/16, Wc = Wp/16"]
        F2["fmap2 (Right features)<br/>Shape: (B, C, Hc, Wc)"]
        Net["net (Combined context)<br/>Shape: (B, 2C, Hc, Wc)"]
        
        PadL --> Stack
        PadR --> Stack
        Stack --> Encoder
        Encoder --> F1
        Encoder --> F2
        Encoder --> Net
    end

    %% Proposal Initialization
    subgraph ProposalInit [3. Disparity Proposal Initialization]
        ConcatF["Concat [fmap1, fmap2]<br/>Shape: (B, 2C, Hc, Wc)"]
        PropProj["prop_proj (MLP)<br/>Shape: (B, C, Hc, Wc)"]
        PropDec["prop_decoder (VitIter)<br/>Shape: (B, C, Hc, Wc)"]
        
        ProbBins["prop_bins_head (MLP)<br/>Shape: (B, n_bins, Hc, Wc)"]
        ProbMask["prop_mask_head (MLP)<br/>Shape: (B, 4*9, Hc, Wc)"]
        
        UpsampleProb["convex_upsample()<br/>Shape: (B, n_bins, 2Hc, 2Wc)"]
        SoftmaxProb["Softmax over bins"]
        SoftArgmax["Weighted Sum with idx_bins_2x"]
        
        DispInit["disp (Initial Disparity)<br/>Shape: (B, 1, Hc, Wc)"]
        InitPred["output['init']<br/>Shape: (B, n_bins, 2Hc, 2Wc)"]

        ConcatF --> PropProj --> PropDec
        PropDec --> ProbBins
        PropDec --> ProbMask
        ProbBins & ProbMask --> UpsampleProb
        UpsampleProb --> InitPred
        
        ProbBins --> SoftmaxProb
        SoftmaxProb --> SoftArgmax
        SoftArgmax --> DispInit
    end

    %% Iterative Refinement
    subgraph RefinementLoop [4. Iterative Refinement Loop]
        DispDet["disp.detach()"]
        Warp["disp_warp(fmap2, disp)<br/>(F.grid_sample)<br/>Shape: (B, C, Hc, Wc)"]
        ConcatRef["Concat [fmap1, warped_fmap2, net, disp]<br/>Shape: (B, 3C + 2C + 1, Hc, Wc)"]
        DeltaProj["delta_proj (MLP)<br/>Shape: (B, C, Hc, Wc)"]
        DeltaDec["delta_decoder (VitIter)<br/>Shape: (B, C, Hc, Wc)"]
        
        DeltaDisp["delta_disp_head (MLP)<br/>Shape: (B, 1, Hc, Wc)"]
        DeltaMask["delta_mask_head (MLP)<br/>Shape: (B, 4*9, Hc, Wc)"]
        DeltaInfo["delta_dist_head (MLP)<br/>Shape: (B, 4, Hc, Wc)"]
        
        DispUpdate["disp = disp + delta_disp"]
        
        UpsampleDisp["convex_upsample(disp * 2, mask)<br/>Shape: (B, 1, 2Hc, 2Wc)"]
        UpsampleInfo["convex_upsample(info, mask)<br/>Shape: (B, 4, 2Hc, 2Wc)"]

        F1 & F2 & Net & DispInit --> DispDet
        DispDet --> Warp
        F1 & Warp & Net & DispDet --> ConcatRef
        ConcatRef --> DeltaProj --> DeltaDec
        DeltaDec --> DeltaDisp
        DeltaDec --> DeltaMask
        DeltaDec --> DeltaInfo
        
        DispDet & DeltaDisp --> DispUpdate
        DispUpdate & DeltaMask --> UpsampleDisp
        DeltaInfo & DeltaMask --> UpsampleInfo
    end

    %% Postprocessing
    subgraph Postprocessing [5. Postprocessing & Unpad]
        UnpadInit["Padder.unpad(output['init'])<br/>Shape: (B, n_bins, 2Hc, 2Wc)"]
        UnpadDisp["Padder.unpad(disp_up)<br/>Shape: (B, 1, H, W)"]
        UnpadInfo["Padder.unpad(info_up)<br/>Shape: (B, 4, H, W)"]

        InitPred --> UnpadInit
        UpsampleDisp --> UnpadDisp
        UpsampleInfo --> UnpadInfo
    end

    %% Outputs
    subgraph Outputs
        OutInit["output['init']"]
        OutDelta["output['delta_disp_preds']"]
        OutInfo["output['delta_info_preds']"]
        OutFinal["output['disp_pred']"]
    end
    
    UnpadInit --> OutInit
    UnpadDisp --> OutDelta
    UnpadInfo --> OutInfo
    UnpadDisp --> OutFinal
```

---

## 🧬 Key Data Operations Explained

### 1. Coordinate Grids and Warping (`disp_warp`)
Located in [model/utils.py](file:///C:/Users/tlamn/Documents/AI/WAFT-Stereo/model/utils.py#L62-L69), the warping shifts pixels in the right feature map `fmap2` using the estimated disparity `disp`.
* **Grid Creation**: A coordinate grid `[x, y]` is generated using `meshgrid()` in the coordinate scale of the feature maps.
* **Disparity Offset**: The estimated disparity map `disp` is concatenated with zeros (representing no vertical translation) to construct the flow vector: `[-disp, 0]`.
* **Sample Grid**: The flow vectors are added to the coordinate grid, normalized to the range `[-1, 1]` via `normalize_coords()`, and sampled via PyTorch's `F.grid_sample()`.

### 2. Convex Upsampling (`convex_upsample`)
Because feature maps and iterative updates occur at a 1/8 resolution scale (relative to the padded image resolution, since features are downsampled by a factor of 16 and upsampled to a 1/8 scale via `UpsampleFeats`), a convex upsampling kernel is used to lift estimated outputs back to a 1/4 resolution scale before unpadding:
* **Mask Generation**: The mask heads output a tensor of size `(B, 4 * 9, H, W)`. The `4` corresponds to a $2 \times 2$ grid of upsampled sub-pixels, and `9` corresponds to a $3 \times 3$ neighborhood of source low-resolution pixels.
* **Weighted Neighborhood Combination**:
  1. The mask is normalized with a Softmax function over the 9 neighbors.
  2. The target feature maps are unfolded using a $3 \times 3$ patch size.
  3. The target sub-pixel value is computed as a weighted sum of the source $3 \times 3$ neighborhood.
  4. Rearranged to output shape `(B, C, 2H, 2W)`.

### 3. Iterative Feedback Loops
During refinement loops, gradients do not flow backwards through disparity updates into previous loops. This is enforced via `disp.detach()`. The network updates the residual (`delta_disp`) in a self-correcting manner based on the misalignment remaining between `fmap1` and the `warped_fmap2`.
