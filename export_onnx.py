import os
import torch

# Explicitly disable xformers during ONNX export to avoid CPU trace errors
try:
    import thirdparty.DepthAnythingV2.depth_anything_v2.dinov2_layers.attention as attention_module
    import thirdparty.DepthAnythingV2.depth_anything_v2.dinov2_layers.block as block_module
    import thirdparty.DepthAnythingV2.depth_anything_v2.dinov2_layers.swiglu_ffn as swiglu_module
    attention_module.XFORMERS_AVAILABLE = False
    block_module.XFORMERS_AVAILABLE = False
    swiglu_module.XFORMERS_AVAILABLE = False
    print("Disabled xformers for ONNX export compatibility.")
except ImportError:
    pass

import torch.nn as nn
import torch.nn.functional as F
from bridgedepth.config import get_cfg
from algorithms.waft import WAFT
from peft import PeftModel


def safe_normalize_coords(grid):
    h, w = grid.size()[2:]
    grid_x = 2.0 * (grid[:, 0:1] / (w - 1.0)) - 1.0
    grid_y = 2.0 * (grid[:, 1:2] / (h - 1.0)) - 1.0
    grid_norm = torch.cat([grid_x, grid_y], dim=1)
    return grid_norm.permute(0, 2, 3, 1)

def safe_meshgrid(img):
    b, _, h, w = img.size()
    # Explicitly use float type for grid coordinates
    x_range = torch.arange(0, w, dtype=img.dtype, device=img.device).view(1, 1, w).expand(1, h, w)
    y_range = torch.arange(0, h, dtype=img.dtype, device=img.device).view(1, h, 1).expand(1, h, w)
    grid = torch.cat((x_range, y_range), dim=0)
    return grid.unsqueeze(0).expand(b, 2, h, w)

def safe_disp_warp(feature, disp):
    grid = safe_meshgrid(feature)
    offset = torch.cat((-disp, torch.zeros_like(disp)), dim=1)
    sample_grid = grid + offset
    sample_grid = safe_normalize_coords(sample_grid)
    return F.grid_sample(feature, sample_grid, mode='bilinear', padding_mode='zeros')

class WAFTOnnxWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.max_disp = model.max_disp
        self.n_bins = model.n_bins
        self.iters = model.iters

    def forward(self, img1, img2):
        # inputs are pre-normalized and padded (shape: B, 3, H, W)
        fmap1, fmap2, net = self.model.encoder(torch.stack([img1, img2], dim=1))
        n, _, h, w = fmap1.shape

        idx_bins_2x = torch.linspace(0, self.max_disp/2, self.n_bins, device=fmap1.device, dtype=fmap1.dtype).view(1, self.n_bins, 1, 1)
        idx_bins_1x = torch.linspace(0, self.max_disp/1, self.n_bins, device=fmap1.device, dtype=fmap1.dtype).view(1, self.n_bins, 1, 1)

        prop_hidden = self.model.prop_proj(torch.cat([fmap1, fmap2], dim=1))
        prop_hidden = self.model.prop_decoder(prop_hidden)
        prob_mask = .25 * self.model.prop_mask_head(prop_hidden)
        prob_bins = self.model.prop_bins_head(prop_hidden)
        prob_up = self.model.convex_upsample(prob_bins, prob_mask)
        
        prob_bins = F.softmax(prob_bins, dim=1)
        disp = torch.sum(prob_bins * idx_bins_2x, dim=1, keepdim=True)

        delta_disp_preds = []
        for itr in range(self.iters):
            disp = disp.detach()
            # Use safe version of disp_warp
            warped_fmap2 = safe_disp_warp(fmap2, disp)
            net = self.model.delta_proj(torch.cat([fmap1, warped_fmap2, net, disp], dim=1))
            net = self.model.delta_decoder(net)
            delta_disp = self.model.delta_disp_head(net)
            mask = .25 * self.model.delta_mask_head(net)
            disp = disp + delta_disp
            disp_up = self.model.convex_upsample(disp * 2, mask)
            delta_disp_preds.append(disp_up)

        if self.iters > 0:
            disp_final = delta_disp_preds[-1].squeeze(1)
        else:
            disp_final = torch.sum(F.softmax(prob_up, dim=1) * idx_bins_1x, dim=1)
            
        return disp_final

def main():
    config_file = "configs/SynLarge/DAv2S-4.yaml"
    ckpt_file = "ckpts/DAv2S-4.pth"
    onnx_file = "ckpts/DAv2S-4.onnx"

    print("Loading config...")
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.freeze()

    print("Initializing model...")
    model = WAFT(cfg)
    
    print(f"Loading checkpoint from {ckpt_file}...")
    checkpoint = torch.load(ckpt_file, map_location='cpu', weights_only=False)
    weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(weights, strict=False)

    print("Merging and unloading LoRA weights...")
    for name, module in model.named_modules():
        if isinstance(module, PeftModel):
            print(f"Merging LoRA weights for: {name}")
            module.merge_and_unload()

    model.eval()
    onnx_wrapper = WAFTOnnxWrapper(model)
    onnx_wrapper.eval()

    # Create dummy inputs
    dummy_img1 = torch.randn(1, 3, 480, 640)
    dummy_img2 = torch.randn(1, 3, 480, 640)

    print("Exporting model to ONNX...")
    # opset 16 or 17 is recommended for grid_sample support
    torch.onnx.export(
        onnx_wrapper,
        (dummy_img1, dummy_img2),
        onnx_file,
        input_names=['img1', 'img2'],
        output_names=['disp_pred'],
        dynamic_axes={
            'img1': {0: 'batch_size', 2: 'height', 3: 'width'},
            'img2': {0: 'batch_size', 2: 'height', 3: 'width'},
            'disp_pred': {0: 'batch_size', 1: 'height', 2: 'width'}
        },
        opset_version=16,
        verbose=False
    )
    print(f"Successfully exported ONNX model to {onnx_file}!")

if __name__ == '__main__':
    main()
