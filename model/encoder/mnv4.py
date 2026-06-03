import torch
import torch.nn as nn
import timm
from peft import LoraConfig, get_peft_model
from einops import rearrange
from model.layers.dpt import UpsampleFeats


class MNv4ProjFeats(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for in_ch, out_ch in zip(in_channels, out_channels)
        ])
        
    def forward(self, out_features):
        out = []
        for i, x in enumerate(out_features):
            x = self.projects[i](x)
            # Upsample by a factor of 2 to align hierarchical strides with DPT head requirements
            x = nn.functional.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=True)
            out.append(x)
        return out


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
        self.out_c = [(int)(self.output_dim * (2 ** i)) for i in range(4)]
        self.output_dim = self.out_c[0]

        # Check if the backbone is a hybrid model (only hybrid models have attention layers suitable for LoRA)
        is_hybrid = "hybrid" in model_name

        if r is not None and is_hybrid:
            lora_config = LoraConfig(
                r=r,
                lora_alpha=alpha if alpha is not None else r * 2,
                target_modules=["query", "key", "value", "proj"],  # hybrid variants only
            )
            self.encoder = get_peft_model(backbone, lora_config)
        else:
            if r is not None and not is_hybrid:
                print(f"WARNING: LoRA is not applicable to pure convolutional variant '{model_name}'. Using backbone with fully trainable pretrained parameters.")
            self.encoder = backbone

        self.fmap_proj     = MNv4ProjFeats(dims, self.out_c)
        self.fmap_upsample = UpsampleFeats(self.output_dim, self.out_c)

        self.hidden_proj     = MNv4ProjFeats([d * 2 for d in dims], self.out_c)
        self.hidden_upsample = UpsampleFeats(self.output_dim, self.out_c)

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
