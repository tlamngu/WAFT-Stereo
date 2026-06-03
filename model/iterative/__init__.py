from .vit import VitIter
from .mnv4 import MNv4Iter

def fetch_iterative_module(cfg, input_dim=3):
    if cfg.TYPE == 'vit':
        decoder_lora_rank = cfg.LORA_RANK
        decoder_lora_alpha = cfg.LORA_ALPHA
        iter_decoder = VitIter(
            cfg.ARCH, 
            input_dim,
            patch_size=cfg.PATCH_SIZE,
            alpha=decoder_lora_alpha,
            r=decoder_lora_rank,
        )
    elif cfg.TYPE == 'mnv4':
        iter_decoder = MNv4Iter(
            input_dim=input_dim,
            hidden_dim=cfg.HIDDEN_DIM if 'HIDDEN_DIM' in cfg else 128,
            num_uib_blocks=cfg.NUM_UIB_BLOCKS if 'NUM_UIB_BLOCKS' in cfg else 4,
            res_layers=cfg.RES_LAYERS if 'RES_LAYERS' in cfg else 4,
        )
    else:
        raise ValueError(f"Unknown iterative module: {cfg.TYPE}")
    
    return iter_decoder