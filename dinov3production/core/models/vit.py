from functools import partial
from torch import nn
from ..vision_transformer import DinoV3VisionTransformer
from ..registry import register_model

def _create_dinov3_model(patch_size, embed_dim, depth, num_heads, **kwargs):
    model = DinoV3VisionTransformer(
        patch_size=patch_size, 
        embed_dim=embed_dim, 
        depth=depth, 
        num_heads=num_heads, 
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        num_register_tokens=4,
        use_swiglu=True,
        use_rope=True,
        **kwargs
    )
    return model

# --- Patch Size 14 ---

@register_model('dinov3_vits14')
def dinov3_vits14(**kwargs):
    return _create_dinov3_model(14, 384, 12, 6, **kwargs)

@register_model('dinov3_vitb14')
def dinov3_vitb14(**kwargs):
    return _create_dinov3_model(14, 768, 12, 12, **kwargs)

@register_model('dinov3_vitl14')
def dinov3_vitl14(**kwargs):
    return _create_dinov3_model(14, 1024, 24, 16, **kwargs)

@register_model('dinov3_vitg14')
def dinov3_vitg14(**kwargs):
    return _create_dinov3_model(14, 1536, 40, 24, **kwargs)

@register_model('dinov3_vit7b14')
def dinov3_vit7b14(**kwargs):
    """ The 7B Parameter Giant """
    # Heads=32, Dim=4096 -> Head Dim = 128
    return _create_dinov3_model(14, 4096, 40, 32, **kwargs)


# --- Patch Size 16 ---

@register_model('dinov3_vits16')
def dinov3_vits16(**kwargs):
    return _create_dinov3_model(16, 384, 12, 6, **kwargs)

@register_model('dinov3_vitb16')
def dinov3_vitb16(**kwargs):
    return _create_dinov3_model(16, 768, 12, 12, **kwargs)

@register_model('dinov3_vitl16')
def dinov3_vitl16(**kwargs):
    return _create_dinov3_model(16, 1024, 24, 16, **kwargs)

@register_model('dinov3_vit7b16')
def dinov3_vit7b16(**kwargs):
    return _create_dinov3_model(16, 4096, 40, 32, **kwargs)

# --- Patch Size 8 ---

@register_model('dinov3_vits8')
def dinov3_vits8(**kwargs):
    return _create_dinov3_model(8, 384, 12, 6, **kwargs)

@register_model('dinov3_vitb8')
def dinov3_vitb8(**kwargs):
    return _create_dinov3_model(8, 768, 12, 12, **kwargs)

@register_model('dinov3_vitl8')
def dinov3_vitl8(**kwargs):
    return _create_dinov3_model(8, 1024, 24, 16, **kwargs)
