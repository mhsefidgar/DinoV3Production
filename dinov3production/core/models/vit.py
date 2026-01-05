from functools import partial
from torch import nn
from ..vision_transformer import DinoV3VisionTransformer
from ..registry import register_model

@register_model('dinov3_vits14')
def dinov3_vits14(**kwargs):
    model = DinoV3VisionTransformer(
        patch_size=14, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

@register_model('dinov3_vitb14')
def dinov3_vitb14(**kwargs):
    model = DinoV3VisionTransformer(
        patch_size=14, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

@register_model('dinov3_vitl14')
def dinov3_vitl14(**kwargs):
    model = DinoV3VisionTransformer(
        patch_size=14, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

@register_model('dinov3_vitg14')
def dinov3_vitg14(**kwargs):
    model = DinoV3VisionTransformer(
        patch_size=14, embed_dim=1536, depth=40, num_heads=24, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# --- Patch Size 16 ---

@register_model('dinov3_vits16')
def dinov3_vits16(**kwargs):
    model = DinoV3VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

@register_model('dinov3_vitb16')
def dinov3_vitb16(**kwargs):
    model = DinoV3VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

@register_model('dinov3_vitl16')
def dinov3_vitl16(**kwargs):
    model = DinoV3VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# --- Patch Size 8 ---

@register_model('dinov3_vits8')
def dinov3_vits8(**kwargs):
    model = DinoV3VisionTransformer(
        patch_size=8, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

@register_model('dinov3_vitb8')
def dinov3_vitb8(**kwargs):
    model = DinoV3VisionTransformer(
        patch_size=8, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

@register_model('dinov3_vitl8')
def dinov3_vitl8(**kwargs):
    model = DinoV3VisionTransformer(
        patch_size=8, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
