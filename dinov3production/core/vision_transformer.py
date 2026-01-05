import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class RotaryEmbedding(nn.Module):
    """
    2D Axial Rotary Positional Embedding.
    """
    def __init__(self, dim, max_resolution=224, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        
    def forward(self, x, H, W):
        # x: [B, N, heads, head_dim] (or similar, depending on where applied)
        device = x.device
        
        # 1D freqs
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim // 4, 2, device=device).float() / (self.dim // 4)))
        
        t_h = torch.arange(H, device=device, dtype=inv_freq.dtype)
        t_w = torch.arange(W, device=device, dtype=inv_freq.dtype)
        
        freqs_h = torch.outer(t_h, inv_freq) # [H, D/4]
        freqs_w = torch.outer(t_w, inv_freq) # [W, D/4]
        
        # Construct grid
        freqs_h = freqs_h.unsqueeze(1).repeat(1, W, 1) # [H, W, D/4]
        freqs_w = freqs_w.unsqueeze(0).repeat(H, 1, 1) # [H, W, D/4]
        
        freqs = torch.cat([freqs_h, freqs_w], dim=-1).flatten(0, 1) # [H*W, D/2]
        
        # Complex rope (sin/cos split style)
        freqs = torch.cat([freqs, freqs], dim=-1) # [H*W, D] 
        
        return freqs.unsqueeze(0).unsqueeze(2) # [1, N, 1, D]

def apply_rotary_pos_emb(q, k, freqs):
    # Sin/Cos injection
    cos = freqs.cos().to(q.dtype)
    sin = freqs.sin().to(q.dtype)
    
    # Rotate half helper
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_rope=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.use_rope = use_rope

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope_freqs=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.use_rope and rope_freqs is not None:
             # rope_freqs: [1, n_patches, 1, D]
            n_special = N - rope_freqs.shape[1]
            if n_special > 0:
                q_spatial, k_spatial = q[:, n_special:], k[:, n_special:]
                q_special, k_special = q[:, :n_special], k[:, :n_special]
                
                q_spatial, k_spatial = apply_rotary_pos_emb(q_spatial, k_spatial, rope_freqs)
                
                q = torch.cat([q_special, q_spatial], dim=1)
                k = torch.cat([k_special, k_spatial], dim=1)
            else:
                q, k = apply_rotary_pos_emb(q, k, rope_freqs)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0., bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.w1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w2 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = self.act(x1) * x2
        x = self.drop(hidden)
        x = self.w3(x)
        x = self.drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., 
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, 
                 use_rope=False, use_swiglu=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, use_rope=use_rope)
        self.norm2 = norm_layer(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        if use_swiglu:
            self.mlp = SwiGLU(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.SiLU, drop=drop)
        else:
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, rope_freqs=None):
        x = x + self.attn(self.norm1(x), rope_freqs=rope_freqs)
        x = x + self.mlp(self.norm2(x))
        return x

class DinoV3VisionTransformer(nn.Module):
    """
    DINOv3 Vision Transformer with RoPE, SwiGLU, and Register Tokens.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, 
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, 
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., 
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                 num_register_tokens=4, use_swiglu=True, use_rope=True):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_register_tokens = num_register_tokens
        self.use_rope = use_rope

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
        
        if not use_rope:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1 + num_register_tokens, embed_dim))
        else:
            self.pos_embed = None
            self.rope = RotaryEmbedding(embed_dim // num_heads)
            
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                use_rope=use_rope, use_swiglu=use_swiglu)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        nn.init.trunc_normal_(self.register_tokens, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def get_class_and_patch_tokens(self, x):
        """
        Helper for compatibility with DinoTxt code.
        """
        tokens = self.forward_features(x)
        cls_token = tokens[:, 0:1]
        patch_tokens = tokens[:, 1+self.num_register_tokens:]
        return cls_token, None, patch_tokens

    def forward_features(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x) # [B, N, D]
        
        rope_freqs = None
        if self.use_rope:
            h_patches = H // self.patch_embed.patch_size
            w_patches = W // self.patch_embed.patch_size
            rope_freqs = self.rope(x, h_patches, w_patches) 

        cls_tokens = self.cls_token.expand(B, -1, -1)
        register_tokens = self.register_tokens.expand(B, -1, -1)
        
        x = torch.cat((cls_tokens, register_tokens, x), dim=1)
        
        if self.pos_embed is not None:
             if x.shape[1] == self.pos_embed.shape[1]:
                 x = x + self.pos_embed
        
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x, rope_freqs=rope_freqs)

        x = self.norm(x)
        return x

    def forward_head(self, x):
        return self.head(x[:, 0])

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x
