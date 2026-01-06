import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

# --- Helper Utilities ---

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    # Standard RoPE application
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot

# --- Modules ---

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        hp, wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, hp, wp

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.inv_freq = None

    def forward(self, x, H, W):
        device = x.device
        target_dim = self.dim // 2 # Half for H, half for W
        
        if self.inv_freq is None or self.inv_freq.device != device:
            self.inv_freq = 1.0 / (self.base ** (torch.arange(0, target_dim, 2, device=device).float() / target_dim))

        t_h = torch.arange(H, device=device, dtype=self.inv_freq.dtype)
        t_w = torch.arange(W, device=device, dtype=self.inv_freq.dtype)
        
        freqs_h = torch.outer(t_h, self.inv_freq)
        freqs_w = torch.outer(t_w, self.inv_freq)
        
        freqs_h = freqs_h.view(H, 1, -1).repeat(1, W, 1)
        freqs_w = freqs_w.view(1, W, -1).repeat(H, 1, 1)
        
        freqs = torch.cat([freqs_h, freqs_w], dim=-1).flatten(0, 1) 
        freqs = torch.cat([freqs, freqs], dim=-1) 
        return freqs.unsqueeze(0).unsqueeze(0) # [1, 1, N, D]

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

    def forward(self, x, rope_cos=None, rope_sin=None, num_special=0):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.use_rope and rope_cos is not None:
            # Separate [CLS, REG...] from the patch tokens
            q_spec, q_spat = q[:, :, :num_special, :], q[:, :, num_special:, :]
            k_spec, k_spat = k[:, :, :num_special, :], k[:, :, num_special:, :]
            
            # Apply RoPE only to spatial tokens
            q_spat, k_spat = apply_rotary_pos_emb(q_spat, k_spat, rope_cos, rope_sin)
            
            q = torch.cat([q_spec, q_spat], dim=2)
            k = torch.cat([k_spec, k_spat], dim=2)

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
        x1, x2 = self.w1(x), self.w2(x)
        hidden = self.act(x1) * x2
        return self.drop(self.w3(hidden))

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_rope=False, use_swiglu=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, use_rope=use_rope)
        self.norm2 = norm_layer(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        if use_swiglu:
            self.mlp = SwiGLU(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(dim, mlp_hidden_dim),
                act_layer(),
                nn.Dropout(drop),
                nn.Linear(mlp_hidden_dim, dim),
                nn.Dropout(drop)
            )

    def forward(self, x, rope_cos=None, rope_sin=None, num_special=0):
        x = x + self.attn(self.norm1(x), rope_cos, rope_sin, num_special)
        x = x + self.mlp(self.norm2(x))
        return x

class DinoV3VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, 
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, 
                 drop_rate=0., attn_drop_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                 num_register_tokens=4, use_swiglu=True, use_rope=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_register_tokens = num_register_tokens
        self.use_rope = use_rope

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
        
        if use_rope:
            self.rope = RotaryEmbedding(embed_dim // num_heads)
            self.pos_embed = None
        else:
            num_patches = (img_size // patch_size) ** 2
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1 + num_register_tokens, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
                  drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer,
                  use_rope=use_rope, use_swiglu=use_swiglu) for _ in range(depth)])
        
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        nn.init.trunc_normal_(self.cls_token, std=.02)
        nn.init.trunc_normal_(self.register_tokens, std=.02)
        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        B, C, H, W = x.shape
        x, grid_H, grid_W = self.patch_embed(x)
        
        rope_cos, rope_sin = None, None
        num_special = 1 + self.num_register_tokens
        
        if self.use_rope:
            freqs = self.rope(x, grid_H, grid_W)
            rope_cos, rope_sin = freqs.cos(), freqs.sin()

        cls_tokens = self.cls_token.expand(B, -1, -1)
        reg_tokens = self.register_tokens.expand(B, -1, -1)
        x = torch.cat((cls_tokens, reg_tokens, x), dim=1)

        if self.pos_embed is not None:
            x = x + self.pos_embed

        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x, rope_cos=rope_cos, rope_sin=rope_sin, num_special=num_special)
        
        return self.norm(x)

    def forward(self, x):
        x = self.forward_features(x)
        return self.head(x[:, 0])
