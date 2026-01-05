import torch
from torch import nn

class MockTokenizer:
    def tokenize(self, texts):
        # Dummy tokenizer returning random tokens
        # In real scenario: usage of CLIP tokenizer or similar
        # Returns [B, 77]
        return torch.randint(0, 1000, (len(texts), 77))
        
    def __call__(self, texts):
        return self.tokenize(texts)

class MockBackbone(nn.Module):
    def __init__(self, patch_size=14):
        super().__init__()
        self.patch_size = patch_size

class MockVisualModel(nn.Module):
    def __init__(self, embed_dim=1024, patch_size=14):
        super().__init__()
        self.backbone = MockBackbone(patch_size=patch_size)
        self.embed_dim = embed_dim
        
    def get_class_and_patch_tokens(self, x):
        # x: [B, 3, H, W]
        B = x.shape[0]
        H, W = x.shape[2:]
        h, w = H // self.backbone.patch_size, W // self.backbone.patch_size
        D = self.embed_dim
        
        # Returns: cls_tokens, _, patch_tokens
        # cls: [B, 1, D]
        # patch: [B, N, D]
        cls_tokens = torch.randn(B, 1, D).to(x.device)
        patch_tokens = torch.randn(B, h*w, D).to(x.device)
        return cls_tokens, None, patch_tokens

class MockDinoTxtModel(nn.Module):
    def __init__(self, embed_dim=1024, patch_size=14):
        super().__init__()
        self.visual_model = MockVisualModel(embed_dim=embed_dim, patch_size=patch_size)
        self.text_embed_dim = embed_dim # Simplify for mock

    def encode_image(self, image):
        # [B, 3, H, W] -> [B, D]
        return torch.randn(image.shape[0], self.visual_model.embed_dim).to(image.device)

    def encode_text(self, text):
        # The user code expects [B, 2*D] or similar to split it?
        # User: feats = feats[:, feats.shape[1] // 2 :]
        # This implies it returns [CLS, ....] and they want the second half?
        # Or maybe [Projection1, Projection2]?
        # Let's return [B, 2*D] so splitting gives [B, D]
        return torch.randn(text.shape[0], 2 * self.text_embed_dim).to(text.device)
    
    def forward(self, x):
        return self.encode_image(x)

def dinov3_vitl16_dinotxt_tet1280d20h24l(pretrained=True):
    """
    Loads the DINOv3 Text model and tokenizer (Large, Patch 16).
    """
    print("Loading DINOv3-Text model (Mock Large)...")
    model = MockDinoTxtModel(embed_dim=1024, patch_size=16)
    tokenizer = MockTokenizer()
    return model, tokenizer

def dinov3_vits14_dinotxt_swiglu_highres(pretrained=True):
    """
    Loads the DINOv3 Text model and tokenizer (Small, Patch 14).
    """
    print("Loading DINOv3-Text model (Mock Small)...")
    # ViT-S usually has embed_dim=384, patch_size=14
    model = MockDinoTxtModel(embed_dim=384, patch_size=14)
    tokenizer = MockTokenizer()
    return model, tokenizer
