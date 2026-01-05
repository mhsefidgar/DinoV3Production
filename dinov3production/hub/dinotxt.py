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
    def __init__(self):
        super().__init__()
        self.patch_size = 14

class MockVisualModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = MockBackbone()
        
    def get_class_and_patch_tokens(self, x):
        # x: [B, 3, H, W]
        B = x.shape[0]
        H, W = x.shape[2:]
        h, w = H // self.backbone.patch_size, W // self.backbone.patch_size
        D = 1024 # Embedding dim
        
        # Returns: cls_tokens, _, patch_tokens
        # cls: [B, 1, D]
        # patch: [B, N, D]
        cls_tokens = torch.randn(B, 1, D).to(x.device)
        patch_tokens = torch.randn(B, h*w, D).to(x.device)
        return cls_tokens, None, patch_tokens

class MockDinoTxtModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual_model = MockVisualModel()

    def encode_image(self, image):
        # [B, 3, H, W] -> [B, 768] (or whatever dim)
        return torch.randn(image.shape[0], 768).to(image.device)

    def encode_text(self, text):
        # The user code expects [B, 2*D] or similar to split it?
        # User: feats = feats[:, feats.shape[1] // 2 :]
        # This implies it returns [CLS, ....] and they want the second half?
        # Or maybe [Projection1, Projection2]?
        # Let's return [B, 2048] so splitting gives [B, 1024]
        return torch.randn(text.shape[0], 2048).to(text.device)
    
    def forward(self, x):
        return self.encode_image(x)

def dinov3_vitl16_dinotxt_tet1280d20h24l(pretrained=True):
    """
    Loads the DINOv3 Text model and tokenizer.
    """
    print("Loading DINOv3-Text model (Mock)...")
    model = MockDinoTxtModel()
    tokenizer = MockTokenizer()
    return model, tokenizer
