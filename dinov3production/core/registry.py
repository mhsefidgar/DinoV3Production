import os
from .vision_transformer import DinoV3VisionTransformer

_MODEL_REGISTRY = {}

def register_model(name):
    def decorator(cls):
        _MODEL_REGISTRY[name] = cls
        return cls
    return decorator

def list_models():
    return list(_MODEL_REGISTRY.keys())

def create_model(model_name, pretrained=False, **kwargs):
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not found. Available: {list_models()}")
    
    model_fn = _MODEL_REGISTRY[model_name]
    model = model_fn(**kwargs)
    
    if pretrained:
        # Placeholder for loading weights from HuggingFace Hub or local cache
        print(f"Loading pretrained weights for {model_name}...")
        # from huggingface_hub import hf_hub_download
        # checkpoint = torch.load(...)
        # model.load_state_dict(checkpoint)
        pass
        
    return model
