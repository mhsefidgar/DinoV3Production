import torch
import torch.nn as nn

try:
    from peft import get_peft_model, LoraConfig, TaskType
except ImportError:
    get_peft_model = None
    LoraConfig = None
    TaskType = None

def get_lora_config(r=8, lora_alpha=32, lora_dropout=0.1):
    """
    Returns a default LoRA configuration optimized for DINOv3 ViT models.
    """
    if LoraConfig is None:
        raise ImportError("peft library is not installed. Please install it via `pip install peft`.")
    
    # Target common linear layers in ViT attention blocks
    target_modules = ["qkv", "fc1", "fc2", "proj"]
    
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        modules_to_save=["head"], # Save the classifier head
    )
    return config

def wrap_with_lora(model, r=8, lora_alpha=32):
    """
    Wraps a DINOv3 model with LoRA adapters.
    """
    if get_peft_model is None:
        raise ImportError("peft library is not installed.")
        
    config = get_lora_config(r, lora_alpha)
    peft_model = get_peft_model(model, config)
    return peft_model
