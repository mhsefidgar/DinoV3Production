from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import torch

def make_classification_eval_transform(resize_size=256, crop_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Creates standard classification evaluation transforms:
    Resize -> CenterCrop -> ToTensor -> Normalize
    """
    return transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

def resize_to_patch_multiple(image: Image, patch_size: int, target_size: int = None) -> torch.Tensor:
    """
    Resizes an image (or mask) such that its dimensions are divisible by the patch_size.
    Preserves aspect ratio if target_size is not None by resizing the smaller edge to target_size (approx).
    
    Args:
        image: PIL Image
        patch_size: int, e.g. 14 or 16
        target_size: int, optional. If provided, scales image so at least one dim is around target_size.
    """
    w, h = image.size
    
    if target_size:
        # Scale logic from user snippet:
        # h_patches = int(image_size / patch_size)
        # w_patches = int((w * image_size) / (h * patch_size))
        # This logic essentially scales H to target_size, then W proportionally
        h_patches = target_size // patch_size
        scale_factor = target_size / (h * patch_size) 
        w_patches = int((w * target_size) / (h * patch_size))
    else:
        h_patches = h // patch_size
        w_patches = w // patch_size
        
    new_h = h_patches * patch_size
    new_w = w_patches * patch_size
    
    return TF.to_tensor(TF.resize(image, (new_h, new_w)))

def quantize_mask(mask_tensor: torch.Tensor, patch_size: int = 16) -> torch.Tensor:
    """
    Quantizes a mask tensor to patch grid using a box filter (average pooling).
    
    Args:
        mask_tensor: [C, H, W] tensor (usually C=1 for mask)
        patch_size: int, stride of the quantization.
        
    Returns:
        Quantized mask with shape [H//patch_size, W//patch_size]
    """
    if mask_tensor.ndim == 2:
        mask_tensor = mask_tensor.unsqueeze(0)
    if mask_tensor.ndim == 3 and mask_tensor.shape[0] > 1:
        # Just use first channel?
        mask_tensor = mask_tensor[0:1]
        
    weight = torch.full((1, 1, patch_size, patch_size), 1.0 / (patch_size * patch_size))
    if mask_tensor.device != weight.device:
        weight = weight.to(mask_tensor.device)
        
    quantized = torch.nn.functional.conv2d(mask_tensor.unsqueeze(0), weight, stride=patch_size, bias=None)
    return quantized.squeeze().detach()
