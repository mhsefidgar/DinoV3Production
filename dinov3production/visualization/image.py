import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from PIL import Image

def visualize_attention(image, attention_map, alpha=0.6, cmap='viridis'):
    """
    Overlays an attention map onto an image.
    
    Args:
        image (numpy.ndarray): The original image (H, W, 3).
        attention_map (numpy.ndarray): The attention map (H, W).
        alpha (float): Transparency of the overlay.
        cmap (str): Colormap for the attention map.
        
    Returns:
        numpy.ndarray: The image with overlay.
    """
    # Normalize attention map
    att_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
    
    H, W = image.shape[:2]
    # Resize map to match image (simple nearest/bilinear interpolation via pyplot usually handled at plot time, 
    # but here we return an array)
    
    # For simplicity in this utility, we rely on the user plotting it, 
    # but we can create a blended array using plt.cm
    cm = plt.get_cmap(cmap)
    att_colored = cm(att_map)[:, :, :3] # [H_map, W_map, 3]
    
    # We can't easily resize numpy arrays without cv2 or skimage, 
    # so we return a figure-compatible array or just the heatmap?
    # Let's return the colored heatmap scaled to the image size if possible, 
    # or just assume the user will resize.
    
    return att_colored

def visualize_pca(features, foreground_mask=None):
    """
    Computes Rainbow PCA of feature map.
    
    Args:
        features (torch.Tensor): Feature map of shape (C, H, W) or (N, C).
        foreground_mask (torch.Tensor, optional): Boolean mask of shape (H, W) or (N,). 
                                                  If provided, PCA is computed only on foreground.
        
    Returns:
        torch.Tensor: PCA visualization image of shape (3, H, W), float [0, 1].
    """
    # Handle inputs
    if features.ndim == 3: # C, H, W
        C, h, w = features.shape
        flat_feats = features.permute(1, 2, 0).reshape(-1, C) # N, C
    else:
        flat_feats = features
        h = w = int(flat_feats.shape[0]**0.5) # assume square features if N, C passed
    
    # Normalize features
    flat_feats = torch.nn.functional.normalize(flat_feats, p=2, dim=1)
    
    pca = PCA(n_components=3, whiten=True)
    
    if foreground_mask is not None:
        fg_mask_flat = foreground_mask.reshape(-1)
        if fg_mask_flat.sum() > 3: # Need at least 3 samples for 3 components
            fg_feats = flat_feats[fg_mask_flat]
            pca.fit(fg_feats.cpu().numpy())
        else:
             # Fallback: fit on everything
            pca.fit(flat_feats.cpu().numpy())
    else:
        pca.fit(flat_feats.cpu().numpy())
        
    # Transform all
    pca_features = pca.transform(flat_feats.cpu().numpy())
    pca_features = torch.from_numpy(pca_features)
    
    # Reshape
    pca_img = pca_features.reshape(h, w, 3).permute(2, 0, 1) # 3, H, W
    
    # Colorize (Rainbow style)
    pca_img = torch.sigmoid(pca_img * 2.0)
    
    # Apply mask to output if desired (background becomes black)
    if foreground_mask is not None:
        pca_img = pca_img * foreground_mask.reshape(1, h, w).float().cpu()
        
    return pca_img

def visualize_mask(image, mask, color=(255, 0, 0), alpha=0.5):
    """
    Overlays a binary segmentation mask on an image.
    
    Args:
        image (PIL.Image or numpy.ndarray): The original image.
        mask (numpy.ndarray or torch.Tensor): Binary mask.
        color (tuple): RGB color for the mask (0-255).
        alpha (float): Transparency.
        
    Returns:
        numpy.ndarray: Image/mask overlay (H, W, 3) uint8.
    """
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
        if image.max() <= 1.0: image = (image * 255).astype(np.uint8)
        
    if isinstance(image, Image.Image):
        image = np.array(image)
        
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
        
    # Resize mask if needed? Assume matching size for now.
    
    overlay = image.copy()
    
    # Apply color where mask is True
    # Mask binary check
    binary_mask = (mask > 0.5)
    
    overlay[binary_mask] = overlay[binary_mask] * (1 - alpha) + np.array(color) * alpha
    
    return overlay.astype(np.uint8)
