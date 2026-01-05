import torch
import torch.nn.functional as F
import functools
from torch import Tensor

@torch.compile(disable=True)
def propagate(
    current_features: Tensor,  # [h", w", D]
    context_features: Tensor,  # [t, h, w, D]
    context_probs: Tensor,  # [t, h, w, M]
    neighborhood_mask: Tensor,  # [h", w", h, w]
    topk: int = 5,
    temperature: float = 0.2,
) -> Tensor:
    """
    Propagates segmentation masks using space-time correspondence.
    """
    t, h, w, M = context_probs.shape

    # Compute similarity current -> context
    # [h", w", D] @ [t, h, w, D].T -> [h", w", t, h, w]
    dot = torch.einsum(
        "ijd, tuvd -> ijtuv",
        current_features,  # [h", w", D]
        context_features,  # [t, h, w, D]
    )  # [h", w", t, h, w]

    # Restrict focus to local neighborhood
    dot = torch.where(
        neighborhood_mask[:, :, None, :, :],  # [h", w", 1, h, w]
        dot,  # [h", w", t, h, w]
        -torch.inf,
    )

    # Select top-k patches inside the neighborhood
    # Reshape to [h"w", thw] for easier topk
    dot_flat = dot.flatten(2, -1).flatten(0, 1)  # [h"w", thw]
    
    # We need to handle potential issues if K > thw, though unlikely in practice
    current_topk = min(topk, dot_flat.shape[1])
    
    k_th_largest = torch.topk(dot_flat, dim=1, k=current_topk).values  # [h"w", k]
    
    # Mask out non-top-k
    dot_flat = torch.where(
        dot_flat >= k_th_largest[:, -1:],  # Use the k-th value as threshold
        dot_flat,
        -torch.inf,
    )

    # Propagate probabilities
    weights = F.softmax(dot_flat / temperature, dim=1)  # [h"w", thw]
    
    # [h"w", thw] @ [thw, M] -> [h"w", M]
    context_probs_flat = context_probs.flatten(0, 2)
    current_probs = torch.mm(weights, context_probs_flat)

    # Re-normalize (softmax sum is 1, but math safety)
    current_probs = current_probs / current_probs.sum(dim=1, keepdim=True).clamp(min=1e-6)

    return current_probs.unflatten(0, (current_features.shape[0], current_features.shape[1]))  # [h", w", M]

@functools.lru_cache()
def make_neighborhood_mask(h: int, w: int, size: float, shape: str = "circle") -> Tensor:
    """
    Creates a boolean mask defining a local neighborhood for each patch.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    ij = torch.stack(
        torch.meshgrid(
            torch.arange(h, dtype=torch.float32, device=device),
            torch.arange(w, dtype=torch.float32, device=device),
            indexing="ij",
        ),
        dim=-1,
    )  # [h, w, 2]
    
    if shape == "circle":
        ord = 2
    elif shape == "square":
        ord = torch.inf
    else:
        raise ValueError(f"Invalid {shape=}")
        
    # Distance from every patch (i,j) to every other patch (u,v)
    # [h, w, 1, 1, 2] - [1, 1, h, w, 2]
    norm = torch.linalg.vector_norm(
        ij[:, :, None, None, :] - ij[None, None, :, :, :],
        ord=ord,
        dim=-1,
    )  # [h, w, h, w]
    
    mask = norm <= size
    return mask
