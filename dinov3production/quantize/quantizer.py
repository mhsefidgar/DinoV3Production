import torch
import torch.nn as nn
import warnings

# Try to import torchao
try:
    from torchao.quantization import quantize_, int8_dynamic_activation_int8_weight, int4_weight_only
except ImportError:
    quantize_ = None

class Quantizer:
    """
    DINOv3 Quantization Toolkit.
    Supports INT8 Dynamic and INT4 Weight-Only quantization for edge deployment.
    """
    def __init__(self, model):
        self.model = model

    def to_int8(self):
        """
        Apply dynamic INT8 quantization (weights INT8, activations INT8 dynamic).
        Good for CPU inference.
        """
        if quantize_ is None:
            warnings.warn("torchao not installed. Falling back to torch.quantization (legacy) or skipping.")
            return self.model
            
        print("Quantizing to INT8 (Dynamic)...")
        quantize_(self.model, int8_dynamic_activation_int8_weight())
        return self.model

    def to_int4(self):
        """
        Apply INT4 weight-only quantization.
        Ideal for reducing VRAM usage on consumer GPUs.
        """
        if quantize_ is None:
            warnings.warn("torchao not installed. Skipping INT4 quantization.")
            return self.model

        print("Quantizing to INT4 (Weight-Only)...")
        quantize_(self.model, int4_weight_only())
        return self.model

    def export_pt2e(self, example_input):
        """
        Export using PyTorch 2.0 Export (PT2E) flow.
        """
        print("Exporting via torch.export (PT2E)...")
        exported_program = torch.export.export(self.model, (example_input,))
        return exported_program
