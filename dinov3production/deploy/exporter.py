import torch
import os
from ..core.registry import create_model

class EdgeExporter:
    """
    Handles converting DINOv3 models to deployment formats (ONNX).
    """
    def __init__(self, model_or_name):
        if isinstance(model_or_name, str):
            self.model = create_model(model_or_name, pretrained=False)
        else:
            self.model = model_or_name
        
        self.model.eval()

    def to_onnx(self, output_path="model.onnx", opset_version=17):
        """
        Experts the model to ONNX with specific opset version for Transformer compatibility.
        """
        print(f"Exporting model to {output_path} with opset {opset_version}...")
        
        # Create dummy input
        # Note: DINOv3 usually expects 224x224 or divisible by patch size
        dummy_input = torch.randn(1, 3, 224, 224)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        torch.onnx.export(
            self.model, 
            dummy_input, 
            output_path,
            opset_version=opset_version,
            input_names=['image'],
            output_names=['features'],
            dynamic_axes={
                'image': {0: 'batch_size'},
                'features': {0: 'batch_size'}
            }
        )
        print("Export successful.")
