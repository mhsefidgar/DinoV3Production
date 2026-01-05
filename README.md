# DINOv3Production Library

A production-grade library for DINOv3, featuring high-level APIs, LoRA fine-tuning, quantization tools, and edge deployment support.

## Features
- **Core**: Optimized Vision Transformer 3.0 implementation with Register Tokens and Gram Anchoring.
- **Micro-Finetuning**: Integrated `peft` support for LoRA.
- **Quantization**: INT8/INT4 support via `torchao`.
- **Edge Deployment**: One-line ONNX export for TensorRT/CoreML.

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### 1. Load a Model
```python
import torch
from dinov3Production import create_model

# Load DINOv3 ViT-Base with 14x14 patches
model = create_model('dinov3_vitb14', pretrained=False)
model.eval()

img = torch.randn(1, 3, 224, 224)
output = model(img)
print(output.shape)  # [1, 1000] (if num_classes=1000)
```

### 2. Fine-tune with LoRA
Fine-tune only 1% of the parameters.
```python
from dinov3Production.finetune.peft import wrap_with_lora

model = create_model('dinov3_vitb14')
peft_model = wrap_with_lora(model, r=8)
peft_model.print_trainable_parameters()
```

### 3. Quantize for Edge
Compress the model to 4-bit weights.
```python
from dinov3Production.quantize.quantizer import Quantizer

quantizer = Quantizer(model)
quantized_model = quantizer.to_int4()
```

### 4. Export to ONNX
```python
from dinov3Production.deploy.exporter import EdgeExporter

exporter = EdgeExporter(model)
exporter.to_onnx("output/dinov3_vitb14.onnx")
```

## Architecture
- `dinov3Production.core`: Model definitions and checking.
- `dinov3Production.finetune`: Adapters for efficient training.
- `dinov3Production.quantize`: Hardware-aware compression.
- `dinov3Production.deploy`: Conversion tools for production engines.
