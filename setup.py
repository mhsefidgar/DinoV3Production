from setuptools import setup, find_packages

setup(
    name="dinov3production",
    version="0.1.0",
    description="Production-grade DINOv3 Library",
    author="Antigravity",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "timm",
        "transformers",
        "peft",
        "torchao",
        "onnx",
        "onnxruntime",
        "albumentations",
        "huggingface_hub"
    ],
)
