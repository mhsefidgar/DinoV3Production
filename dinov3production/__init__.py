from .core.registry import create_model, list_models, register_model
from .core.vision_transformer import DinoV3VisionTransformer
# Import models to ensure they register themselves
from .core.models import vit

__version__ = "0.1.0"
