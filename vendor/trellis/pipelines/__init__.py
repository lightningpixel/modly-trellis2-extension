import importlib

__attributes = {
    "TrellisImageTo3DPipeline": "trellis_image_to_3d",
    "TrellisTextTo3DPipeline": "trellis_text_to_3d",
}

__submodules = ["samplers"]

__all__ = list(__attributes.keys()) + __submodules


def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module = importlib.import_module(f".{__attributes[name]}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


def from_pretrained(path: str, config_file: str = "pipeline.json"):
    """
    Load a pipeline from a model folder or a Hugging Face model hub.

    Args:
        path: The path to the model. Can be either local path or a Hugging Face model name.
    """
    import os
    import json
    is_local = os.path.exists(f"{path}/{config_file}")

    if is_local:
        config_file = f"{path}/{config_file}"
    else:
        from huggingface_hub import hf_hub_download
        config_file = hf_hub_download(path, config_file)

    with open(config_file, 'r') as f:
        config = json.load(f)
    return globals()[config['name']].from_pretrained(path, config_file=os.path.basename(config_file))


if __name__ == '__main__':
    from . import samplers
    from .trellis_image_to_3d import TrellisImageTo3DPipeline
    from .trellis_text_to_3d import TrellisTextTo3DPipeline
