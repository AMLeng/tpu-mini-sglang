import functools
import glob
import os
from collections.abc import Callable, Iterator

import jax
from flax import nnx
from huggingface_hub import snapshot_download
from jax.sharding import Mesh
from safetensors import safe_open

from tpu_mini_sglang.model_config import ModelConfig
from tpu_mini_sglang.models.model_base import ModelBase
from tpu_mini_sglang.models.registry import get_model_architecture


def _get_weights_iterator(model_path: str) -> Iterator[tuple[str, jax.Array]]:
    """Download weights from Huggingface and iterate through them from the CPU"""
    hf_weights_folder = snapshot_download(model_path, tqdm_class=None)
    weights_files = glob.glob(os.path.join(hf_weights_folder, "*.safetensors"))
    weights_files.sort()

    for st_file in weights_files:
        # Load all safetensors weights onto the CPU first
        with (
            jax.default_device(jax.local_devices(backend="cpu")[0]),
            safe_open(st_file, framework="flax") as f,
        ):
            for name in f.keys():  # noqa: SIM118
                # We can cheaply access the safetensor keys,
                # and retrieve the tensors themselves one by one
                yield name, f.get_tensor(name)


def load_model(config: ModelConfig, mesh: Mesh) -> ModelBase:
    """Loads model with associated weights from Huggingface"""
    # nnx.eval_shape is the key here which allows us to construct a model without allocation
    # We then use the model's custom load_weights function to load in the weights from Huggingface
    with jax.set_mesh(mesh):
        model_class = get_model_architecture(config.hf_text_config)
        model = nnx.eval_shape(lambda: model_class(config.hf_text_config, mesh=mesh))
        weight_iterator = _get_weights_iterator(config.model_path)
        model.load_weights(weight_iterator)
        return model


def get_jitted_model(config: ModelConfig, mesh: Mesh) -> Callable:
    model = load_model(config, mesh)

    @jax.jit()
    def apply_model(my_model: nnx.Module, *args, **kwargs):
        return my_model(*args, **kwargs)

    return functools.partial(apply_model, model)
