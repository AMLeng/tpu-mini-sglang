from typing import Any

import jax
import torchax  # type: ignore[import-untyped]
import torchax.interop  # type: ignore[import-untyped]
from jax.tree_util import register_pytree_node
from transformers import AutoModelForCausalLM, cache_utils, modeling_outputs


def output_flatten(v):
    # Separate the keys (names) from the actual tensor values
    # v.__dict__ contains all attributes of the dataclass
    children = []
    aux_metadata = []

    for key, value in v.items():
        if value is not None:
            children.append(value)
            aux_metadata.append(key)
        else:
            # Keep track of which fields were None
            aux_metadata.append((key, None))

    return tuple(children), aux_metadata


def output_unflatten(aux_metadata, children):
    # Reconstruct the dictionary for the constructor
    children_iter = iter(children)
    kwargs: dict[str, Any] = {}

    for item in aux_metadata:
        if isinstance(item, tuple) and item[1] is None:
            kwargs[item[0]] = None
        else:
            kwargs[item] = next(children_iter)

    return modeling_outputs.CausalLMOutputWithPast(**kwargs)


register_pytree_node(
    modeling_outputs.CausalLMOutputWithPast,
    output_flatten,
    output_unflatten,
)


def _flatten_dynamic_cache(dynamic_cache):
    keys = tuple(layer.keys for layer in dynamic_cache.layers)
    values = tuple(layer.values for layer in dynamic_cache.layers)

    return (keys, values), None


def _unflatten_dynamic_cache(aux, children):
    keys, values = children
    cache = cache_utils.DynamicCache()

    for layer_idx, (k, v) in enumerate(zip(keys, values, strict=True)):
        cache.update(k, v, layer_idx)

    return cache


register_pytree_node(
    cache_utils.DynamicCache,
    _flatten_dynamic_cache,
    _unflatten_dynamic_cache,
)


def get_jitted_function_from_pretrained(model_path: str):
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="bfloat16")
    weights, func = torchax.extract_jax(model)

    def apply_model(input_ids):
        return func(weights, (input_ids,), {"use_cache": False})

    return jax.jit(apply_model)
