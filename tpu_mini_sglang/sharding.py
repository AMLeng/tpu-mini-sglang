import jax
import numpy as np
from jax.sharding import Mesh, PartitionSpec


def create_device_mesh(data_parallelism: int, tensor_parallelism: int) -> Mesh:
    devices = jax.devices()
    # Order in the mesh matters because it determines physical layout
    ordered_axes = {"data": data_parallelism, "tensor": tensor_parallelism}

    mesh_shape = []
    mesh_axes = []
    total_size = 1
    for axis, dim in ordered_axes.items():
        mesh_shape.append(dim)
        mesh_axes.append(axis)
        total_size *= dim

    if total_size != len(devices):
        raise ValueError(
            f"Sharding mesh of shape {mesh_shape} does not work for {len(devices)} devices"
        )
    return Mesh(np.asarray(devices).reshape(mesh_shape), axis_names=tuple(mesh_axes))


class ShardingAxisName:
    MLP_TENSOR = "tensor"
    ATTN_HEAD = "tensor"
    VOCAB = ("tensor", "data")


RPA_CACHE_SHARDING = PartitionSpec(None, None, ShardingAxisName.ATTN_HEAD, None, None)
