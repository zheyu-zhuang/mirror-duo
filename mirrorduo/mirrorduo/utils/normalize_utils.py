import numpy as np
from equi_diffpo.common.pytorch_util import dict_apply_reduce, dict_apply_split
from equi_diffpo.model.common.normalizer import SingleFieldLinearNormalizer
import torch


def mirrorduo_action_normalizer_from_stat(stat, enable_mirror=True):
    """
    Create a normalizer from statistics.
    Args:
        stat (dict): A dictionary containing statistics with keys 'max', 'min', 'mean', and 'std'.
    Returns:
        SingleFieldLinearNormalizer: A normalizer object.
    "
    """

    result = dict_apply_split(
        stat, lambda x: {"pos": x[..., :3], "rot": x[..., 3:-1], "gripper": x[..., -1:]}
    )

    output_max = 1
    output_min = -1
    range_eps = 1e-7

    def get_pos_param_info(stat):
        # -1, 1 normalization
        input_max = stat["max"]
        input_min = stat["min"]
        if enable_mirror:
            abs_max = np.max([np.abs(stat["max"][0]), np.abs(stat["min"][0])])
            input_max[0] = abs_max
            input_min[0] = -abs_max
        scale, offset = get_scale_and_offset(
            input_min, input_max, output_max=output_max, output_min=output_min, range_eps=range_eps
        )
        return {"scale": scale, "offset": offset}, stat

    def get_rot_param_info(stat):
        example = stat["max"]
        scale = np.ones_like(example)
        offset = np.zeros_like(example)
        info = {
            "min": np.full_like(example, -1),
            "max": np.ones_like(example),
            "mean": np.zeros_like(example),
            "std": np.ones_like(example),
        }
        return {"scale": scale, "offset": offset}, info

    def get_gripper_param_info(stat):
        # -1, 1 normalization
        scale, offset = get_scale_and_offset(
            stat["min"],
            stat["max"],
            output_max=output_max,
            output_min=output_min,
            range_eps=range_eps,
        )
        return {"scale": scale, "offset": offset}, stat

    pos_param, pos_info = get_pos_param_info(result["pos"])
    rot_param, rot_info = get_rot_param_info(result["rot"])
    gripper_param, gripper_info = get_gripper_param_info(result["gripper"])

    param = dict_apply_reduce(
        [pos_param, rot_param, gripper_param], lambda x: np.concatenate(x, axis=-1)
    )
    info = dict_apply_reduce(
        [pos_info, rot_info, gripper_info], lambda x: np.concatenate(x, axis=-1)
    )

    return SingleFieldLinearNormalizer.create_manual(
        scale=param["scale"], offset=param["offset"], input_stats_dict=info
    )


def mirrorduo_pos_normalizer_from_stat(stat, enable_mirror=True):
    output_max = 1
    output_min = -1
    range_eps = 1e-7

    def get_pos_param_info(stat):
        # -1, 1 normalization
        input_max = stat["max"]
        input_min = stat["min"]
        if enable_mirror:
            abs_max = np.max([np.abs(stat["max"][0]), np.abs(stat["min"][0])])
            input_max[0] = abs_max
            input_min[0] = -abs_max
        scale, offset = get_scale_and_offset(
            input_min, input_max, output_max=output_max, output_min=output_min, range_eps=range_eps
        )
        return {"scale": scale, "offset": offset}, stat

    param, info = get_pos_param_info(stat)

    return SingleFieldLinearNormalizer.create_manual(
        scale=param["scale"], offset=param["offset"], input_stats_dict=info
    )


def get_scale_and_offset(input_min, input_max, output_max=1, output_min=-1, range_eps=1e-7):
    input_range = input_max - input_min
    ignore_dim = input_range < range_eps
    input_range[ignore_dim] = output_max - output_min
    scale = (output_max - output_min) / input_range
    offset = output_min - scale * input_min
    offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]
    return scale, offset


def array_to_stats(arr):
    if isinstance(arr, np.ndarray):
        stat = {
            "min": np.min(arr, axis=0),
            "max": np.max(arr, axis=0),
            "mean": np.mean(arr, axis=0),
            "std": np.std(arr, axis=0),
        }
    elif isinstance(arr, torch.Tensor):
        if arr.device.type == "cuda":
            arr = arr.cpu()
        stat = {
            "min": torch.min(arr, dim=0).values.numpy(),
            "max": torch.max(arr, dim=0).values.numpy(),
            "mean": torch.mean(arr, dim=0).numpy(),
            "std": torch.std(arr, dim=0).numpy(),
        }
    return stat
