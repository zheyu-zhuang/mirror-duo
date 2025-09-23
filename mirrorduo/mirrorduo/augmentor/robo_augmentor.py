import logging
import math
import textwrap
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from colorama import Fore, Style
from einops import rearrange

from mirrorduo.augmentor.background_randomizer import BackgroundRandomizer
from mirrorduo.utils.junk_utils import (
    divider,
    find_all_keys,
    is_matched_key,
    plain_divider,
)
import cv2
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RoboAugmentor:
    """
    RoboAugmentor applies data augmentations to robot learning datasets, focusing on
    image overlays (background randomization) and mirroring (symmetry-based augmentation)
    for both observations and actions.

    This class is designed for visuomotor policy learning, where robust and equivariant
    representations are crucial. It supports augmentation of:
      - **Image observations** (background overlay, horizontal flip)
      - **Proprioception states** (position, rotation, gripper state)
      - **Actions** (position, rotation, gripper action)

    Augmentation Modes
    ------------------
    1. Overlay Augmentation:
       - Randomly replaces image backgrounds with external images.
       - Blends original and background images with a configurable alpha.
       - Warmup scheduling gradually increases augmentation probability and blending.

    2. Mirror Augmentation:
       - `batch_wise`: Flips images, proprioception, and actions in-place
         with a given probability.
       - `action_only`: Mirrors only actions, duplicating the batch by
         interleaving original and mirrored actions.
       - `batch_wise_plus_action_only`: First applies batch-wise mirroring,
         then mirrors and interleaves actions.
    Args:
        shape_meta (Dict[str, Any]): Metadata describing dataset structure.
            Must contain:
                - `obs`: keys for observations (e.g., images, proprio states).
                - `action`: key for actions.
        configs (Optional[Dict[str, Any]]): Configuration for overlay and mirror augmentations.
            Example structure:
            {
                "overlay": {
                    "enable": True,
                    "prob": 0.5,
                    "blend_alpha": 0.5,
                    "warmup_epochs": 10,
                    "input_shape": [3, 84, 84],
                    "background_path": "path/to/backgrounds"
                },
                "mirror": {
                    "enable": True,
                    "mode": "batch_wise",
                    "prob": 0.5
                }
            }

    Call Behavior:
        robo_aug = RoboAugmentor(shape_meta, configs)
        batch, overlay_idx, mirror_idx = robo_aug(batch, epoch_idx)

        - `batch` is the augmented batch (obs + action).
        - `overlay_idx` contains indices of trajectories that received overlay augmentation.
        - `mirror_idx` contains indices of trajectories that were mirrored.

    Raises:
        ValueError: If required keys are missing from `shape_meta` or `batch`, 
                    or if image normalization is invalid.
        AssertionError: If configuration values are out of valid range.

    Example:
        >>> shape_meta = {
        ...     "obs": {"agentview_image": [3, 84, 84], "robot0_pos": [3]},
        ...     "action": [10],
        ... }
        >>> configs = {
        ...     "overlay": {"enable": True, "prob": 0.5, "blend_alpha": 0.5,
        ...                 "input_shape": [3, 84, 84], "background_path": "./bgs"},
        ...     "mirror": {"enable": True, "mode": "batch_wise", "prob": 0.5}
        ... }
        >>> augmentor = RoboAugmentor(shape_meta, configs)
        >>> augmented_batch, overlay_idx, mirror_idx = augmentor(batch, epoch_idx=5)
    """

    def __init__(
        self,
        shape_meta: Dict[str, Any],
        configs: Optional[Dict[str, Any]] = None,
    ):
        assert shape_meta is not None, "shape_meta must be provided"

        self.config = self._validate_configs(configs, allow_empty=False)
        self.image_keys, self.proprio_keys, self.action_keys = self._validate_shape_meta(
            shape_meta
        )

        self.overlay_config = self.config["overlay"]
        self.mirror_config = self.config["mirror"]

        if self.overlay_config["enable"]:
            self.background_randomizer = BackgroundRandomizer(
                self.overlay_config["input_shape"],
                self.overlay_config["background_path"],
            )

        self.rot_flip_mask = torch.tensor([1, -1, -1, -1, 1, 1, -1, 1, 1], dtype=torch.float32)
        self.pos_flip_mask = torch.tensor([-1, 1, 1], dtype=torch.float32)
        # action follows [pos, rot_6d, gripper], i.e. [3, 6, 1]:
        self.action_flip_mask = torch.cat(
            [self.pos_flip_mask, self.rot_flip_mask[:6], torch.ones(1)]
        )
        self.printed_warmup_warning = False
        self.epoch_idx = -1
        print(self)

    def __call__(self, batch: Dict[str, Any], epoch_idx: int = 0) -> Dict[str, Any]:
        assert "obs" in batch, "Batch must contain 'obs' key."
        assert "action" in batch, "Batch must contain 'action' key."
        batch["obs"], overlay_idx = self.apply_overlay(batch["obs"], epoch_idx)
        batch, mirror_idx = self.apply_mirror(batch)
        return batch, overlay_idx, mirror_idx

    def apply_overlay(self, Obs: Dict[str, Any], epoch_idx: int) -> Dict[str, Any]:
        """
        Applies background overlay augmentation to image observations in the batch.

        Args:
            Obs (dict): Batch of observations with image tensors.
            epoch_idx (int): Current training epoch (used for warmup).
            clone_input (bool): If True, does not mutate the original Obs.

        Returns:
            dict: Modified (or original) observation batch.
        """

        cfg = self.overlay_config
        warmup_epochs = cfg.get("warmup_epochs", 0)
        if not cfg.get("enable", False) or cfg.get("prob", 0) == 0:
            return Obs, None

        alpha_min = cfg["blend_alpha"]
        prob_max = cfg["prob"]

        # gradually increase the blending strength and probability
        alpha = max(1 - (1 - alpha_min) * (epoch_idx / warmup_epochs), alpha_min)
        prob = min(prob_max, (1 - prob_max) * (min(epoch_idx, warmup_epochs) / warmup_epochs))

        if prob == 0:
            return Obs, None

        if epoch_idx >= warmup_epochs and not self.printed_warmup_warning:
            logger.info(f"Warmup completed: alpha = {alpha:.2f}, prob = {prob:.2f}. ")
            self.printed_warmup_warning = True

        for key in self.image_keys:
            if key not in Obs:
                raise ValueError(f"Missing image key '{key}' in observations.")

            imgs = Obs[key]
            self.background_randomizer.to_device(imgs.device)
            if imgs.max() > 1.0 or imgs.min() < 0.0:
                raise ValueError(f"Image values for '{key}' must be in [0, 1].")

            # flatten temporal dimension if present
            has_temporal_dim = imgs.ndim == 5
            assert imgs.ndim in [4, 5], f"Invalid image shape: {imgs.shape}"
            imgs = rearrange(imgs, "b c h w -> b 1 c h w") if imgs.ndim == 4 else imgs
            B, T = imgs.shape[:2]

            num_aug = math.ceil(B * prob)
            rand_traj_idx = torch.randperm(B)[:num_aug]

            bgs = self.background_randomizer(num_aug * T)
            bgs = rearrange(bgs, "(b t) c h w -> b t c h w", b=num_aug, t=T)

            if isinstance(alpha, (int, float)):
                blend = alpha
            elif isinstance(alpha, (list, tuple)) and len(alpha) == 2:
                lo, hi = alpha
                blend = torch.rand(num_aug, 1, 1, 1, 1, device=imgs.device) * (hi - lo) + lo
            else:
                raise ValueError("blend_alpha must be a float or a list of two floats")

            imgs[rand_traj_idx] = blend * imgs[rand_traj_idx] + (1 - blend) * bgs
            Obs[key] = imgs.squeeze(1) if not has_temporal_dim else imgs
        return Obs, rand_traj_idx

    def apply_mirror(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if not self.mirror_config.get("enable", False):
            return batch, None
        mode = self.mirror_config["mode"]
        if mode == "batch_wise":
            batch, rand_idx = self._batch_wise_mirror(batch)
        elif mode == "action_only":
            batch, rand_idx = self._action_only_mirror(batch)
        elif mode == "batch_wise_plus_action_only":
            batch, rand_idx = self._batch_wise_mirror(batch)
            batch = self._action_only_mirror(batch)
        else:
            raise ValueError(f"Invalid mirror mode '{mode}'.")
        return batch, rand_idx

    def _action_only_mirror(self, batch: Dict[str, Any]):
        action = batch["action"]
        rand_idx = torch.ones(action.shape[0], dtype=bool)
        mirror_action = self._mirror_action(action, rand_idx, clone_input=True)

        action = action.unsqueeze(1)
        mirror_action = mirror_action.unsqueeze(1)
        action_only = torch.cat([action, mirror_action], dim=1)
        action_only = rearrange(action_only, "b n ... -> (b n) ...")
        batch["action"] = action_only
        return batch, rand_idx

    def _batch_wise_mirror(self, batch: Dict[str, Any]):
        B = batch["action"].shape[0]
        prob = self.mirror_config["prob"]
        # Apply mirroring to each trajectory with probability p
        num_aug = math.ceil(B * prob)
        rand_idx = torch.randperm(B)[:num_aug]

        for key in self.image_keys:
            if key not in batch["obs"]:
                raise ValueError(f"Missing image key '{key}' in observations.")
            batch["obs"][key] = self._mirror_images(batch["obs"][key], rand_idx, clone_input=False)
        for key in self.proprio_keys:
            if key not in batch["obs"]:
                raise ValueError(f"Missing proprioception key '{key}' in observations.")
            batch["obs"][key] = self._mirror_proprio(
                batch["obs"][key], key, rand_idx, clone_input=False
            )
        for key in self.action_keys:
            if key not in batch:
                raise ValueError(f"Missing'{key}' in batch.")
            batch[key] = self._mirror_action(batch[key], rand_idx, clone_input=False)
        return batch, rand_idx

    def _mirror_images(self, images, rand_idx, clone_input):
        if clone_input:
            images = images.clone()
        images[rand_idx] = images[rand_idx].flip(-1)
        # visualize flipped images
        # viz = images[rand_idx]
        # viz_img = viz[0][0].permute(1, 2, 0).cpu().numpy()
        # viz_img = cv2.cvtColor(viz_img, cv2.COLOR_RGB2BGR)
        # cv2.imshow("Flipped Image", viz_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return images

    def _mirror_proprio(self, proprio, key, rand_idx, clone_input):
        if clone_input:
            proprio = proprio.clone()
        self.rot_flip_mask = self.rot_flip_mask.to(proprio.device)
        self.pos_flip_mask = self.pos_flip_mask.to(proprio.device)
        # gripper has identity mapping
        if is_matched_key("gripper", key):
            return proprio
        if is_matched_key("rot", key):
            is_6d = proprio.shape[-1] == 6
            flip_mask = self.rot_flip_mask[:6] if is_6d else self.rot_flip_mask
        elif is_matched_key("pos", key):
            flip_mask = self.pos_flip_mask
        proprio[rand_idx] *= flip_mask
        return proprio

    def _mirror_action(self, action, rand_idx, clone_input):
        flip_mask = self.action_flip_mask.to(action.device)
        if clone_input:
            action = action.clone()
        action[rand_idx] = action[rand_idx] * flip_mask
        return action

    # ---------------------------------------------------------------------------- #
    #                                     Utils                                    #
    # ---------------------------------------------------------------------------- #

    def _validate_shape_meta(
        self, shape_meta: Dict[str, Any]
    ) -> Tuple[List[str], List[str], List[str]]:
        obs_keys = shape_meta.get("obs", {}).keys()
        if not obs_keys:
            raise ValueError("Missing 'obs' in shape_meta.")

        image_keys = find_all_keys("image", obs_keys)
        proprio_keys = find_all_keys(["rot", "pos", "quat", "gripper"], obs_keys, reject=["quat"])
        action_keys = find_all_keys("action", shape_meta.keys())

        if not image_keys:
            raise ValueError("No image keys found in shape_meta.")
        if not proprio_keys:
            raise ValueError("No proprioception keys found. Expected keys with _rot, _pos.")
        if not action_keys:
            raise ValueError("No action key found. Expected a key containing 'action'.")

        # gripper is excluded from proprioception keys for mirroring
        assert len(proprio_keys) + len(image_keys) == len(obs_keys)
        assert len(action_keys) == 1, "Only one action key is expected."

        return image_keys, proprio_keys, action_keys

    @staticmethod
    def _validate_configs(configs: Optional[Dict[str, Any]], allow_empty=False) -> Dict[str, Any]:
        if configs is None:
            configs = {}

        if not allow_empty and not configs:
            raise ValueError("configs cannot be empty.")

        # Ensure nested config sections exist
        overlay_cfg = configs.get("overlay", {})
        mirror_cfg = configs.get("mirror", {})

        config_spec = {
            "overlay": {
                "enable": (bool, False),
                "prob": ((int, float), 0.5),
                "blend_alpha": ((float, list), 0.5),
                "warmup_epochs": ((int, float), 0),
                "input_shape": (list, None),
                "background_path": (str, ""),
            },
            "mirror": {
                "enable": (bool, False),
                "mode": (str, "batch_wise"),
                "prob": ((int, float), 0.5),
            },
        }

        def _check_type(value, expected_type):
            if isinstance(expected_type, tuple):
                return any(isinstance(value, t) for t in expected_type)
            return isinstance(value, expected_type)

        for section_name, section_cfg in [("overlay", overlay_cfg), ("mirror", mirror_cfg)]:
            for key, (expected_type, default_val) in config_spec[section_name].items():
                if key not in section_cfg:
                    logger.warning(
                        f"Missing '{key}' in {section_name} config. Using default: {default_val}"
                    )
                    section_cfg[key] = default_val
                elif not _check_type(section_cfg[key], expected_type):
                    logger.warning(
                        f"Invalid type for '{key}' in {section_name} config. Expected {expected_type.__name__}, got {type(section_cfg[key]).__name__}. Using default: {default_val}"
                    )
                    section_cfg[key] = default_val
            configs[section_name] = section_cfg

        valide_mirror_modes = [
            "batch_wise",
            "action_only",
            "batch_wise_plus_action_only",
        ]
        if configs["mirror"]["mode"] not in valide_mirror_modes:
            raise ValueError(
                f"Invalid mirror mode '{configs['mirror']['mode']}'. Expected one of {valide_mirror_modes}."
            )
        if configs["mirror"]["mode"] == "action_only":
            configs["mirror"]["prob"] = 1.0
            logger.warning("Overriding mirror probability to 1.0 for 'action_only' mode.")

        assert 0 <= configs["overlay"]["prob"] <= 1, "Overlay probability must be between 0 and 1."
        return configs

    def __repr__(self) -> str:
        lines = []

        # Overlay config summary
        overlay_cfg = self.config.get("overlay", {})
        mirror_cfg = self.config.get("mirror", {})

        overlay_status = "✅ Enabled" if overlay_cfg.get("enable", False) else "❌ Disabled"
        blend_alpha = overlay_cfg.get("blend_alpha")
        prob = overlay_cfg.get("prob")
        warmup = overlay_cfg.get("warmup_epochs")

        if isinstance(blend_alpha, (list, tuple)) and len(blend_alpha) == 2:
            blend_eq = f"alpha * IMG + (1 - alpha) * BG, where alpha ~ [{blend_alpha[0]:.1f}, {blend_alpha[1]:.1f}]"
        elif isinstance(blend_alpha, (int, float)):
            blend_eq = f"{blend_alpha:.2f} * IMG + {1 - blend_alpha:.2f} * BG"
        else:
            blend_eq = "Invalid blend_alpha"

        lines.append(divider("RoboAugmentor Summary", char="=", show=False))
        lines.append(f"{Fore.CYAN}Overlay Augmentation:{Style.RESET_ALL}")
        lines.append(f"  {Fore.YELLOW}Status         :{Style.RESET_ALL} {overlay_status}")
        lines.append(
            f"  {Fore.YELLOW}Blending       :{Style.RESET_ALL} {Fore.MAGENTA}{blend_eq}{Style.RESET_ALL}"
        )
        lines.append(f"  {Fore.YELLOW}Probability    :{Style.RESET_ALL} {prob}")
        lines.append(f"  {Fore.YELLOW}Warmup Epochs  :{Style.RESET_ALL} {warmup}")
        lines.append(f"  {Fore.YELLOW}Image Keys     :{Style.RESET_ALL} {self.image_keys}")

        # Mirror config summary
        mirror_status = "✅ Enabled" if mirror_cfg.get("enable", False) else "❌ Disabled"
        mirror_mode = mirror_cfg.get("mode", "batch_wise")
        mirror_prob = mirror_cfg.get("prob", 0.5)
        mode_str = "-".join([x.capitalize() for x in mirror_mode.split("_")])

        desc_map = {
            "batch_wise": f"(Obs., Act.) are mirrored in-place (p = {mirror_prob:.2f}).",
            "action_only": (
                f"Only actions are mirrored and interleaved (p = {mirror_prob:.2f}). "
                "Doubles the action batch size. Suitable for equivariant encoder & non-equivariant decoder."
            ),
            "batch_wise_plus_action_only": (
                f"(Obs., Act.) are mirrored in-place (p = {mirror_prob:.2f}), "
                "then actions are mirrored again and interleaved."
            ),
        }

        lines.append(f"\n{Fore.CYAN}Mirror Augmentation:{Style.RESET_ALL}")
        lines.append(f"  {Fore.YELLOW}Status         :{Style.RESET_ALL} {mirror_status}")
        lines.append(f"  {Fore.YELLOW}Mode           :{Style.RESET_ALL} {mode_str}")
        lines.append(f"  {Fore.YELLOW}Probability    :{Style.RESET_ALL} {mirror_prob}")
        if mirror_mode in ["batch_wise", "batch_wise_plus_action_only"]:
            lines.append(f"  {Fore.YELLOW}Image Keys     :{Style.RESET_ALL} {self.image_keys}")
            lines.append(f"  {Fore.YELLOW}Proprio Keys   :{Style.RESET_ALL} {self.proprio_keys}")
            lines.append(f"  {Fore.YELLOW}Action Keys    :{Style.RESET_ALL} {self.action_keys}")
        elif mirror_mode == "action_only":
            lines.append(f"  {Fore.YELLOW}Action Keys    :{Style.RESET_ALL} {self.action_keys}")
        lines.append(
            f"  {Fore.YELLOW}Behavior       :{Style.RESET_ALL} {textwrap.fill(desc_map.get(mirror_mode, 'Unknown mode'), width=70, subsequent_indent=' ' * 20)}"
        )
        lines.append(plain_divider(char="=", show=False))
        return "\n".join(lines)


def test():
    import copy

    import numpy as np
    from scipy.spatial.transform import Rotation as R

    shape_meta = {
        "obs": {
            "agentview_image": {
                "shape": [3, 84, 84],
                "type": "rgb",
            },
            "robot0_eye_in_hand_image": {
                "shape": [3, 84, 84],
                "type": "rgb",
            },
            "robot0_eef_delta_pos": {
                "shape": [3],
            },
            "robot0_eef_delta_rot": {
                "shape": [9],
            },
            "robot0_gripper_qpos": {
                "shape": [2],
            },
        },
        "action": {
            "shape": [10],
        },
    }

    configs = {
        "overlay": {
            "enable": False,
            "prob": 0.5,
            "blend_alpha": [0.2, 0.8],
            "warmup_epochs": 10,
            "input_shape": [84, 84],
            "background_path": "data/backgrounds",
        },
        "mirror": {
            "enable": True,
            "mode": "action_only",
            "prob": 0.5,
        },
    }

    augmentor = RoboAugmentor(shape_meta=shape_meta, configs=configs)
    B = 10
    T = 2
    rots = R.random(B * T).as_matrix().reshape(-1, 3, 3)
    E = np.diag([-1, 1, 1]).reshape(1, 3, 3)

    flipped_rots = E @ rots @ E
    flipped_rots = flipped_rots.reshape(B, T, 9)
    flipped_rots = flipped_rots[:, :, :6]

    action = torch.rand(B, T, 10)
    action[:, :, 3:9] = torch.from_numpy(rots.reshape(B, T, 9)[:, :, :6])

    imgs = torch.zeros(B, T, 3, 84, 84)
    # half of the images are random noise
    imgs[:, :, :, :, :42] = torch.rand(B, T, 3, 84, 42)

    batch = {
        "obs": {
            "agentview_image": imgs,
            "robot0_eye_in_hand_image": imgs.clone(),
            "robot0_eef_delta_pos": torch.rand(B, T, 3),
            "robot0_eef_delta_rot": torch.from_numpy(rots.reshape(B, T, 9)),
            "robot0_gripper_qpos": torch.rand(B, T, 2),
        },
        "action": action,
    }

    original_batch = copy.deepcopy(batch)

    _, overlay_idx, mirror_idx = augmentor(batch, epoch_idx=11)

    # Test overlay
    augmented_obs = batch["obs"]
    # visualize the augmented images
    for key in augmentor.image_keys:
        if key in augmented_obs:
            imgs = augmented_obs[key]
            B, T = imgs.shape[:2] if imgs.ndim == 5 else (imgs.shape[0], 1)
            imgs = imgs.view(B * T, *imgs.shape[-3:])  # Flatten temporal dimension if present

            fig, axes = plt.subplots(B, T, figsize=(10, 10))
            for i in range(B):
                for j in range(T):
                    axes[i, j].imshow(imgs[i * T + j].permute(1, 2, 0).cpu().numpy())
                    axes[i, j].axis("off")
                    axes[i, j].set_title(f"Batch {i}, Time {j}")
            plt.show()

    # Test mirror
    augmented_action = batch["action"]
    if configs["mirror"]["mode"] == "action_only":
        for i in range(0, 2 * B, 2):
            orig_action = augmented_action[i]
            unflipped_action = augmented_action[i + 1] * augmentor.action_flip_mask
            assert torch.allclose(
                orig_action, unflipped_action
            ), f"Action {i} not mirrored correctly."
    elif configs["mirror"]["mode"] == "batch_wise":
        for i in range(B):
            if mirror_idx[i]:
                aug_action = augmented_action[i]
                aug_rot = aug_action[:, 3:9].reshape(-1, 6).numpy()
                flipped_rot = flipped_rots[i]
                assert np.allclose(aug_rot, flipped_rot), f"Action {i} not mirrored correctly."
    else:
        pass

    # Test proprioception
    augmented_obs = batch["obs"]
    for key in augmentor.proprio_keys:
        if key in augmented_obs:
            aug_proprio = augmented_obs[key]
            original_proprio = original_batch["obs"][key]
            if is_matched_key("rot", key):
                assert torch.allclose(
                    aug_proprio[mirror_idx], original_proprio[mirror_idx] * augmentor.rot_flip_mask
                )
            elif is_matched_key("pos", key):
                assert torch.allclose(
                    aug_proprio[mirror_idx], original_proprio[mirror_idx] * augmentor.pos_flip_mask
                )

    # Test mirrored images
    for key in augmentor.image_keys:
        if key in augmented_obs:
            aug_images = augmented_obs[key]
            original_images = original_batch["obs"][key]
            assert torch.allclose(
                aug_images[mirror_idx], original_images[mirror_idx].flip(-1)
            ), f"Images {i} not mirrored correctly."

    # visualize_flipped_images

    for key in augmentor.image_keys:
        if key in augmented_obs:
            imgs = augmented_obs[key]
            B, T = imgs.shape[:2] if imgs.ndim == 5 else (imgs.shape[0], 1)
            imgs = imgs.view(B * T, *imgs.shape[-3:])
            # Flatten temporal dimension if present
            fig, axes = plt.subplots(B, T, figsize=(10, 10))
            for i in range(B):
                for j in range(T):
                    axes[i, j].imshow(imgs[i * T + j].permute(1, 2, 0).cpu().numpy())
                    axes[i, j].axis("off")
                    axes[i, j].set_title(f"Batch {i}, Time {j}")
            plt.show()

    print("All tests passed!")


if __name__ == "__main__":
    test()
