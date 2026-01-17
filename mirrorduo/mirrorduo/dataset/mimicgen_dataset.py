import json
import os
import warnings
from typing import Literal, Optional

import cv2
import lmdb
import numpy as np
import torch

from equi_diffpo.common.normalize_util import (
    array_to_stats,
    get_identity_normalizer_from_stat,
    get_image_range_normalizer,
    get_range_normalizer_from_stat,
)
from equi_diffpo.model.common.normalizer import LinearNormalizer
from mirrorduo.utils.junk_utils import is_matched_key
from mirrorduo.utils.normalize_utils import (
    mirrorduo_action_normalizer_from_stat,
    mirrorduo_pos_normalizer_from_stat,
)

ImageFormat = Literal["HWC", "CHW"]


def decode_jpg_bytes(
    buf: bytes,
    image_size: Optional[int] = None,
    *,
    bgr_to_rgb: bool = False,
    to_float: bool = True,
    fmt: ImageFormat = "CHW",
) -> np.ndarray:
    arr = np.frombuffer(buf, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # HWC uint8, BGR
    if img is None:
        raise ValueError("cv2.imdecode failed (buf may be corrupted)")

    if image_size is not None and (img.shape[0] != image_size or img.shape[1] != image_size):
        img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)

    if bgr_to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if to_float:
        img = img.astype(np.float32) / 255.0

    if fmt == "CHW":
        img = np.moveaxis(img, -1, 0)

    return img


class MimicGenDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        shape_meta: dict,
        dataset_path: str,
        action_mode: str,
        enable_mirror: bool = True,
        horizon=16,
        val_ratio=0.02,
        n_demo=None,
        n_obs_steps=1,
        # LMDB read knobs
        lmdb_readahead=False,
        cache_dir=None,
    ):
        # Obs
        self.n_obs_steps = int(n_obs_steps)
        if self.n_obs_steps < 1:
            raise ValueError("n_obs_steps must be >= 1")

        # Dataset params
        self.horizon = int(horizon)
        self.val_ratio = float(val_ratio)
        self.action_mode = action_mode
        self.lmdb_readahead = bool(lmdb_readahead)
        self.rgb_keys, self.lowdim_keys = self._get_keys(shape_meta)

        # Cache paths
        folder = os.path.dirname(os.path.abspath(os.path.expanduser(dataset_path)))
        parent_name = os.path.basename(folder)
        if cache_dir is None:
            cache_dir = os.path.join(folder, f"{parent_name}_84_lmdb")

        self.cache_dir = cache_dir
        self.lmdb_path = os.path.join(self.cache_dir, "images.lmdb")
        self._lmdb_env = None  # lazy-open per worker/process

        # Load dataset
        self._check_cache(dataset_path=dataset_path)
        self._load_metadata()
        self._load_action()
        self._load_lowdim()

        # set demos
        self.set_active_demos(n_demo)
        self.mode_set = False
        self.start_idx = 0
        self.end_idx = 0
        self.enable_mirror = enable_mirror

    def set_active_demos(self, n_demo=None):
        # active episode count
        n = self.n_demo_all if n_demo is None else int(n_demo)
        if n < 1 or n > self.n_demo_all:
            raise ValueError(f"n_demo must be in [1, {self.n_demo_all}], got {n}")
        self.n_demo_active = n

        # episode lengths + cumulative step indices
        self.episode_lengths_active = self.episode_lengths_all[:n]
        self.cum_lengths_active = np.cumsum([0] + self.episode_lengths_active).astype(np.int64)
        self.n_samples_active = int(self.cum_lengths_active[-1])

        # episode-wise split
        self.n_train_episodes = int(n * (1.0 - self.val_ratio))
        self.n_eval_episodes = n - self.n_train_episodes

        self.train_length = int(self.cum_lengths_active[self.n_train_episodes])
        self.eval_length = int(self.n_samples_active - self.train_length)

    def set_mode(self, mode: str):
        if mode == "train":
            start, end = 0, self.train_length
        elif mode == "eval":
            start, end = self.train_length, self.train_length + self.eval_length
        elif mode == "all":
            start, end = 0, self.n_samples_active
        else:
            raise ValueError(f"mode must be one of ['train','eval','all'], got {mode!r}")

        self.start_idx = int(start)
        self.end_idx = int(end)
        self.mode_set = True

    def __len__(self):
        return int(self.end_idx - self.start_idx)

    def __getitem__(self, idx):
        if not self.mode_set:
            raise RuntimeError("Dataset mode not set. set_mode('mode').")
        obs_idx = int(self.start_idx + idx)
        eps_index, obs_indices, action_indices = self.sampler(obs_idx)
        obs = self.get_obs(eps_index, obs_indices)
        action = np.stack([self.action[a_idx] for a_idx in action_indices], axis=0)
        action = action.reshape(self.horizon, -1)
        sample = {"obs": obs, "action": action.astype(np.float32)}
        return sample

    def get_normalizer(self) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # action
        stat = array_to_stats(self.action)
        if stat["mean"].shape[-1] > 10:
            raise NotImplementedError("Functions for dual-arm not yet implemented.")
        # Note that stats loads based on the action key, but "action" is hardcoded, for compatibility
        normalizer["action"] = mirrorduo_action_normalizer_from_stat(stat, self.enable_mirror)

        # obs
        for key in self.lowdim_keys:
            stat = array_to_stats(self.lowdim[key])
            if is_matched_key("pos", key):
                this_normalizer = mirrorduo_pos_normalizer_from_stat(stat, self.enable_mirror)
            elif is_matched_key("rot", key):
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif is_matched_key("qpos", key):
                this_normalizer = get_range_normalizer_from_stat(stat)
            else:
                raise RuntimeError("unsupported")
            normalizer[key] = this_normalizer

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    # ------------------------------- LMDB Backend ------------------------------- #
    def __getstate__(self):
        d = self.__dict__.copy()
        d["_lmdb_env"] = None
        return d

    def _get_lmdb_env(self):
        if self._lmdb_env is None:
            self._lmdb_env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=self.lmdb_readahead,
                meminit=False,
                subdir=False,
                max_readers=2048,
            )
        return self._lmdb_env

    def get_obs(self, eps_index, obs_indices):
        obs = {}
        env = self._get_lmdb_env()

        with env.begin(write=False) as txn:
            for img_key in self.rgb_keys:
                imgs = []
                for gidx in obs_indices:
                    k = f"{img_key}/{int(gidx):08d}".encode("ascii")
                    buf = txn.get(k)
                    if buf is None:
                        raise KeyError(f"Missing LMDB key: {k!r}")
                    img = decode_jpg_bytes(buf, image_size=None, to_float=True, fmt="CHW")
                    imgs.append(img)
                obs[img_key] = np.stack(imgs, axis=0)

        for key in self.lowdim_keys:
            obs[key] = self.lowdim[key][obs_indices].astype(np.float32)
        return obs

    def sampler(self, idx: int):
        cum = self.cum_lengths_active
        total = int(cum[-1])
        if idx < 0 or idx >= total:
            raise IndexError(f"idx {idx} out of range [0, {total})")

        episode_idx = int(np.searchsorted(cum, idx, side="right") - 1)
        episode_start = int(cum[episode_idx])
        ep_len = int(self.episode_lengths_active[episode_idx])
        local_idx = int(idx - episode_start)

        # obs indices: [-k..0] padded on the left
        temporal_indices = range(-(self.n_obs_steps - 1), 1)
        obs_indices = []
        for off in temporal_indices:
            j = local_idx + off
            if j >= 0:
                obs_indices.append(episode_start + j)

        pad = self.n_obs_steps - len(obs_indices)
        if pad > 0:
            pad_val = obs_indices[0] if obs_indices else episode_start
            obs_indices = [pad_val] * pad + obs_indices

        # action indices: [0..horizon-1] clamped to episode end
        action_indices = []
        for t in range(self.horizon):
            j = local_idx + t
            if j >= ep_len:
                j = ep_len - 1
            action_indices.append(episode_start + j)

        return episode_idx, obs_indices, action_indices

    def get_trajectory(self, episode_idx: int):
        """
        Get full trajectory of a given episode index (within active demos).
        """
        assert 0 <= episode_idx < self.n_demo_active, "Invalid episode index."

        episode_start = int(self.cum_lengths_active[episode_idx])
        ep_len = int(self.episode_lengths_active[episode_idx])
        obs_indices = list(range(episode_start, episode_start + ep_len))

        obs = self.get_obs(episode_idx, obs_indices)

        action = self.action[episode_start : episode_start + ep_len]

        trajectory = {
            "obs": obs,
            "action": action.astype(np.float32),
        }

        return trajectory

    # --------------------------------- Utilities -------------------------------- #
    def _action_to_xyz_6d(self, action):
        pos = action[:, :3]
        rot = action[:, 3:12]
        gripper = action[:, 12:]
        rot_6d = rot[:, :6]
        return np.concatenate([pos, rot_6d, gripper], axis=1)

    def _get_keys(self, shape_meta):
        rgb_keys, lowdim_keys = [], []
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            t = attr.get("type", "low_dim")
            if t == "rgb":
                rgb_keys.append(key)
            elif t == "low_dim":
                lowdim_keys.append(key)
        return rgb_keys, lowdim_keys

    def _load_numpy_array(self, rel_path: str):
        """
        Load a .npy relative to cache_dir.
        """
        path = os.path.join(self.cache_dir, rel_path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")
        return np.load(path, allow_pickle=True)

    def _load_metadata(self):
        meta_path = os.path.join(self.cache_dir, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Missing meta.json: {meta_path}")
        with open(meta_path, "r") as f:
            self.meta = json.load(f)

        self.episode_lengths_all = list(map(int, self.meta["episode_lengths"]))
        self.n_demo_all = len(self.episode_lengths_all)

    def _load_action(self):
        act_mode = self.action_mode
        if self.action_mode not in ("absolute", "relative", "delta"):
            raise ValueError(f"Unsupported action_mode: {self.action_mode}")

        fname = f"{act_mode}_action.npy"
        action = self._load_numpy_array(os.path.join("action", fname))
        self.action = action.astype(np.float32)

    def _load_lowdim(self):
        lowdim = {}
        for key in self.lowdim_keys:
            arr = self._load_numpy_array(os.path.join("lowdim", f"{key}.npy"))
            lowdim[key] = arr.astype(np.float32)
        self.lowdim = lowdim

    def _check_cache(self, dataset_path: str):
        if not os.path.exists(self.cache_dir):
            msg = (
                "[MimicGenDataset] Cache dir not found.\n"
                f"  dataset_path: {os.path.abspath(os.path.expanduser(dataset_path))}\n"
                f"  cache_dir   : {self.cache_dir}\n"
                "\nBuild the cache first, e.g.:\n"
                "  python decompress_hdf5_to_lmdb.py --in_dir <dataset_root> --cfg <your_hydra_cfg>\n"
            )
            warnings.warn(msg, category=UserWarning, stacklevel=2)
            raise FileNotFoundError(msg)

        must_have = ["images.lmdb", "meta.json"]
        missing = []
        for name in must_have:
            p = os.path.join(self.cache_dir, name)
            if not os.path.exists(p):
                missing.append(p)

        if not missing:
            return

        msg = (
            "[MimicGenDataset] Cache looks incomplete (missing core files).\n"
            f"  dataset_path: {os.path.abspath(os.path.expanduser(dataset_path))}\n"
            f"  cache_dir   : {os.path.abspath(self.cache_dir)}\n"
            "  missing:\n"
            + "\n".join([f"    - {p}" for p in missing])
            + "\n\nBuild the cache first, e.g.:\n"
            "  python decompress_hdf5_to_lmdb.py --in_dir <dataset_root> --cfg <your_hydra_cfg>\n"
        )
        warnings.warn(msg, category=UserWarning, stacklevel=2)
        raise FileNotFoundError(msg)
