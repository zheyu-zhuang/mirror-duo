import concurrent.futures
import copy
import multiprocessing
import os
import shutil
from typing import Dict, List

import h5py
import numpy as np
import torch
import zarr
from equi_diffpo.codecs.imagecodecs_numcodecs import Jpeg2k, register_codecs
from equi_diffpo.common.normalize_util import (
    array_to_stats,
    get_identity_normalizer_from_stat,
    get_image_range_normalizer,
    get_range_normalizer_from_stat,
)
from equi_diffpo.common.pytorch_util import dict_apply
from equi_diffpo.common.replay_buffer import ReplayBuffer
from equi_diffpo.common.sampler import SequenceSampler, get_val_mask
from equi_diffpo.dataset.base_dataset import BaseImageDataset, LinearNormalizer
from equi_diffpo.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from equi_diffpo.model.common.rotation_transformer import RotationTransformer
from filelock import FileLock
from omegaconf import OmegaConf
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from mirrorduo.utils.core_utils import (
    get_action_key,
)
from mirrorduo.utils.junk_utils import is_matched_key
from mirrorduo.utils.normalize_utils import (
    mirrorduo_action_normalizer_from_stat,
    mirrorduo_pos_normalizer_from_stat,
)

register_codecs()


class RealReplayMirrorDuoImageDataset(BaseImageDataset):
    def __init__(
        self,
        shape_meta: dict,
        dataset_path: str,
        action_mode: str,
        enable_mirror: bool,
        horizon=1,
        pad_before=0,
        pad_after=0,
        n_obs_steps=None,
        rotation_rep="rotation_6d",
        use_legacy_normalizer=False,
        use_cache=False,
        seed=42,
        val_ratio=0.0,
        n_demo=100,
    ):
        self.n_demo = n_demo
        self.enable_mirror = enable_mirror
        # return all supported action keys for zarr, e.g. "absolute_action", "relative_action"
        # Note that plural action keys, i.e. "absolute_actions", are defaulted in the robomimic dataset
        self.all_action_keys = get_action_key("all", original_key="action")

        # which action key to use for __getitem__
        self.action_item_key = get_action_key(action_mode, original_key="action")
        replay_buffer = None
        if use_cache:
            cache_zarr_path = dataset_path + f".{n_demo}" + ".zarr"
            cache_lock_path = cache_zarr_path + ".lock"
            print("Acquiring lock on cache.")
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    # cache does not exists
                    try:
                        print("Cache does not exist. Creating!")
                        # store = zarr.DirectoryStore(cache_zarr_path)
                        replay_buffer = self.convert_to_replay(
                            store=zarr.MemoryStore(),
                            shape_meta=shape_meta,
                            dataset_path=dataset_path,
                            n_demo=n_demo,
                        )
                        print("Saving cache to disk.")
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(store=zip_store)
                    except Exception as e:
                        shutil.rmtree(cache_zarr_path)
                        raise e
                else:
                    print("Loading cached ReplayBuffer from Disk.")
                    with zarr.ZipStore(cache_zarr_path, mode="r") as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore()
                        )
                    print("Loaded!")
        else:
            replay_buffer = self.convert_to_replay(
                store=zarr.MemoryStore(),
                shape_meta=shape_meta,
                dataset_path=dataset_path,
                n_demo=n_demo,
            )

        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            type = attr.get("type", "low_dim")
            if type == "rgb":
                rgb_keys.append(key)
            elif type == "low_dim":
                lowdim_keys.append(key)

        # for key in rgb_keys:
        #     replay_buffer[key].compressor.numthreads=1

        key_first_k = dict()
        if n_obs_steps is not None:
            # only take first k obs from images
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed
        )
        train_mask = ~val_mask
        sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k,
        )

        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.n_obs_steps = n_obs_steps
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.use_legacy_normalizer = use_legacy_normalizer

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # action
        stat = array_to_stats(self.replay_buffer[self.action_item_key])
        if stat["mean"].shape[-1] > 10:
            raise NotImplementedError("Functions for dual-arm not yet implemented.")
        # Note that stats loads based on the action key, but "action" is hardcoded, for compatibility
        normalizer["action"] = mirrorduo_action_normalizer_from_stat(stat, self.enable_mirror)

        # obs
        for key in self.lowdim_keys:
            stat = array_to_stats(self.replay_buffer[key])
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

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        # to save RAM, only return first n_obs_steps of OBS
        # since the rest will be discarded anyway.
        # when self.n_obs_steps is None
        # this slice does nothing (takes all)
        T_slice = slice(self.n_obs_steps)

        obs_dict = dict()
        for key in self.rgb_keys:
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_dict[key] = np.moveaxis(data[key][T_slice], -1, 1).astype(np.float32) / 255.0
            # T,C,H,W
            del data[key]
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            del data[key]

        torch_data = {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "action": torch.from_numpy(data[self.action_item_key].astype(np.float32)),
        }
        return torch_data

    def convert_to_replay(
        self,
        store,
        shape_meta,
        dataset_path,
        n_workers=None,
        max_inflight_tasks=None,
        n_demo=100,
    ):
        if n_workers is None:
            n_workers = multiprocessing.cpu_count()
        if max_inflight_tasks is None:
            max_inflight_tasks = n_workers * 5

        # parse shape_meta
        rgb_keys = list()
        lowdim_keys = list()
        # construct compressors and chunks
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            shape = attr["shape"]
            type = attr.get("type", "low_dim")
            if type == "rgb":
                rgb_keys.append(key)
            elif type == "low_dim":
                lowdim_keys.append(key)

        root = zarr.group(store)
        data_group = root.require_group("data", overwrite=True)
        meta_group = root.require_group("meta", overwrite=True)

        def create_array(data_group, name, data):
            return data_group.array(
                name=name,
                data=data,
                shape=data.shape,
                chunks=data.shape,
                compressor=None,
                dtype=data.dtype,
            )

        with h5py.File(dataset_path) as file:
            # count total steps
            demos = file["data"]
            episode_ends = list()
            prev_end = 0
            for i in range(n_demo):
                demo = demos[f"demo_{i}"]
                episode_length = demo["actions"].shape[0]
                episode_end = prev_end + episode_length
                prev_end = episode_end
                episode_ends.append(episode_end)
            n_steps = episode_ends[-1]
            episode_starts = [0] + episode_ends[:-1]
            _ = meta_group.array(
                "episode_ends", episode_ends, dtype=np.int64, compressor=None, overwrite=True
            )

            # save lowdim data
            for key in tqdm(lowdim_keys + self.all_action_keys, desc="Loading lowdim data"):
                # first fetch the original key and data, then convert to the new key
                target_key = key
                data_key = "obs/" + key
                if key in self.all_action_keys:
                    data_key = key.replace("action", "actions")  # robomimic uses plural
                this_data = list()

                if data_key not in demos["demo_0"]:
                    this_data = None
                else:
                    for i in range(n_demo):
                        demo = demos[f"demo_{i}"]
                        this_data.append(demo[data_key][:].astype(np.float32))
                    this_data = np.concatenate(this_data, axis=0)

                    if key in self.all_action_keys:
                        assert this_data.shape == (n_steps,) + tuple(shape_meta["action"]["shape"])
                    else:
                        assert this_data.shape == (n_steps,) + tuple(shape_meta["obs"][target_key]["shape"])

                    create_array(data_group, key, this_data)

            def img_copy(zarr_arr, zarr_idx, hdf5_arr, hdf5_idx):
                try:
                    zarr_arr[zarr_idx] = hdf5_arr[hdf5_idx]
                    # make sure we can successfully decode
                    _ = zarr_arr[zarr_idx]
                    return True
                except Exception as e:
                    return False

            with tqdm(
                total=n_steps * len(rgb_keys), desc="Loading image data", mininterval=1.0
            ) as pbar:
                # one chunk per thread, therefore no synchronization needed
                with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
                    futures = set()
                    for key in rgb_keys:
                        data_key = "obs/" + key
                        shape = tuple(shape_meta["obs"][key]["shape"])
                        c, h, w = shape
                        this_compressor = Jpeg2k(level=50)
                        img_arr = data_group.require_dataset(
                            name=key,
                            shape=(n_steps, h, w, c),
                            chunks=(1, h, w, c),
                            compressor=this_compressor,
                            dtype=np.uint8,
                        )
                        for episode_idx in range(n_demo):
                            demo = demos[f"demo_{episode_idx}"]
                            hdf5_arr = demo["obs"][key]
                            for hdf5_idx in range(hdf5_arr.shape[0]):
                                if len(futures) >= max_inflight_tasks:
                                    # limit number of inflight tasks
                                    completed, futures = concurrent.futures.wait(
                                        futures, return_when=concurrent.futures.FIRST_COMPLETED
                                    )
                                    for f in completed:
                                        if not f.result():
                                            raise RuntimeError("Failed to encode image!")
                                    pbar.update(len(completed))

                                zarr_idx = episode_starts[episode_idx] + hdf5_idx
                                futures.add(
                                    executor.submit(
                                        img_copy, img_arr, zarr_idx, hdf5_arr, hdf5_idx
                                    )
                                )
                    completed, futures = concurrent.futures.wait(futures)
                    for f in completed:
                        if not f.result():
                            raise RuntimeError("Failed to encode image!")
                    pbar.update(len(completed))

        replay_buffer = ReplayBuffer(root)
        return replay_buffer


def normalizer_from_stat(stat):
    max_abs = np.maximum(stat["max"].max(), np.abs(stat["min"]).max())
    scale = np.full_like(stat["max"], fill_value=1 / max_abs)
    offset = np.zeros_like(stat["max"])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale, offset=offset, input_stats_dict=stat
    )
