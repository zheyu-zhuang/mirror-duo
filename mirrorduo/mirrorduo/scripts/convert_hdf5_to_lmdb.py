"""
Convert a robomimic/mirrorduo HDF5 dataset into an LMDB+NPY cache compatible with
your MimicGenDataset loader, with these semantics:

- shape_meta uses NON-delta defaults:
    robot0_eef_pos: (3,)
    robot0_eef_rot: (6,)      # rot6d in cache
    robot0_gripper_qpos: (2,)
  plus images

- HDF5 provides:
    obs/robot0_eef_rot: SO(3) rotation matrices (T,9) or (T,3,3)
  We convert to rot6d (T,6) using a RotationTransformer

- Save BOTH:
    lowdim/robot0_eef_pos.npy (base)
    lowdim/robot0_eef_delta_pos.npy (delta)
  and similarly for rot:
    lowdim/robot0_eef_rot.npy (base rot6d)
    lowdim/robot0_eef_delta_rot.npy (delta rot6d)

- Save ALL action modes:
    action/absolute_action.npy
    action/relative_action.npy
    action/delta_action.npy
  Each saved as (N,10) using your existing converter:
    xyz(3) + rot6d(6) + gripper(1)

- Save images to LMDB:
    images.lmdb with keys f"{img_key}/{global_step:08d}"

- Write meta.json with episode_lengths etc, plus build_done.flag
"""

import os
import json
import shutil
from pathlib import Path
from typing import Optional, List, Dict

import h5py
import lmdb
import numpy as np
import cv2
from tqdm import tqdm

from equi_diffpo.model.common.rotation_transformer import RotationTransformer
from mirrorduo.utils.core_utils import get_delta_positions, get_delta_rotations

# --------------------------
# Default SHAPE_META (rot=6)
# --------------------------
SHAPE_META = {
    "obs": {
        "agentview_image": {"shape": [3, 84, 84], "type": "rgb"},
        "robot0_eye_in_hand_image": {"shape": [3, 84, 84], "type": "rgb"},
        "robot0_eef_pos": {"shape": [3]},
        "robot0_eef_rot": {"shape": [6]},  # rot6d in cache
        "robot0_gripper_qpos": {"shape": [2]},
        # NOTE: deltas are NOT in shape_meta by default,
        # but we still save them as extra cache fields:
        # robot0_eef_delta_pos.npy, robot0_eef_delta_rot.npy
    },
    "action": {"shape": [10]},
}


# --------------------------
# Utilities
# --------------------------
def encode_rgb_to_jpg_bytes(img_rgb: np.ndarray, quality: int = 90) -> bytes:
    """
    Legacy convention: NO channel conversion before encoding.

    Expects:
        img_rgb: HWC uint8 (you treat it as RGB, but we keep bytes consistent with legacy)
    Returns:
        JPEG bytes
    """
    if img_rgb.dtype != np.uint8:
        img_rgb = img_rgb.astype(np.uint8)

    ok, buf = cv2.imencode(".jpg", img_rgb, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode(.jpg) failed")
    return buf.tobytes()


def _sorted_demo_keys(h5_data_group) -> List[str]:
    demos = [k for k in h5_data_group.keys() if k.startswith("demo_")]
    demos.sort(key=lambda x: int(x.split("_")[1]))
    return demos


# --------------------------
# Keep YOUR action converter
# --------------------------
def convert_action_to_xyz_6d_gripper(raw_actions, rotation_transformer):
    assert rotation_transformer is not None
    if raw_actions.shape[-1] == 14:
        raise NotImplementedError("Dual arm action conversion not implemented yet.")

    xyz = raw_actions[..., :3]  # (T,3)
    rot = raw_actions[..., 3:6]  # axis-angle (T,3)
    gripper = raw_actions[..., 6:]  # (T,1) typically

    rot_6d = rotation_transformer.forward(rot)  # (T,6)
    out = np.concatenate([xyz, rot_6d, gripper], axis=-1)
    return out.astype(np.float32)


# --------------------------
# Main cache builder
# --------------------------
def build_lmdb_cache_from_hdf5(
    hdf5_path: str,
    cache_dir: str,
    shape_meta: dict = SHAPE_META,
    n_demo: Optional[int] = None,  # None => all
    start_index: int = 0,
    image_size: Optional[int] = 84,  # resize images to this (or None to keep)
    jpeg_quality: int = 90,
    lmdb_map_size_gb: int = 32,
    commit_every: int = 5000,
    keep_failed: bool = True,
):
    hdf5_path = str(Path(hdf5_path).expanduser().resolve())
    cache_dir = str(Path(cache_dir).expanduser().resolve())

    # prepare output dir
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(os.path.join(cache_dir, "lowdim"), exist_ok=True)
    os.makedirs(os.path.join(cache_dir, "action"), exist_ok=True)

    # keys from shape_meta
    rgb_keys, lowdim_keys = [], []
    for k, attr in shape_meta["obs"].items():
        t = attr.get("type", "low_dim")
        if t == "rgb":
            rgb_keys.append(k)
        else:
            lowdim_keys.append(k)

    # sanity expected default lowdim
    for req in ("robot0_eef_pos", "robot0_eef_rot", "robot0_gripper_qpos"):
        if req not in lowdim_keys:
            raise ValueError(f"shape_meta missing required lowdim key: {req}")

    # rotation transformer:
    # - action transformer: axis_angle -> rot6d (matches your converter)
    action_rot_tf = RotationTransformer(from_rep="axis_angle", to_rep="rotation_6d")

    # open dataset, decide demos and episode lengths
    with h5py.File(hdf5_path, "r") as f:
        data = f["data"]
        demos = _sorted_demo_keys(data)
        demos = demos[int(start_index) :]
        if n_demo is not None:
            demos = demos[: int(n_demo)]
        if len(demos) == 0:
            raise RuntimeError("No demos selected (check start_index / n_demo).")

        kept_demo_keys = []
        episode_lengths = []

        for ep in demos:
            grp = data[ep]
            if "absolute_actions" not in grp:
                raise KeyError(f"{ep} missing absolute_actions")
            T = int(grp["absolute_actions"].shape[0])
            kept_demo_keys.append(ep)
            episode_lengths.append(T)

    if len(episode_lengths) == 0:
        raise RuntimeError("All demos filtered out (keep_failed=False and all failed?).")

    n_steps = int(sum(episode_lengths))

    # prealloc actions (save all modes)
    abs_all = np.empty((n_steps, 10), dtype=np.float32)
    rel_all = np.empty((n_steps, 10), dtype=np.float32)
    del_all = np.empty((n_steps, 10), dtype=np.float32)

    # prealloc lowdim (base + delta extras)
    # base rot is saved as rot6d (N,6)
    lowdim_out: Dict[str, np.ndarray] = {
        "robot0_eef_pos": np.empty((n_steps, 3), dtype=np.float32),
        "robot0_eef_rot": np.empty((n_steps, 6), dtype=np.float32),
        "robot0_gripper_qpos": np.empty((n_steps, 2), dtype=np.float32),
        "robot0_eef_delta_pos": np.empty((n_steps, 3), dtype=np.float32),  # extra
        "robot0_eef_delta_rot": np.empty((n_steps, 6), dtype=np.float32),  # extra
    }

    # LMDB init
    lmdb_path = os.path.join(cache_dir, "images.lmdb")
    env_lmdb = lmdb.open(
        lmdb_path,
        map_size=int(lmdb_map_size_gb * (1024**3)),
        subdir=False,
        readonly=False,
        meminit=False,
        map_async=True,
        max_dbs=1,
    )
    txn = env_lmdb.begin(write=True)
    put_count = 0

    # fill
    global_step = 0
    cursor = 0

    with h5py.File(hdf5_path, "r") as f:
        data = f["data"]

        pbar = tqdm(total=len(kept_demo_keys), desc="Build cache (LMDB + NPY)", unit="demo")
        for ep, T in zip(kept_demo_keys, episode_lengths):
            grp = data[ep]
            obs = grp["obs"]
            sl = slice(cursor, cursor + int(T))

            # --- actions (keep your converter) ---
            abs_act_raw = np.asarray(grp["absolute_actions"][:], dtype=np.float32)
            rel_act_raw = np.asarray(grp["relative_actions"][:], dtype=np.float32)
            del_act_raw = np.asarray(grp["delta_actions"][:], dtype=np.float32)

            abs_all[sl] = convert_action_to_xyz_6d_gripper(abs_act_raw, action_rot_tf)
            rel_all[sl] = convert_action_to_xyz_6d_gripper(rel_act_raw, action_rot_tf)
            del_all[sl] = convert_action_to_xyz_6d_gripper(del_act_raw, action_rot_tf)

            # --- lowdim ---
            eef_pos = np.asarray(obs["robot0_eef_pos"][:], dtype=np.float32)  # (T,3)
            eef_rot = np.asarray(obs["robot0_eef_rot"][:], dtype=np.float32)  # (T,3,3)
            g_qpos = np.asarray(obs["robot0_gripper_qpos"][:], dtype=np.float32)  # (T,2)

            # base
            lowdim_out["robot0_eef_pos"][sl] = eef_pos
            # not that, this 6d is the first two rows of the rotation matrix, rather than the two columns
            # as rotation_transformer uses row-convention for rot6d
            lowdim_out["robot0_eef_rot"][sl] = eef_rot.reshape(int(T), -1)[:, :6]  # to rot6d
            lowdim_out["robot0_gripper_qpos"][sl] = g_qpos

            # deltas (use your existing utils)
            lowdim_out["robot0_eef_delta_pos"][sl] = get_delta_positions(eef_pos)
            lowdim_out["robot0_eef_delta_rot"][sl] = get_delta_rotations(eef_rot, return_6d=True)
            # ---
            imgs_by_cam = {}
            for img_key in rgb_keys:
                imgs = obs[img_key][:]  # (T,H,W,3)
                imgs_by_cam[img_key] = np.asarray(imgs, dtype=np.uint8)

            for t in range(int(T)):
                gidx = global_step
                for img_key in rgb_keys:
                    img = imgs_by_cam[img_key][t]  # HWC uint8

                    if image_size is not None and (
                        img.shape[0] != int(image_size) or img.shape[1] != int(image_size)
                    ):
                        img = cv2.resize(
                            img, (int(image_size), int(image_size)), interpolation=cv2.INTER_AREA
                        )

                    jpg = encode_rgb_to_jpg_bytes(
                        img, quality=jpeg_quality
                    )  # <-- your legacy function
                    k = f"{img_key}/{gidx:08d}".encode("ascii")
                    txn.put(k, jpg)
                    put_count += 1

                    if put_count % int(commit_every) == 0:
                        txn.commit()
                        txn = env_lmdb.begin(write=True)

                global_step += 1

            cursor += int(T)
            pbar.update(1)

        pbar.close()

    # finalize LMDB
    txn.put(b"__len__", str(global_step).encode("ascii"))
    txn.commit()
    env_lmdb.sync()
    env_lmdb.close()

    # save npys
    np.save(os.path.join(cache_dir, "action", "absolute_action.npy"), abs_all)
    np.save(os.path.join(cache_dir, "action", "relative_action.npy"), rel_all)
    np.save(os.path.join(cache_dir, "action", "delta_action.npy"), del_all)

    for k, arr in lowdim_out.items():
        np.save(os.path.join(cache_dir, "lowdim", f"{k}.npy"), arr)

    meta = {
        "source_hdf5": hdf5_path,
        "rgb_keys": rgb_keys,
        # loader will only auto-load keys that exist in *its* shape_meta;
        # but we include extras here for completeness.
        "lowdim_keys": sorted(list(lowdim_out.keys())),
        "episode_lengths": list(map(int, episode_lengths)),
        "n_demo": int(len(episode_lengths)),
        "n_samples": int(n_steps),
        "image_size": int(image_size) if image_size is not None else None,
        "jpeg_quality": int(jpeg_quality),
        "action_dim": 10,
        "kept_demos": kept_demo_keys,
    }
    with open(os.path.join(cache_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    with open(os.path.join(cache_dir, "build_done.flag"), "w") as f:
        f.write("build completed\n")

    print(f"[built] cache_dir={cache_dir} demos={meta['n_demo']} samples={meta['n_samples']}")


if __name__ == "__main__":
    import argparse

    def _default_hdf5_path(task_name: str) -> str:
        # data/{task}/{task}.hdf5
        task = str(task_name)
        return str(Path("data") / task / f"{task}_mirrorduo.hdf5")

    def _default_cache_dir(task_name: str) -> str:
        # data/{task}/{task}_lmdb
        task = str(task_name)
        return str(Path("data") / task / f"{task}_lmdb")

    ap = argparse.ArgumentParser()
    ap.add_argument("--task_name", type=str, required=True)
    args = ap.parse_args()

    hdf5_file_path = _default_hdf5_path(args.task_name)
    cache_dir = _default_cache_dir(args.task_name)

    if not os.path.exists(hdf5_file_path):
        raise FileNotFoundError(f"Input HDF5 not found: {hdf5_file_path}")

    build_lmdb_cache_from_hdf5(
        hdf5_path=hdf5_file_path,
        cache_dir=cache_dir,
        shape_meta=SHAPE_META,
        n_demo=None,
        start_index=0,
        image_size=84,
        jpeg_quality=90,
        lmdb_map_size_gb=32,
        commit_every=5000,
    )
