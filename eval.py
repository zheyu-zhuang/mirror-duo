import json
import os
import pathlib
import pprint
import random
import sys

import click
import dill
import hydra
import numpy as np
import torch
import wandb
from omegaconf.omegaconf import open_dict

from mirrorduo.utils.junk_utils import divider, plain_divider
from mirrorduo.workspace.base_workspace import BaseWorkspace

# Use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)


def evaluate_checkpoint(checkpoint, output_dir, device, n_eval_rollouts, seed, env_name, n_envs):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    print(f"Action mode: {cfg.action_mode}")
    policy = workspace.model
    if getattr(cfg.training, "use_ema", False):
        policy = workspace.ema_model

    device = torch.device(device)
    policy.to(device)
    policy.eval()

    with open_dict(cfg):
        cfg.task.env_runner.n_train = 0
        cfg.task.env_runner.n_test = n_eval_rollouts
        cfg.task.env_runner.n_envs = n_envs
        cfg.task.env_runner.n_test_vis = 10
        if env_name is not None:
            cfg.task.env_runner.env_name = env_name

    env_runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=output_dir)
    runner_log = env_runner.run(policy)

    json_log = {}
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value

    detailed_log = {
        "Mean Score": json_log.get("test/mean_score", "Not found"),
        "Test Environment": cfg.task.env_runner.env_name if env_name is None else env_name,
        "Number of Rollouts": n_eval_rollouts,
        "Random Seeds": seed,
        "Paths": {
            "Experiment Folder": os.path.dirname(checkpoint),
            "Checkpoint File": os.path.basename(checkpoint),
            "Output Directory": output_dir,
        },
    }

    env_name = cfg.task.env_runner.env_name if env_name is None else env_name
    divider(env_name, char="=", show=True)
    pprint.pprint(detailed_log, indent=2)
    plain_divider(char="=", show=True)

    # Save logs
    with open(os.path.join(output_dir, "eval_log.json"), "w") as f:
        json.dump(detailed_log, f, indent=2, sort_keys=True)

    with open(os.path.join(output_dir, "raw_runner_log.json"), "w") as f:
        json.dump(json_log, f, indent=2, sort_keys=True)


@click.command()
@click.option("-d", "--device", default="cuda:0")
@click.option("-e", "--env_name", default="None")
@click.option("-c", "--checkpoint", required=True, help="Checkpoint path or directory")
@click.option("-n", "--n_eval_rollouts", default=50)
@click.option("-o", "--output_dir", default=None)
@click.option("-s", "--seed", default=100000, type=int)
@click.option("--n_envs", default=50, type=int)
def main(env_name, checkpoint, output_dir, device, n_eval_rollouts, seed, n_envs):
    env_name = None if env_name == "None" else env_name

    if os.path.isfile(checkpoint):
        ckpt_list = [checkpoint]
    elif os.path.isdir(checkpoint):
        ckpt_list = sorted(
            [os.path.join(checkpoint, f) for f in os.listdir(checkpoint) if f.endswith(".ckpt")]
        )
    else:
        raise ValueError(f"Invalid checkpoint path: {checkpoint}")

    print(f"Found {len(ckpt_list)} checkpoints to evaluate.")

    for ckpt_path in ckpt_list:
        ckpt_dir = os.path.dirname(ckpt_path)
        ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]
        # Default output_dir to ../eval/<ckpt_name> relative to the checkpoint folder
        ckpt_output_dir = output_dir
        if ckpt_output_dir is None:
            ckpt_output_dir = os.path.join(os.path.dirname(ckpt_dir), "eval", ckpt_name)
        pathlib.Path(ckpt_output_dir).mkdir(parents=True, exist_ok=True)
        print(f"\nEvaluating checkpoint: {ckpt_path}")
        evaluate_checkpoint(
            checkpoint=ckpt_path,
            output_dir=ckpt_output_dir,
            device=device,
            n_eval_rollouts=n_eval_rollouts,
            seed=seed,
            env_name=env_name,
            n_envs=n_envs,
        )


if __name__ == "__main__":
    main()
