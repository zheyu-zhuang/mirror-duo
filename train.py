"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

from equi_diffpo.workspace.base_workspace import BaseWorkspace
import pathlib
from omegaconf import OmegaConf
import hydra
import sys

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)


max_steps = {
    "square_d0": 400,
    "stack_d1": 400,
    "stack_three_d1": 400,
    "square_d2": 400,
    "threading_d2": 400,
    "coffee_d2": 400,
    "three_piece_assembly_d2": 500,
    "hammer_cleanup_d1": 500,
    "mug_cleanup_d1": 500,
    "kitchen_d1": 800,
    "nut_assembly_d0": 500,
    "pick_place_d0": 1000,
    "coffee_preparation_d1": 800,
    "tool_hang": 700,
    "can": 400,
    "lift": 400,
    "square": 400,
    "simple_pnp": 400,  # dummy for real robot
    "simple_pnp_with_5_mirror": 400,  # dummy for real robot
    "stack_two": 400,  # dummy for real robot
    "stack_green_on_blue": 400,  # dummy for real robot
    }

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("get_max_steps", lambda x: max_steps[x], replace=True)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("mirrorduo", "mirrorduo", "config")),
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
