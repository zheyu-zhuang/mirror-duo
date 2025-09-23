# MirrorDuo  
[![Preprint PDF](https://img.shields.io/badge/Paper-PDF-red)](asset/preprint.pdf)
[![Poster PDF](https://img.shields.io/badge/Poster-PDF-blue)](asset/poster.pdf) 

**MirrorDuo** captures a simple idea: **moving left implies knowing how to move right**. We turn this intuition into a framework for image-based imitation learning in the SE(3) state–action space. By systematically mirroring images, robot states, and actions, MirrorDuo effectively doubles the training data without additional collection, improving data efficiency and policy robustness.
<p align="center">
  <img src="asset/highlight.gif" width="50%">
</p>

## Key Features  

- **Workspace Generalization**  
  One-sided demonstrations can be mirrored to cover the opposite side, effectively expanding the workspace.  

- **Data Efficiency**  
  For both-sided demonstrations, MirrorDuo boosts efficiency — fewer demonstrations are needed to span the same workspace.  

- **Unified Mirroring Operator**  
  Consistently applies mirroring to images, proprioceptive states, and actions.  

- **Two Variants**  
  - *MirrorAug*: A plug-and-play data augmentation module that directly operates on input demonstrations.  
  - *MirrorDiffusion*: A diffusion-based policy learning approach with reflection symmetry built in as an inductive bias.  

- **Plug-and-Play Integration**  
  Works seamlessly with [robomimic](https://github.com/ARISE-Initiative/robomimic) and [robosuite](https://github.com/ARISE-Initiative/robosuite).  

This repository accompanies the paper, accepted by Conference on Robot Learning (CoRL) 2025  

> **MirrorDuo: Reflection-Consistent Visuomotor Learning from Mirrored Demonstration Pairs**  
> Zheyu Zhuang\*¹, Ruiyu Wang\*¹, Giovanni Luca Marchetti², Florian T. Pokorny¹, Danica Kragic¹  
> ¹ Division of Robotics, Perception and Learning, KTH Royal Institute of Technology, Stockholm, Sweden  
> ² Department of Mathematics, KTH Royal Institute of Technology, Stockholm, Sweden  
> \* Equal contribution  

## Installation

1.  Install the following apt packages for mujoco:
    ```bash
    sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
    ```
2. Install gfortran (dependancy for escnn) 
    ```bash
    sudo apt install -y gfortran
    ```

3. This repository uses Git submodules to include external dependencies (e.g., `equidiff`, `robomimic`, `robosuite`).  
To ensure compatibility, submodules are pinned to specific branches.
    ```bash
    git clone --recurse-submodules https://github.com/zheyu-zhuang/mirror-duo.git
    git submodule update --init --recursive
    cd mirror-duo
    ```
4. Install environment: Use Mambaforge (strongly recommended):
    ```bash
    mamba env create -f conda_environment.yaml
    conda activate mirror-duo
    ``` 

5. Apply patches. Modifications of the external libraries, e.g. robotsuite, are kept as patches. 
    ```bash
    chmod +x patch_manager.sh
    ./patch_manager.sh apply
    ```
    `Note`: The script is interactive by default and will prompt you to re-enter apply as a confirmation.
## Dataset

### 1. Download Dataset
Download datasets from [MimicGen on Hugging Face](https://huggingface.co/datasets/amandlek/mimicgen_datasets/tree/main/core).  

Place them under:
```
/path/to/mirror-duo/data/<dataset>/<dataset>.hdf5
```

---

### 2. Convert Action Space
The downloaded dataset uses a **relative pose** representation:  
- **Rotation**: expressed in the **world frame** as $R_t R_{t-1}^T$  
- **Translation**: expressed as the **difference in positions** $T_t - T_{t-1}$  

This differs from the **true relative action representation**, where the current pose is expressed in the **previous frame**.  

This step converts dataset actions into the required **true relative action space**, as well as other action modes compatible with MirrorDuo.  

```bash
python mirrorduo/mirrorduo/scripts/convert_dataset_actions.py \
  -i data/<dataset>/<dataset>.hdf5 \
  -o data/<dataset>/<dataset>_mirrorduo.hdf5 \
  -b <num_envs, e.g. 50> \
  -n <num_trajectories, e.g. 300>
```


#### 3. Prepare Random Backgrounds
Add random images for **Random Overlay**:
```bash
unzip backgrounds.zip -d data/
```

### 4. [Optional] Re-render Trajectories under a Mirrored Setup

This step **re-renders** the dataset in a mirrored workspace. Mirroring is applied directly during demonstration rendering to produce **actual mirrored demonstrations**.  
These mirrored demonstrations can later be combined with the original (non-mirrored) data to evaluate data efficiency on one-sided tasks.

```bash
# Template
python mirrorduo/mirrorduo/scripts/rerender_demos.py \
  -i data/<dataset>/<dataset>_mirrorduo.hdf5 \
  -o data/<dataset>/<dataset>_mirrorduo_with_50_mirrored_demos.hdf5 \
  --env_name <EnvName>_Mirror \
  -n <num_trajectories>

# Example
python mirrorduo/mirrorduo/scripts/rerender_demos.py \
  -i data/coffee_d1/coffee_d1_mirrorduo.hdf5 \
  -o data/coffee_d1/coffee_d1_mirrorduo_with_50_mirrored_demos.hdf5 \
  --env_name Square_D1_Mirror \
  -n 50
```

**Notes**
- The output HDF5 filename is arbitrary; examples above are illustrative.  
- `-n` controls how many trajectories to (re)render in the mirrored setup.  
- `--env_name` must match a supported mirrored environment (see list below).  
- Environments containing `Mirror` treat the input dataset as the original and re-render the mirrored trajectory automatically.  

**Currently supported `--env_name` values**  
(in addition to the original MimicGen environments):
- Coffee_D0_Mirror  
- Coffee_D2_Mirror  
- Coffee_D2_Agentview_Far  
- Coffee_D2_Agentview_Far_Mirror  
- Square_D1_Mirror  
- Square_D0_Agentview_Far  
- Square_D0_Agentview_Far_Mirror  

> `Agentview_Far` = demonstrations rendered with the agent-view camera positioned farther back.  

### 5. [Optional] Combine Mirrored Demos with the Original Demos
You can combine original demonstrations with mirrored ones for training.  
For example, if you have **200 original demos** and **50 mirrored demos**, you can later specify a total of **210 demos** during training (meaning 200 original + 10 mirrored).
```bash
# Example
python mirrorduo/mirrorduo/scripts/combine_datasets.py \
  --file_a data/square_d0/square_d0_mirrorduo.hdf5 \
  --file_b data/square_d0/square_d0_mirrored_setup.hdf5 \
  --num_a 200 \
  --num_b 40 \
  --output data/square_d0/square_d0_mirrorduo_200_plus_40.hdf5
```

## Train

#### [MirrorAug] Train with mirror-based data augmentation
```bash
# Example with dataset `square_d2`:
python train.py --config-name=train_diffusion_unet.yaml \
  exp_name=<your_exp_name> \
  task_name=square_d2 \
  n_envs=<num_parallel_envs> \
  action_mode=<delta|relative> \
  n_demo=<e.g.,200> \
  enable_overlay=True \
  enable_mirror=True \
  pretrained=True
```

**Tips**
- Best results in the paper used **random background overlay** (`enable_overlay=True`) + **pretrained visual encoders** (`pretrained=True`).  
- Large `n_envs` (e.g., 50) may require ~64 GB RAM.  

#### [MirrorDiffusion] Train with reflection-equivariant policy
```bash
# Example with dataset `square_d2`:
python train.py --config-name=train_mirrorduo_diffusion_unet.yaml \
  exp_name=<your_exp_name> \
  task_name=square_d2 \
  n_envs=<num_parallel_envs> \
  action_mode=delta \
  n_demo=200 \
  enable_overlay=True
```

---
    
**Mixing Mirrored Demos**  
If you augment your dataset by adding **x mirrored demonstrations** on top of **N original demos** (e.g., N=200, x=50), but only want to include **10 mirrored demos** during training, set:
```
n_demo = N + 10    # e.g., n_demo=210
```

**Dataset Path**  
- The default dataset path is:
  ```
  ./data/<dataset>/<dataset>_mirrorduo.hdf5
  ```
- If you have a **custom dataset name** or a **combined dataset** (e.g., `*_200_plus_40.hdf5`), update the dataset path in the corresponding config file.  
  Example config file:
  ```
  ./mirrorduo/mirrorduo/scripts/train_diffusion_unet.yaml
  ```

## Evaluate

To evaluate a trained policy, run:

```bash
python eval.py \
  --checkpoint <path_to_checkpoint_or_dir> \
  --env_name <EnvName> \
  --n_envs <num_parallel_envs> \
  --n_eval_rollouts <num_rollouts> \
  --device <cpu|cuda> \
  --seed <random_seed> \
  --output_dir <save_results_dir>
```

**Example**
```bash
python eval.py \
  --checkpoint experiments/mirrorduo/square_d0_seed_0/.../epoch=0070.ckpt \
  --env_name Square_D0_Mirror \
  --n_envs 25 \
  --n_eval_rollouts 50 \
  --device cuda \
  --output_dir results/square_d0_eval \
  --seed 42
```

**Testing in Mirrored Setups**  
To evaluate performance in a mirrored workspace, simply change `--env_name` to one of the mirrored environments listed earlier (e.g., `Square_D1_Mirror`).


## Acknowledgements  

We thank the open-source community. This repository:  

- Builds upon the original [Equivariant Diffusion Policy](https://github.com/pointW/equidiff)  
- Depends heavily on:  
  - [robosuite](https://github.com/ARISE-Initiative/robosuite)  
  - [robomimic](https://github.com/ARISE-Initiative/robomimic)  
  - [mimicgen](https://github.com/NVlabs/mimicgen)  

## License

This project is licensed under the [MIT License](LICENSE).

## Citation  

```bibtex
@inproceedings{mirrorduo2025,
  title     = {MirrorDuo: Reflection-Consistent Visuomotor Learning from Mirrored Demonstration Pairs},
  author    = {Zhuang, Zheyu and Wang, Ruiyu and Marchetti, Giovanni Luca and Pokorny, Florian T. and Kragic, Danica},
  booktitle = {9th Conference on Robot Learning (CoRL)},
  year      = {2025},
}
