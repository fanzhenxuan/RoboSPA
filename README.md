<h1 align="center">RoboSPA</h1>
<h3 align="center">Can VLA Models Go Beyond Simple Scenes and Short-Horizon Tasks? [EMNLP 2026]🔥🔥🔥</h3>

<p align="center">
  🌐 <a href="https://fanzhenxuan.github.io/RoboSPA/">Project Page</a> ·
  📄 <a href="https://arxiv.org/abs/2609.05324">Paper</a> ·
  💻 <a href="https://github.com/fanzhenxuan/RoboSPA">Code</a> ·
  🤗 <a href="https://huggingface.co/datasets/zxfan/RoboSPA">Dataset</a>
</p>

<p align="center">
  <b>
  <a href="https://fanzhenxuan.github.io/">Zhenxuan Fan</a><sup>1</sup>,
  Bo Zhang<sup>2</sup>,
  Yutong Lin<sup>1</sup>,
  Yuqian Yuan<sup>1</sup>,
  Juekai Lin<sup>1</sup>,
  Liang Liang<sup>1</sup>,
  Zhuoyi Huang<sup>3</sup>,
  Wenqiao Zhang<sup>1,*</sup>,
  Juncheng Li<sup>1,*</sup>,
  Siliang Tang<sup>1</sup>,
  Jun Xiao<sup>1</sup>,
  Yueting Zhuang<sup>1</sup>
  </b>
</p>

<p align="center">
  <sub>
  <sup>1</sup> Zhejiang University &nbsp;&nbsp;
  <sup>2</sup> University of Electronic Science and Technology of China &nbsp;&nbsp;
  <sup>3</sup> South China Normal University
  </sub>
</p>

<p align="center">
  <sub>* Corresponding authors</sub>
</p>

---

**RoboSPA** (**Robo**t **S**patial-**P**rocedural **A**ssessment) is a large-scale robotic manipulation dataset and benchmark for diagnosing embodied reasoning in Vision-Language-Action (VLA) models. It is built on the [RoboTwin 2.0](https://robotwin-platform.github.io/) simulation and data-generation framework (SAPIEN + CuRobo).

RoboSPA focuses on two capability dimensions:

- 🔍 **Fine-Grained Spatial Reasoning** — identify the intended target among visually similar candidates
- 🧭 **Long-Horizon Procedural Planning** — complete multi-step procedures under ordering, repetition, and memory constraints

| Item | Scale |
| --- | --- |
| Task categories | 10 |
| Base tasks | 56 |
| Difficulty levels | 5 (L1–L5) |
| Task variants | 280 |
| Trajectories | 527K |
| Interaction video | 997 hours |
| Timesteps | 108M |
| Embodiments | Aloha-AgileX, ARX-X5, Piper, Franka, UR5 |

---

## 📑 Table of Contents

- [📚 Overview](#-overview)
- [🐣 News](#-news)
- [🛠️ Installation](#️-installation)
- [🚀 Quick Start](#-quick-start)
- [⚙️ Task Configurations](#️-task-configurations)
- [📦 Data Collection](#-data-collection)
- [💽 Released Dataset](#-released-dataset)
- [🤖 Other Policy Baselines](#-other-policy-baselines)
- [🧩 Deploy Your Own Policy](#-deploy-your-own-policy)
- [📋 Task List](#-task-list)
- [👍 Citation](#-citation)
- [📬 Contact](#-contact)
- [😺 Acknowledgement](#-acknowledgement)
- [🏷️ License](#️-license)

---

## 📚 Overview

Existing VLA benchmarks mostly score binary task completion in clean, short-horizon settings. RoboSPA instead scales **spatial ambiguity** and **procedural horizon** in a controlled way, while keeping task semantics fixed.

Each of the 56 base tasks has five variants (`_1` … `_5`):

- Spatial tasks add more plausible candidates
- Procedural tasks add more subgoals / longer sequences

Beyond Success Rate (SR), RoboSPA reports:

- **ONTA** (Object-Normalized Target Accuracy) for spatial tasks — chance-corrected target selection
- **Progress Score (PS)** for procedural tasks — fraction of completed subgoals

The paper evaluates RDT, GO-1, π0.5, and X-VLA in a **single-task** setting on clean-scene Aloha-AgileX data. At L5, every evaluated model stays below 25% overall success.

This repository reuses RoboTwin 2.0's expert-script data pipeline: task environments live in `envs/`, collection is driven by `collect_data.sh`, and policy evaluation goes through `script/eval_policy.py`.

---

## 🐣 News

- 📄 **2026/09/04** — Paper on arXiv: [arXiv:2609.05324](https://arxiv.org/abs/2609.05324)
- 🎉 **2026/08** — Accepted at EMNLP 2026 (Main Conference)

---

## 🛠️ Installation

Please follow the [RoboTwin 2.0 installation guide](https://robotwin-platform.github.io/doc/usage/robotwin-install.html) to set up the environment and download assets.

---

## 🚀 Quick Start

Train and evaluate **π0.5** on `Pick_Blocks_Size_1` (`demo_clean`). Put training HDF5 under `data/` from the [released dataset](https://huggingface.co/datasets/zxfan/RoboSPA). π0.5 env setup (`uv`, CuRobo): [RoboTwin π0.5 guide](https://robotwin-platform.github.io/doc/usage/Pi05.html).

```bash
conda activate RoboTwin
cd policy/pi05
mkdir -p processed_data training_data
```

### 1. Train

```bash
bash process_data_pi05.sh Pick_Blocks_Size_1 demo_clean 50
mkdir -p training_data/demo_clean
cp -r processed_data/Pick_Blocks_Size_1-demo_clean-50 training_data/demo_clean/
bash generate.sh ./training_data/demo_clean/ demo_clean_repo
```

Set `repo_id="demo_clean_repo"` on `pi05_aloha_full_base` in `src/openpi/training/config.py`, then:

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_aloha_full_base
bash finetune.sh pi05_aloha_full_base demo_clean 0
#                train_config          model_name  gpu
```

Checkpoints: `policy/pi05/checkpoints/pi05_aloha_full_base/demo_clean/`. Default eval `checkpoint_id` is `30000` in `deploy_policy.yml`.

### 2. Evaluate

```bash
bash eval.sh Pick_Blocks_Size_1 demo_clean pi05_aloha_full_base demo_clean 0 0
#            task_name          eval_cfg  train_config         model_name seed gpu
```

100 rollouts, unseen instructions. Results go to `eval_result/`.

Train clean, test randomized:

```bash
bash eval.sh Pick_Blocks_Size_1 demo_randomized pi05_aloha_full_base demo_clean 0 0
```

---

## ⚙️ Task Configurations

YAML files live in `task_config/`. Create a new one with:

```bash
bash task_config/create_task_config.sh my_config
```

Shipped configs:

| File | Typical use |
| --- | --- |
| `demo_clean.yml` | Clean table, no clutter / lighting randomization. Paper training & eval setting. |
| `demo_randomized.yml` | Random background, clutter, lighting, table height. Robustness split. |

Important fields (defaults match `demo_clean.yml`):

| Field | Meaning |
| --- | --- |
| `episode_num` | Number of successful demos to collect (paper uses **50** per variant). |
| `embodiment` | Robot. Dual-arm: `[aloha-agilex]`. Two single arms: `[piper, franka-panda, 0.6]`. |
| `language_num` | Instruction templates instantiated per episode (default 100). |
| `save_freq` | Save every N sim steps. |
| `collect_data` | If `false`, only search seeds, do not write HDF5. |
| `use_seed` | If `true`, replay an existing `seed.txt` instead of searching. |

Supported `embodiment` names: `aloha-agilex`, `ARX-X5`, `piper`, `franka-panda`, `ur5-wsg`.

Camera names are in `task_config/_camera_config.yml` (`D435`, `L515`, and larger variants). Domain-randomization keys match [RoboTwin 2.0 configuration docs](https://robotwin-platform.github.io/doc/usage/configurations.html). Paths in this repo are `task_config/` (not `env_cfg/task_config/`).

Example: collect on Franka instead of Aloha-AgileX:

```yaml
embodiment: [franka-panda]
```

---

## 📦 Data Collection

### Paper protocol (recommended)

For each **task variant** you care about:

```bash
bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
```

Examples:

```bash
# Spatial L1, clean Aloha-AgileX (default embodiment in demo_clean.yml)
bash collect_data.sh Pick_Blocks_Size_1 demo_clean 0

# Same task, domain-randomized scenes
bash collect_data.sh Pick_Blocks_Size_1 demo_randomized 0

# Procedural L5
bash collect_data.sh Place_Phone_Press_Stapler_5 demo_clean 0
```

In the paper:

- Expert scripts in `envs/` are executed with CuRobo motion planning
- Each variant uses **50** successful trajectories
- For each **base task**, a policy is trained on all five levels (**250** demos)
- Instructions: 60 templates per variant (50 train / 10 held-out), instantiated into 100 train / 100 eval utterances
- Clean vs randomized scenes share the same success conditions; randomization only changes visuals and clutter

`task_name` must match an `envs/<task_name>.py` class of the same name, e.g. `Pick_Blocks_Size_1`.

### Resume

Collection is resumable. Existing `seed.txt` and `data/episode*.hdf5` are skipped. If a run dies, run the same command again.

### Instructions only

Templates already live in `description/task_instruction/<task_name>.json`. After HDF5 exists, you can regenerate episode-level instructions with:

```bash
cd description
bash gen_episode_instructions.sh Pick_Blocks_Size_1 demo_clean 100
cd ..
```

---

## 💽 Released Dataset

Pre-collected trajectories (527K, 5 embodiments, clean + randomized) are released at:

**https://huggingface.co/datasets/zxfan/RoboSPA**

Use the Hugging Face dataset if you only need training data. Collect locally when you need a new embodiment, camera, or randomization setting.

---

## 🤖 Other Policy Baselines

These stacks are inherited from RoboTwin 2.0. π0.5 train / eval is in [Quick Start](#-quick-start). For other policies, follow that folder’s README, then run its `eval.sh`. Results land in `eval_result/`.

| Policy | Path | Notes |
| --- | --- | --- |
| RDT | `policy/RDT/` | Paper baseline |
| GO-1 | `policy/GO1/` | Paper baseline; separate GO-1 conda env |
| π0 / π0.5 | `policy/pi0/`, `policy/pi05/` | π0.5 is a paper baseline; see Quick Start |
| X-VLA | — | Paper baseline; see [RMBench X-VLA README](https://github.com/RoboTwin-Platform/RMBench/blob/main/policy/X-VLA/README.md) |
| OpenVLA-OFT | `policy/openvla-oft/` | RLDS conversion required |
| DP / DP3 | `policy/DP/`, `policy/DP3/` | Diffusion policies |
| ACT | `policy/ACT/` | Inherited from RoboTwin 2.0 |
| TinyVLA / DexVLA / LLaVA-VLA | `policy/TinyVLA/`, `policy/DexVLA/`, `policy/LLaVA-VLA/` | Community ports |

X-VLA training and evaluation follow the [RMBench X-VLA README](https://github.com/RoboTwin-Platform/RMBench/blob/main/policy/X-VLA/README.md). Replace RMBench paths and task names with RoboSPA equivalents (`Pick_Blocks_Size_1`, etc.).

Paper eval setting: **clean scene**, **Aloha-AgileX**, **100 trials** per variant, **unseen** instructions.

Official RoboTwin policy docs: [https://robotwin-platform.github.io/doc/usage/index.html](https://robotwin-platform.github.io/doc/usage/index.html). Replace original RoboTwin task names (e.g. `beat_block_hammer`) with RoboSPA names (e.g. `Hit_Blocks_Hammer_Order_1`).

---

## 🧩 Deploy Your Own Policy

Copy `policy/Your_Policy/` and implement three files:

1. `deploy_policy.py` — `get_model`, `eval` (call `TASK_ENV.take_action`), `reset_model`
2. `deploy_policy.yml` — checkpoint paths and `instruction_type`
3. `eval.sh` — CLI overrides passed into `script/eval_policy.py`

Action interfaces:

```python
TASK_ENV.take_action(action, action_type="qpos")      # joints + grippers
TASK_ENV.take_action(action, action_type="ee")         # xyz + quaternion + gripper
TASK_ENV.take_action(action, action_type="delta_ee")
```

`qpos` layout: `[left_arm_joints, left_gripper, right_arm_joints, right_gripper]`.

```bash
cd policy/Your_Policy
bash eval.sh Pick_Blocks_Size_1 demo_clean <ckpt_setting> 0 0
```

More detail: [RoboTwin — Deploy Your Policy](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html).

---

## Task List

`task_name` = `{BaseTask}_{level}` with `level ∈ {1,2,3,4,5}`. Example: `Pick_Blocks_Size_1`, `Place_Phone_Press_Stapler_5`.

### Fine-Grained Spatial Reasoning

| Abbr | Category | Base tasks |
| --- | --- | --- |
| GAC | Geometric Attribute Cognition | `Pick_Blocks_Size`, `Pick_Blocks_Height`, `Pick_Blocks_Length`, `Pick_Blocks_Area` |
| SDE | Spatial Distance Estimation | `Pick_Blocks_Distance`, `Pick_Mugs_Distance`, `Pick_Pill_Bottles_Distance`, `Pick_Mixed_Objects_Distance_A`, `Pick_Mixed_Objects_Distance_B` |
| CPI | Canonical Position Indexing | `Pick_Blocks_Canonical`, `Pick_Cups_Canonical`, `Pick_Rubik_Cubes_Canonical`, `Pick_Mixed_Objects_Canonical_A`, `Pick_Mixed_Objects_Canonical_B` |
| RRR | Referential Relational Reasoning | `Pick_Cans_Relational`, `Pick_Soaps_Relational`, `Pick_Tea_Box_Relational`, `Pick_Mixed_Objects_Relational_A`, `Pick_Mixed_Objects_Relational_B` |
| CVR | Cross-View Reasoning | `Pick_Breads_Multi_View`, `Pick_Sauce_Can_Multi_View`, `Pick_Seals_Multi_View`, `Pick_Mixed_Objects_Multi_View_A`, `Pick_Mixed_Objects_Multi_View_B` |

### Long-Horizon Procedural Planning

| Abbr | Category | Base tasks |
| --- | --- | --- |
| RPF | Repetitive Procedure Following | `Press_Stapler_Repeat`, `Lift_Pot_Repeat`, `Lift_Fan_Repeat`, `Click_Bell_Repeat` |
| OFE | Order-Free Execution | `Put_Bottles_Dustbin`, `Move_Blocks_Apart`, `Move_Playing_Cards_Away`, `Place_Bowls_Plates`, `Separate_Fries_Bread`, `Rank_Blocks_Height`, `Rank_Blocks_Color`, `Rank_Blocks_Size` |
| OCE | Order-Constrained Execution | `Hit_Blocks_Hammer_Order`, `Click_Objects_Order`, `Stamp_Seals_Order`, `Click_Bell_Clockwise_Order`, `Stack_Blocks_Color_Order`, `Stack_Blocks_Size_Order`, `Stack_Blocks_Length_Order` |
| CAC | Composite Action Coordination | `Place_Phone_Press_Stapler`, `Place_Bottle_Cup`, `Hang_Mug_Stack_Blocks`, `Place_Object_Scale_Click`, `Place_Burger_Fries_Click_Bell`, `Click_Can_Place_Items`, `Stamp_Seals_Press_Stapler`, `Click_Bell_Open_Microwave_Place_Object` |
| MIP | Memory-Intensive Planning | `Observe_Blocks_Move_Memory`, `Observe_Objects_Click_Memory`, `Remember_Color_Cover`, `Remember_Orientation_Restore`, `Press_Stapler_Memory` |

Environment files: `envs/<task_name>.py`. Instruction templates: `description/task_instruction/<task_name>.json`. Per-task eval step limits: `task_config/_eval_step_limit.yml`.

---

## Citation

If you use RoboSPA, please cite:

```bibtex
@inproceedings{robospa2026,
  title     = {RoboSPA: Can VLA Models Go Beyond Simple Scenes and Short-Horizon Tasks?},
  author    = {Zhenxuan Fan and Bo Zhang and Yutong Lin and Yuqian Yuan and Juekai Lin
               and Liang Liang and Zhuoyi Huang and Wenqiao Zhang and Juncheng Li
               and Siliang Tang and Jun Xiao and Yueting Zhuang},
  booktitle = {Proceedings of EMNLP},
  year      = {2026}
}
```

---

## 📬 Contact

Questions or suggestions: [zxfan@zju.edu.cn](mailto:zxfan@zju.edu.cn).

---

## 😺 Acknowledgement

Simulation, expert data collection, embodiment assets, and the policy-eval interface are based on [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin). Physics and rendering use [SAPIEN](https://sapien.ucsd.edu/). Motion planning uses [CuRobo](https://github.com/NVlabs/curobo).

---

## License

This repository is released under the MIT license. See [LICENSE](./LICENSE).
