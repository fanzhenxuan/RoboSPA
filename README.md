# RoboSPA

This repository contains task environment code and the corresponding collected demonstration data for RoboSPA.

**We only include the 10 tasks shown in the main figure of the paper.**

## Repository Structure

```text
RoboSPA/
├── envs/
├── data/
└── README.md
```

## envs

The `envs` folder contains the implementation code for the tasks. Each task has a corresponding Python file in this folder.

The task files include:

```text
envs/
├── Click_Bell_Clockwise_Order_5.py
├── Lift_Fan_Repeat_5.py
├── Pick_Blocks_Size_5.py
├── Pick_Mixed_Objects_Canonical_B_5.py
├── Pick_Mixed_Objects_Multi_View_B_5.py
├── Pick_Pill_Bottles_Distance_5.py
├── Pick_Soaps_Relational_5.py
├── Place_Bowls_Plates_5.py
├── Place_Object_Scale_Click_5.py
└── Remember_Orientation_Restore_5.py
```

In addition to task-specific files, `envs` may also contain shared base classes, utilities, robot/camera configurations, and other supporting modules used by the tasks.

## data

The `data` folder contains the collected demonstration data corresponding to the tasks in `envs`.

Each task folder contains one collected trajectory, named `episode0`.

The task data folders include:

```text
data/
├── Click_Bell_Clockwise_Order_5/
├── Lift_Fan_Repeat_5/
├── Pick_Blocks_Size_5/
├── Pick_Mixed_Objects_Canonical_B_5/
├── Pick_Mixed_Objects_Multi_View_B_5/
├── Pick_Pill_Bottles_Distance_5/
├── Pick_Soaps_Relational_5/
├── Place_Bowls_Plates_5/
├── Place_Object_Scale_Click_5/
└── Remember_Orientation_Restore_5/
```

A typical task data folder is organized as follows:

```text
data/
└── Task_Name/
    └── demo_clean_aloha/
        ├── data/
        │   └── episode0.hdf5
        ├── instructions/
        │   └── episode0.json
        ├── video/
        │   └── episode0.mp4
        ├── _traj_data/
        │   └── episode0.pkl
        ├── scene_info.json
        └── seed.txt
```

## File Description

For each task:

- `data/episode0.hdf5` stores the collected trajectory data.
- `instructions/episode0.json` stores the instruction information for the trajectory.
- `video/episode0.mp4` stores the recorded video of the trajectory.
- `_traj_data/episode0.pkl` stores additional trajectory-related data.
- `scene_info.json` stores scene information for the task.
- `seed.txt` stores the random seed used for the task.

## Notes

- `envs` contains the task implementation code.
- `data` contains the collected demonstration data for the corresponding tasks.
- Each task currently contains only one trajectory: `episode0`.
- The task names in `data` correspond to the task implementation files in `envs`.