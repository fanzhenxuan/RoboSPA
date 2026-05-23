from copy import deepcopy
from ._base_task import Base_Task
from .utils import *
import sapien
import numpy as np
import random

class RememberOrientationRestoreBase(Base_Task):
    """
    无颜色版：
    1. 前区桌面上展示 SHOW_NUM 个 T 型块，朝向随机
    2. observe 2 秒
    3. 挡住前区
    4. 后区桌面上初始化一排 SHOW_NUM 个 T 型块，朝向随机
    5. 机器人按“从左到右”的顺序，逐个恢复后区 T 的朝向
       - 前区第 i 个 T 的朝向 -> 后区第 i 个 T
    """

    SHOW_NUM = 1

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)
        self.max_reward = 0.0

    def yaw_quat(self, theta):
        # 四元数顺序：[w, x, y, z]
        return np.array(
            [np.cos(theta / 2), 0.0, 0.0, np.sin(theta / 2)],
            dtype=float,
        )

    def quat_angle_diff_rad(self, q, q_ref):
        q = np.asarray(q, dtype=float)
        q_ref = np.asarray(q_ref, dtype=float)

        q /= (np.linalg.norm(q) + 1e-12)
        q_ref /= (np.linalg.norm(q_ref) + 1e-12)

        dot = abs(np.dot(q, q_ref))
        dot = np.clip(dot, -1.0, 1.0)
        return 2.0 * np.arccos(dot)

    def create_t_block(self, pose, model_id):
        actor = create_actor(
            scene=self,
            pose=pose,
            modelname="007_T_block",
            model_id=model_id,
            convex=True,
        )
        if actor is None:
            raise RuntimeError("Failed to load model '007_T_block'.")
        return actor

    def choose_arm_by_x(self, x):
        return ArmTag("left" if x < 0 else "right")

    def load_actors(self):
        if not (1 <= self.SHOW_NUM <= 5):
            raise ValueError(f"SHOW_NUM must be in [1, 5], got {self.SHOW_NUM}")

        # =========================
        # 基础状态
        # =========================
        self.get_obs_cnt = 0
        self.orig_left_endpose = self.get_arm_pose("left")
        self.orig_right_endpose = self.get_arm_pose("right")
        self.num_pairs = self.SHOW_NUM
        self.stage_sum = self.num_pairs
        self.stage = 0
        self.progress = 0
        self.task_success = [0] * self.stage_sum
        self.fail_flag = False
        self.wall = None
        self.pre_arm = None

        self.front_blocks = []
        self.rear_blocks = []

        self.front_dir_names = []
        self.rear_dir_names = []

        self.front_target_quats = []
        self.rear_anchor_poses = []

        self.DIR_TO_YAW = {
            "up": 0.0,
            "right": -np.pi / 2,
            "down": np.pi,
            "left": np.pi / 2,
        }
        self.DIR_NAMES = ["up", "right", "down", "left"]
        self.model_id_front = 0 if random.random() < 0.5 else 1
        self.model_id_rear = 1 - self.model_id_front
        # =========================
        # 前区 / 挡板 / 后区位置
        # =========================
        # 前区直接放桌面前半区
        self.front_y = 0.18
        # 挡板放在前后区中间
        self.wall_y = 0.10
        # 后区放在靠机器人一侧
        self.rear_y = -0.10

        self.block_half_size = 0.015
        self.table_z = 0.741 + self.block_half_size

        # =========================
        # 前区展示 SHOW_NUM 个 T 型块（直接放桌面）
        # =========================
        if self.num_pairs == 1:
            front_xs = [0.0]
        else:
            front_xs = np.linspace(-0.20, 0.20, self.num_pairs).tolist()

        for i in range(self.num_pairs):
            dir_name = np.random.choice(self.DIR_NAMES)
            target_q = self.yaw_quat(self.DIR_TO_YAW[dir_name])

            pose = sapien.Pose(
                p=[front_xs[i], self.front_y, self.table_z],
                q=target_q,
            )

            block = self.create_t_block(pose, self.model_id_front)

            self.front_blocks.append(block)
            self.front_dir_names.append(dir_name)
            self.front_target_quats.append(target_q)

        # =========================
        # 后区一排 SHOW_NUM 个 T 型块
        # 朝向随机，但尽量与对应前区不同
        # =========================
        if self.num_pairs == 1:
            rear_xs = [0.0]
        else:
            rear_xs = np.linspace(-0.23, 0.23, self.num_pairs).tolist()

        for i in range(self.num_pairs):
            target_dir_name = self.front_dir_names[i]
            candidate_dirs = [d for d in self.DIR_NAMES if d != target_dir_name]
            rear_dir_name = np.random.choice(candidate_dirs)
            rear_q = self.yaw_quat(self.DIR_TO_YAW[rear_dir_name])

            pose = sapien.Pose(
                p=[rear_xs[i], self.rear_y, self.table_z],
                q=rear_q,
            )

            block = self.create_t_block(pose, self.model_id_rear)

            self.rear_blocks.append(block)
            self.rear_dir_names.append(rear_dir_name)
            self.rear_anchor_poses.append(deepcopy(pose))

        for block in self.rear_blocks:
            self.add_prohibit_area(block, padding=0.03)

    def add_wall(self):
        if self.wall is not None:
            return
        self.wall = create_box(
            self.scene,
            sapien.Pose(p=[0, self.wall_y, 0.95]),
            half_size=[0.34, 0.005, 0.2],
            color=(1, 0.9, 0.9),
            name="wall",
            is_static=True,
        )

    def update_progress(self):
        progress = 0
        angle_th = np.deg2rad(15.0)

        for i, block in enumerate(self.rear_blocks):
            pose = block.get_pose()
            anchor_pose = self.rear_anchor_poses[i]
            target_q = self.front_target_quats[i]

            pos_diff = np.linalg.norm(pose.p[:2] - anchor_pose.p[:2])
            angle_diff = self.quat_angle_diff_rad(pose.q, target_q)

            if (
                pos_diff < 0.03
                and angle_diff < angle_th
                and pose.p[2] < 0.77
            ):
                progress += 1

        self.progress = progress
        self.stage = progress

        for i in range(self.stage_sum):
            self.task_success[i] = int(self.progress >= i + 1)

    def rotate_one_block(self, idx):
        block = self.rear_blocks[idx]
        anchor_pose = self.rear_anchor_poses[idx]
        target_q = self.front_target_quats[idx]

        arm_tag = self.choose_arm_by_x(anchor_pose.p[0])
        target_pose = np.concatenate([anchor_pose.p, target_q])

        if self.pre_arm is not None and self.pre_arm != arm_tag:
            self.move(
                self.grasp_actor(
                    block,
                    arm_tag=arm_tag,
                    pre_grasp_dis=0.08,
                    gripper_pos=0.2,
                ),
                self.back_to_origin(arm_tag=self.pre_arm),
            )
        else:
            self.move(
                self.grasp_actor(
                    block,
                    arm_tag=arm_tag,
                    pre_grasp_dis=0.08,
                    gripper_pos=0.2,
                )
            )

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.08))
        self.move(
            self.place_actor(
            block,
            arm_tag=arm_tag,
            target_pose=target_pose,
            functional_point_id=0,
            pre_dis=0.05,
            dis=0.02,
            constrain="align",
        )
        )
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.06, move_axis="arm"))

        self.pre_arm = arm_tag

    def play_once(self):
        # observe 2 秒
        self.delay(delay_time=2, save_freq=-1)

        # 挡住前区
        self.add_wall()

        # 稍停一下
        self.delay(delay_time=1, save_freq=-1)

        self.stage = 0
        self.progress = 0

        # 按后区从左到右恢复
        sorted_pairs = sorted(
            list(enumerate(self.rear_blocks)),
            key=lambda x: x[1].get_pose().p[0]
        )

        for idx, _ in sorted_pairs:
            self.rotate_one_block(idx)
            self.update_progress()
        self.move(self.open_gripper(arm_tag=self.pre_arm))
        self.get_obs_cnt = 10000
        if not hasattr(self, "info") or self.info is None:
            self.info = {}

        self.info["info"] = {}
        return self.info

    def check_success(self):
        if self.fail_flag:
            return False
        self.update_progress()
        print(f"task_success: {self.task_success}, stage: {self.stage}/{self.stage_sum}")
        self.get_obs_cnt += 1
        if self.get_obs_cnt == 500:
            self.add_wall()
        elif self.get_obs_cnt < 500:
            current_left_endpose = self.get_arm_pose("left")
            current_right_endpose = self.get_arm_pose("right")
            if np.linalg.norm(np.array(current_left_endpose[:3]) - np.array(self.orig_left_endpose[:3])) > 0.03 or \
               np.linalg.norm(np.array(current_right_endpose[:3]) - np.array(self.orig_right_endpose[:3])) > 0.03:
                print("Arm position deviation detected!")
                self.fail_flag = True
            return False

        return (self.progress >= self.stage_sum and self.robot.is_left_gripper_open() and self.robot.is_right_gripper_open())