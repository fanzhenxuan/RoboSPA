from copy import deepcopy
from ._base_task import Base_Task
from .utils import *
import sapien
import numpy as np
import random


class RememberOrientationRestoreBase(Base_Task):
    # Number of front/rear T-block pairs shown in the task.
    SHOW_NUM = 1

    def setup_demo(self, **kwags):
        # Initialize the task environment and reset the maximum reward.
        super()._init_task_env_(**kwags)
        self.max_reward = 0.0

    def yaw_quat(self, theta):
        # Convert a yaw angle around the z-axis into a quaternion.
        return np.array(
            [np.cos(theta / 2), 0.0, 0.0, np.sin(theta / 2)],
            dtype=float,
        )

    def quat_angle_diff_rad(self, q, q_ref):
        # Compute the angular difference between two quaternions in radians.
        q = np.asarray(q, dtype=float)
        q_ref = np.asarray(q_ref, dtype=float)

        # Normalize both quaternions to avoid scale-related numerical errors.
        q /= (np.linalg.norm(q) + 1e-12)
        q_ref /= (np.linalg.norm(q_ref) + 1e-12)

        # q and -q represent the same rotation, so use the absolute dot product.
        dot = abs(np.dot(q, q_ref))
        dot = np.clip(dot, -1.0, 1.0)
        return 2.0 * np.arccos(dot)

    def create_t_block(self, pose, model_id):
        # Create one T-shaped block actor.
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
        # Use the left arm for objects on the left side, otherwise use the right arm.
        return ArmTag("left" if x < 0 else "right")

    def load_actors(self):
        # SHOW_NUM controls how many orientation pairs are displayed.
        if not (1 <= self.SHOW_NUM <= 5):
            raise ValueError(f"SHOW_NUM must be in [1, 5], got {self.SHOW_NUM}")

        # Save initial arm poses so the task can verify that arms stay still
        # before the memory wall appears.
        self.get_obs_cnt = 0
        self.orig_left_endpose = self.get_arm_pose("left")
        self.orig_right_endpose = self.get_arm_pose("right")

        # Initialize task progress state.
        self.num_pairs = self.SHOW_NUM
        self.stage_sum = self.num_pairs
        self.stage = 0
        self.progress = 0
        self.task_success = [0] * self.stage_sum
        self.fail_flag = False
        self.wall = None
        self.pre_arm = None

        # Front blocks show the target orientations; rear blocks must be restored to match them.
        self.front_blocks = []
        self.rear_blocks = []

        # Store direction names for debugging or future instruction generation.
        self.front_dir_names = []
        self.rear_dir_names = []

        # Store target quaternions and original rear-block anchor poses.
        self.front_target_quats = []
        self.rear_anchor_poses = []

        # Map readable directions to yaw rotations.
        self.DIR_TO_YAW = {
            "up": 0.0,
            "right": -np.pi / 2,
            "down": np.pi,
            "left": np.pi / 2,
        }
        self.DIR_NAMES = ["up", "right", "down", "left"]

        # Use different T-block model variants for the front and rear rows.
        self.model_id_front = 0 if random.random() < 0.5 else 1
        self.model_id_rear = 1 - self.model_id_front

        # Fixed y positions for the front row, wall, and rear row.
        self.front_y = 0.18
        self.wall_y = 0.10
        self.rear_y = -0.10

        # T-block placement height on the table.
        self.block_half_size = 0.015
        self.table_z = 0.741 + self.block_half_size

        # Spread front blocks evenly across x.
        if self.num_pairs == 1:
            front_xs = [0.0]
        else:
            front_xs = np.linspace(-0.20, 0.20, self.num_pairs).tolist()

        # Create front blocks with random target orientations.
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

        # Spread rear blocks evenly across x.
        if self.num_pairs == 1:
            rear_xs = [0.0]
        else:
            rear_xs = np.linspace(-0.23, 0.23, self.num_pairs).tolist()

        # Create rear blocks with orientations intentionally different from the targets.
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

        # Prevent other objects from being placed too close to the rear blocks.
        for block in self.rear_blocks:
            self.add_prohibit_area(block, padding=0.03)

    def add_wall(self):
        # Add a visual wall between the front and rear rows.
        # This is used to hide the target row after the observation phase.
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
        # Count how many rear blocks are back at their anchor positions
        # and rotated to match the corresponding front target orientation.
        progress = 0
        angle_th = np.deg2rad(15.0)

        for i, block in enumerate(self.rear_blocks):
            pose = block.get_pose()
            anchor_pose = self.rear_anchor_poses[i]
            target_q = self.front_target_quats[i]

            # The rear block should remain near its original xy anchor.
            pos_diff = np.linalg.norm(pose.p[:2] - anchor_pose.p[:2])

            # The rear block orientation should match the remembered target orientation.
            angle_diff = self.quat_angle_diff_rad(pose.q, target_q)

            if (
                pos_diff < 0.03
                and angle_diff < angle_th
                and pose.p[2] < 0.77
            ):
                progress += 1

        self.progress = progress
        self.stage = progress

        # Convert total progress into per-stage success flags.
        for i in range(self.stage_sum):
            self.task_success[i] = int(self.progress >= i + 1)

    def rotate_one_block(self, idx):
        # Rotate one rear block to match the corresponding front target orientation.
        block = self.rear_blocks[idx]
        anchor_pose = self.rear_anchor_poses[idx]
        target_q = self.front_target_quats[idx]

        # Select arm by the block's anchor x position.
        arm_tag = self.choose_arm_by_x(anchor_pose.p[0])

        # Target pose keeps the rear block at its anchor position but changes orientation.
        target_pose = np.concatenate([anchor_pose.p, target_q])

        # If switching arms, send the previously used arm back to origin.
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

        # Lift the block before placing it with the target orientation.
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.08))

        # Place the block back at its anchor pose with the remembered target orientation.
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

        # Lift the gripper away after placement.
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.06, move_axis="arm"))

        # Record the last used arm.
        self.pre_arm = arm_tag

    def play_once(self):
        # Give the agent time to observe the front target orientations.
        self.delay(delay_time=2, save_freq=-1)

        # Add the wall to hide the front row after observation.
        self.add_wall()

        # Wait briefly after adding the wall.
        self.delay(delay_time=1, save_freq=-1)

        # Reset progress before executing the restoration actions.
        self.stage = 0
        self.progress = 0

        # Rotate rear blocks from left to right for a deterministic demonstration order.
        sorted_pairs = sorted(
            list(enumerate(self.rear_blocks)),
            key=lambda x: x[1].get_pose().p[0]
        )

        for idx, _ in sorted_pairs:
            self.rotate_one_block(idx)
            self.update_progress()

        # Open the last used gripper after all blocks are placed.
        self.move(self.open_gripper(arm_tag=self.pre_arm))

        # Skip the delayed observation/failure logic after demonstration playback.
        self.get_obs_cnt = 10000

        if not hasattr(self, "info") or self.info is None:
            self.info = {}

        self.info["info"] = {}
        return self.info

    def check_success(self):
        # Immediately fail if an earlier safety or observation check failed.
        if self.fail_flag:
            return False

        # Refresh progress state.
        self.update_progress()
        print(f"task_success: {self.task_success}, stage: {self.stage}/{self.stage_sum}")

        # During the early observation window, ensure the robot arms stay near their initial poses.
        self.get_obs_cnt += 1
        if self.get_obs_cnt == 500:
            self.add_wall()
        elif self.get_obs_cnt < 500:
            current_left_endpose = self.get_arm_pose("left")
            current_right_endpose = self.get_arm_pose("right")

            if (
                np.linalg.norm(np.array(current_left_endpose[:3]) - np.array(self.orig_left_endpose[:3])) > 0.03
                or np.linalg.norm(np.array(current_right_endpose[:3]) - np.array(self.orig_right_endpose[:3])) > 0.03
            ):
                print("Arm position deviation detected!")
                self.fail_flag = True

            return False

        # Final success requires all rear blocks restored and both grippers open.
        return (
            self.progress >= self.stage_sum
            and self.robot.is_left_gripper_open()
            and self.robot.is_right_gripper_open()
        )