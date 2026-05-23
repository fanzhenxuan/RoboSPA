from ._base_task import Base_Task
from .utils import *

import sapien
import numpy as np
from copy import deepcopy


class Pick_Blocks_Size_5(Base_Task):

    def setup_demo(self, **kwags):
        # Initialize the task environment with the provided keyword arguments.
        super()._init_task_env_(**kwags)

    def load_actors(self):
        # Use one shared random color for all blocks so the task depends on size,
        # not color differences.
        shared_color = (
            np.random.random(),
            np.random.random(),
            np.random.random(),
        )

        # Define six block sizes in strictly descending order.
        # The 1st block is the largest, and the 6th block is the smallest.
        halfsize_lst = [
            np.random.uniform(0.032, 0.033),  # 1st
            np.random.uniform(0.027, 0.028),  # 2nd
            np.random.uniform(0.022, 0.023),  # 3rd
            np.random.uniform(0.017, 0.018),  # 4th
            np.random.uniform(0.014, 0.015),  # 5th
            np.random.uniform(0.009, 0.010),  # 6th
        ]

        # Number of blocks in the scene.
        n_block = 6

        # Keep sampling block layouts until the overall scene has enough spread.
        while True:
            block_pose_lst = []

            for i in range(n_block):
                # Sample an initial pose for the current block.
                block_pose = rand_pose(
                    xlim=[-0.22, 0.22],
                    ylim=[-0.15, 0.10],
                    zlim=[0.741 + halfsize_lst[i]],
                    qpos=[1, 0, 0, 0],
                    ylim_prop=True,
                    rotate_rand=True,
                    rotate_lim=[0, 0, 0.60],
                )

                def check_block_pose(cur_pose, cur_size):
                    # Ensure the current block is far enough from all previously
                    # sampled blocks, accounting for both block sizes.
                    for j in range(len(block_pose_lst)):
                        prev_pose = block_pose_lst[j]
                        prev_size = halfsize_lst[j]

                        min_dist = cur_size + prev_size + 0.052
                        if np.linalg.norm(cur_pose.p[:2] - prev_pose.p[:2]) < min_dist:
                            return False
                    return True

                # Resample if the block is too close to the center line, too close
                # to the robot base area, or too close to another block.
                while (
                    abs(block_pose.p[0]) < 0.035
                    or np.sum((block_pose.p[:2] - np.array([0.0, -0.12])) ** 2) < 0.015
                    or not check_block_pose(block_pose, halfsize_lst[i])
                ):
                    block_pose = rand_pose(
                        xlim=[-0.31, 0.31],
                        ylim=[-0.02, 0.13],
                        zlim=[0.741 + halfsize_lst[i]],
                        qpos=[1, 0, 0, 0],
                        ylim_prop=True,
                        rotate_rand=True,
                        rotate_lim=[0, 0, 0.60],
                    )

                # Store a copy of the accepted pose.
                block_pose_lst.append(deepcopy(block_pose))

            # Require the full layout to cover enough width and depth so the size
            # comparison scene is visually well distributed.
            xy = np.array([pose.p[:2] for pose in block_pose_lst])
            if (xy[:, 0].max() - xy[:, 0].min() < 0.24) or (xy[:, 1].max() - xy[:, 1].min() < 0.08):
                continue
            break

        def create_block(block_pose, size, color):
            # Create a cube with the given half-size and color.
            half_size = (size, size, size)
            return create_box(
                scene=self,
                pose=block_pose,
                half_size=half_size,
                color=color,
                name="box",
            )

        # Create all block actors and register their prohibited placement areas.
        self.block_lst = []
        for i in range(n_block):
            block = create_block(block_pose_lst[i], halfsize_lst[i], shared_color)
            self.block_lst.append(block)
            self.add_prohibit_area(block, padding=0.075)

        # Keep individual references for compatibility with task templates
        # or downstream code that expects named block attributes.
        self.block1 = self.block_lst[0]
        self.block2 = self.block_lst[1]
        self.block3 = self.block_lst[2]
        self.block4 = self.block_lst[3]
        self.block5 = self.block_lst[4]
        self.block6 = self.block_lst[5]

        # Randomly choose one block as the target to pick.
        self.target_idx = np.random.randint(0, n_block)
        self.target_block = self.block_lst[self.target_idx]

        # Convert the target index into a size-rank phrase.
        self.rank_str_lst = ["1st", "2nd", "3rd", "4th", "5th", "6th"]
        self.target_rank_str = self.rank_str_lst[self.target_idx]

    def play_once(self):
        # Track the previously used gripper for possible arm-switch handling.
        self.last_gripper = None

        # Pick the target block and record which arm was used.
        arm_tag = self.pick_block(self.target_block)

        # Store task placeholders for instruction generation.
        self.info["info"] = {
            "{A}": self.target_rank_str,
            "{a}": str(arm_tag),
        }

        return self.info

    def pick_block(self, block):
        # Choose the left or right arm according to the block's x position.
        block_pose = block.get_pose().p
        arm_tag = ArmTag("left" if block_pose[0] < 0 else "right")

        # If switching arms, move the opposite arm back to origin while grasping.
        if self.last_gripper is not None and (self.last_gripper != arm_tag):
            self.move(
                self.grasp_actor(block, arm_tag=arm_tag, pre_grasp_dis=0.10),
                self.back_to_origin(arm_tag=arm_tag.opposite),
            )
        else:
            # Otherwise, directly grasp the block with the selected arm.
            self.move(
                self.grasp_actor(block, arm_tag=arm_tag, pre_grasp_dis=0.10)
            )

        # Lift the grasped block upward.
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.10))

        # Remember which gripper was used last.
        self.last_gripper = arm_tag
        return arm_tag

    def check_success(self):
        # The task succeeds when the target block has been lifted above a height threshold.
        return self.target_block.get_pose().p[2] > 0.82