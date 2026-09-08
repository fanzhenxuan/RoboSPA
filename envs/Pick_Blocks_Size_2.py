from ._base_task import Base_Task
from .utils import *

import sapien
import numpy as np
from copy import deepcopy


class Pick_Blocks_Size_2(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        # 1. All blocks in one episode share a color; color is random across episodes
        shared_color = (
            np.random.random(),
            np.random.random(),
            np.random.random(),
        )

        # 2. The 3 blocks correspond to 1st-largest, 2nd-largest, 3rd-largest
        # Note: these are half-sizes
        halfsize_lst = [
            np.random.uniform(0.032, 0.033),  # 1st
            np.random.uniform(0.027, 0.028),  # 2nd
            np.random.uniform(0.022, 0.023),  # 3rd
        ]

        # 3. Sample 3 block poses, spread out to avoid hitting other blocks while grasping
        while True:
            block_pose_lst = []

            for i in range(3):
                block_pose = rand_pose(
                    xlim=[-0.27, 0.27],
                    ylim=[-0.15, 0.10],
                    zlim=[0.741 + halfsize_lst[i]],
                    qpos=[1, 0, 0, 0],
                    ylim_prop=True,
                    rotate_rand=True,
                    rotate_lim=[0, 0, 0.60],
                )

                def check_block_pose(cur_pose, cur_size):
                    for j in range(len(block_pose_lst)):
                        prev_pose = block_pose_lst[j]
                        prev_size = halfsize_lst[j]

                        # Minimum center-distance constraint based on size
                        min_dist = cur_size + prev_size + 0.055
                        if np.linalg.norm(cur_pose.p[:2] - prev_pose.p[:2]) < min_dist:
                            return False
                    return True

                while (
                    abs(block_pose.p[0]) < 0.04
                    or np.sum((block_pose.p[:2] - np.array([0.0, -0.12])) ** 2) < 0.015
                    or not check_block_pose(block_pose, halfsize_lst[i])
                ):
                    block_pose = rand_pose(
                        xlim=[-0.22, 0.22],
                        ylim=[-0.01, 0.11],
                        zlim=[0.741 + halfsize_lst[i]],
                        qpos=[1, 0, 0, 0],
                        ylim_prop=True,
                        rotate_rand=True,
                        rotate_lim=[0, 0, 0.60],
                    )

                block_pose_lst.append(deepcopy(block_pose))

            # Avoid 3 blocks lining up, to reduce occlusion
            eps = [0.13, 0.035]
            block1_pose = block_pose_lst[0].p
            block2_pose = block_pose_lst[1].p
            block3_pose = block_pose_lst[2].p

            if (
                np.all(abs(block1_pose[:2] - block2_pose[:2]) < eps)
                and np.all(abs(block2_pose[:2] - block3_pose[:2]) < eps)
                and block1_pose[0] < block2_pose[0]
                and block2_pose[0] < block3_pose[0]
            ):
                continue
            else:
                break

        def create_block(block_pose, size, color):
            half_size = (size, size, size)
            return create_box(
                scene=self,
                pose=block_pose,
                half_size=half_size,
                color=color,
                name="box",
            )

        # Fixed mapping:
        # block1 -> 1st
        # block2 -> 2nd
        # block3 -> 3rd
        self.block1 = create_block(block_pose_lst[0], halfsize_lst[0], shared_color)
        self.block2 = create_block(block_pose_lst[1], halfsize_lst[1], shared_color)
        self.block3 = create_block(block_pose_lst[2], halfsize_lst[2], shared_color)

        self.block_lst = [self.block1, self.block2, self.block3]

        # Add a forbidden region around each block to reduce collisions
        self.add_prohibit_area(self.block1, padding=0.08)
        self.add_prohibit_area(self.block2, padding=0.08)
        self.add_prohibit_area(self.block3, padding=0.08)

        # Randomly pick a target block
        self.target_idx = np.random.randint(0, 3)
        self.target_block = self.block_lst[self.target_idx]

        # Return the size ordinal
        self.rank_str_lst = ["1st", "2nd", "3rd"]
        self.target_rank_str = self.rank_str_lst[self.target_idx]

    def play_once(self):
        self.last_gripper = None

        arm_tag = self.pick_block(self.target_block)

        self.info["info"] = {
            "{A}": self.target_rank_str,
            "{a}": str(arm_tag),
        }

        return self.info

    def pick_block(self, block):
        block_pose = block.get_pose().p
        arm_tag = ArmTag("left" if block_pose[0] < 0 else "right")

        if self.last_gripper is not None and (self.last_gripper != arm_tag):
            self.move(
                self.grasp_actor(block, arm_tag=arm_tag, pre_grasp_dis=0.10),
                self.back_to_origin(arm_tag=arm_tag.opposite),
            )
        else:
            self.move(
                self.grasp_actor(block, arm_tag=arm_tag, pre_grasp_dis=0.10)
            )

        # pick only, do not place
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.10))

        self.last_gripper = arm_tag
        return arm_tag

    def check_success(self):
        return self.target_block.get_pose().p[2] > 0.82
