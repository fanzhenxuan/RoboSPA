from ._base_task import Base_Task
from .utils import *

import sapien
import numpy as np
from copy import deepcopy


class Pick_Blocks_Area_1(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        # 1.同一样本内所有block颜色相同；不同样本颜色随机
        shared_color = (
            np.random.random(),
            np.random.random(),
            np.random.random(),
        )

        # 2.2个block分别对应：更大、更小
        halfsize_lst = [
            np.random.uniform(0.032, 0.033),  # 1st
            np.random.uniform(0.027, 0.028),  # 2nd
        ]
        block_half_height = 0.029
        
        # 3.随机生成2个block位置
        while True:
            block_pose_lst = []

            for i in range(2):
                block_pose = rand_pose(
                    xlim=[-0.22, 0.22],
                    ylim=[-0.15, 0.10],
                    zlim=[0.741 + block_half_height],
                    qpos=[1, 0, 0, 0],
                    ylim_prop=True,
                    rotate_rand=True,
                    rotate_lim=[0, 0, 0.60],
                )

                def check_block_pose(cur_pose, cur_size):
                    for j in range(len(block_pose_lst)):
                        prev_pose = block_pose_lst[j]
                        prev_size = halfsize_lst[j]

                        # 基于尺寸的最小中心距离约束
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
                        xlim=[-0.25, 0.25],
                        ylim=[-0.01, 0.11],
                        zlim=[0.741 + block_half_height],
                        qpos=[1, 0, 0, 0],
                        ylim_prop=True,
                        rotate_rand=True,
                        rotate_lim=[0, 0, 0.60],
                    )

                block_pose_lst.append(deepcopy(block_pose))

            break

        def create_block(block_pose, size, color):
            half_size = (size, size, block_half_height)
            return create_box(
                scene=self,
                pose=block_pose,
                half_size=half_size,
                color=color,
                name="box",
            )

        # 固定对应关系：
        # block1 -> 1st
        # block2 -> 2nd
        self.block1 = create_block(block_pose_lst[0], halfsize_lst[0], shared_color)
        self.block2 = create_block(block_pose_lst[1], halfsize_lst[1], shared_color)

        self.block_lst = [self.block1, self.block2]

        # 给每个block加禁区
        self.add_prohibit_area(self.block1, padding=0.08)
        self.add_prohibit_area(self.block2, padding=0.08)

        # 随机选一个目标block (0 or 1)
        self.target_idx = np.random.randint(0, 2)
        self.target_block = self.block_lst[self.target_idx]

        # 返回大小序数词
        self.rank_str_lst = ["1st", "2nd"]
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

        # 只pick，不place
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.10))

        self.last_gripper = arm_tag
        return arm_tag

    def check_success(self):
        return self.target_block.get_pose().p[2] > 0.82