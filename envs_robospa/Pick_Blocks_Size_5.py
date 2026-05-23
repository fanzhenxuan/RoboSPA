from ._base_task import Base_Task
from .utils import *

import sapien
import numpy as np
from copy import deepcopy


class Pick_Blocks_Size_5(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        # 同一样本内所有block颜色相同；不同样本颜色随机
        shared_color = (
            np.random.random(),
            np.random.random(),
            np.random.random(),
        )

        # 6个block：从大到小
        # 注意这里是half_size
        # halfsize_lst = [
        #     np.random.uniform(0.031, 0.033),  # 1st
        #     np.random.uniform(0.028, 0.030),  # 2nd
        #     np.random.uniform(0.025, 0.027),  # 3rd
        #     np.random.uniform(0.022, 0.024),  # 4th
        #     np.random.uniform(0.019, 0.021),  # 5th
        #     np.random.uniform(0.016, 0.018),  # 6th
        # ]

        halfsize_lst = [
            np.random.uniform(0.032, 0.033),  # 1st
            np.random.uniform(0.027, 0.028),  # 2nd
            np.random.uniform(0.022, 0.023),  # 3rd
            np.random.uniform(0.017, 0.018),  # 4th
            np.random.uniform(0.014, 0.015),  # 5th
            np.random.uniform(0.009, 0.010),  # 6th
        ]
        
        n_block = 6

        while True:
            block_pose_lst = []

            for i in range(n_block):
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
                    for j in range(len(block_pose_lst)):
                        prev_pose = block_pose_lst[j]
                        prev_size = halfsize_lst[j]

                        min_dist = cur_size + prev_size + 0.052
                        if np.linalg.norm(cur_pose.p[:2] - prev_pose.p[:2]) < min_dist:
                            return False
                    return True

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

                block_pose_lst.append(deepcopy(block_pose))

            # 避免整体过于挤在中间：要求x和y都有一定离散度
            xy = np.array([pose.p[:2] for pose in block_pose_lst])
            if (xy[:, 0].max() - xy[:, 0].min() < 0.24) or (xy[:, 1].max() - xy[:, 1].min() < 0.08):
                continue
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

        self.block_lst = []
        for i in range(n_block):
            block = create_block(block_pose_lst[i], halfsize_lst[i], shared_color)
            self.block_lst.append(block)
            self.add_prohibit_area(block, padding=0.075)

        self.block1 = self.block_lst[0]
        self.block2 = self.block_lst[1]
        self.block3 = self.block_lst[2]
        self.block4 = self.block_lst[3]
        self.block5 = self.block_lst[4]
        self.block6 = self.block_lst[5]

        # 随机选一个目标block
        self.target_idx = np.random.randint(0, n_block)
        self.target_block = self.block_lst[self.target_idx]

        self.rank_str_lst = ["1st", "2nd", "3rd", "4th", "5th", "6th"]
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