from ._base_task import Base_Task
from .utils import *
import sapien
import math
import numpy as np
from copy import deepcopy


class Rank_Blocks_Height_2(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.stage_sum = 2
        self.stage = 0
        self.task_success = [0, 0]

        color = (
            np.random.random(),
            np.random.random(),
            np.random.random(),
        )

        # 固定底面尺寸，只改变高度
        base_x = np.random.uniform(0.013, 0.015)

        # 在原 5 个里继续去掉最高、最低，再从剩下 3 个里去掉较低的
        # 最终保留：高、中
        half_height_lst = [
            np.random.uniform(0.042, 0.044),  # tall
            np.random.uniform(0.028, 0.030),  # medium
        ]

        # ===== while修改区开始（外层整体采样）=====
        # while True:
        #     block_pose_lst = []
        #     for i in range(2):
        #         block_pose = rand_pose(
        #             xlim=[-0.28, 0.28],
        #             ylim=[-0.08, 0.05],
        #             zlim=[0.741 + half_height_lst[i]],
        #             qpos=[1, 0, 0, 0],
        #             ylim_prop=True,
        #             rotate_rand=True,
        #             rotate_lim=[0, 0, 0.75],
        #         )
        #
        #         def check_block_pose(block_pose):
        #             for j in range(len(block_pose_lst)):
        #                 if np.sum((block_pose.p[:2] - block_pose_lst[j].p[:2]) ** 2) < 0.01:
        #                     return False
        #             return True
        #
        #         while (
        #             abs(block_pose.p[0]) < 0.05
        #             or np.sum((block_pose.p[:2] - np.array([0, -0.1])) ** 2) < 0.01
        #             or not check_block_pose(block_pose)
        #         ):
        #             block_pose = rand_pose(
        #                 xlim=[-0.28, 0.28],
        #                 ylim=[-0.08, 0.05],
        #                 zlim=[0.741 + half_height_lst[i]],
        #                 qpos=[1, 0, 0, 0],
        #                 ylim_prop=True,
        #                 rotate_rand=True,
        #                 rotate_lim=[0, 0, 0.75],
        #             )
        #
        #         block_pose_lst.append(deepcopy(block_pose))
        #
        #     eps = [0.12, 0.03]
        #     block1_pose = block_pose_lst[0].p
        #     block2_pose = block_pose_lst[1].p
        #
        #     aligned_close = np.all(abs(block1_pose[:2] - block2_pose[:2]) < eps)
        #     ordered_x = block1_pose[0] < block2_pose[0]
        #
        #     if aligned_close or ordered_x:
        #         continue
        #     else:
        #         break
    
        outer_max_trials = 100
        outer_trials = 0
    
        while outer_trials < outer_max_trials:
            block_pose_lst = []
            for i in range(2):
                block_pose = rand_pose(
                    xlim=[-0.28, 0.28],
                    ylim=[-0.08, 0.05],
                    zlim=[0.741 + half_height_lst[i]],
                    qpos=[1, 0, 0, 0],
                    ylim_prop=True,
                    rotate_rand=True,
                    rotate_lim=[0, 0, 0.75],
                )
    
                def check_block_pose(block_pose):
                    for j in range(len(block_pose_lst)):
                        if np.sum((block_pose.p[:2] - block_pose_lst[j].p[:2]) ** 2) < 0.01:
                            return False
                    return True
    
                # ===== while修改区开始（内层单个block采样）=====    
                inner_max_trials = 100
                inner_trials = 0
    
                while (
                    abs(block_pose.p[0]) < 0.05
                    or np.sum((block_pose.p[:2] - np.array([0, -0.1])) ** 2) < 0.01
                    or not check_block_pose(block_pose)
                ) and inner_trials < inner_max_trials:
                    block_pose = rand_pose(
                        xlim=[-0.28, 0.28],
                        ylim=[-0.08, 0.05],
                        zlim=[0.741 + half_height_lst[i]],
                        qpos=[1, 0, 0, 0],
                        ylim_prop=True,
                        rotate_rand=True,
                        rotate_lim=[0, 0, 0.75],
                    )
                    inner_trials += 1
    
                if (
                    abs(block_pose.p[0]) < 0.05
                    or np.sum((block_pose.p[:2] - np.array([0, -0.1])) ** 2) < 0.01
                    or not check_block_pose(block_pose)
                ):
                    raise RuntimeError(f"Failed to sample a valid block_pose for block {i} within 100 tries.")
                # ===== while修改区结束（内层单个block采样）=====
    
                block_pose_lst.append(deepcopy(block_pose))
    
            eps = [0.12, 0.03]
            block1_pose = block_pose_lst[0].p
            block2_pose = block_pose_lst[1].p
    
            aligned_close = np.all(abs(block1_pose[:2] - block2_pose[:2]) < eps)
            ordered_x = block1_pose[0] < block2_pose[0]
    
            if aligned_close or ordered_x:
                outer_trials += 1
                continue
            else:
                break
    
        if outer_trials >= outer_max_trials:
            raise RuntimeError("Failed to sample valid block poses arrangement within 100 tries.")
        # ===== while修改区结束（外层整体采样）=====

        def create_block(block_pose, half_height, color):
            half_size = (base_x, base_x, half_height)
            return create_box(
                scene=self,
                pose=block_pose,
                half_size=half_size,
                color=color,
                name="box",
            )

        # 从高到低
        self.block1 = create_block(block_pose_lst[0], half_height_lst[0], color)
        self.block2 = create_block(block_pose_lst[1], half_height_lst[1], color)

        self.add_prohibit_area(self.block1, padding=0.1)
        self.add_prohibit_area(self.block2, padding=0.1)
        self.prohibited_area.append([-0.27, -0.22, 0.27, -0.12])

        y_pose = np.random.uniform(-0.2, -0.1)

        # 从左到右：高 -> 中
        self.block1_target_pose = [
            np.random.uniform(-0.08, -0.06),
            y_pose,
            0.74 + self.table_z_bias,
        ] + [0, 1, 0, 0]

        self.block2_target_pose = [
            np.random.uniform(0.06, 0.08),
            y_pose,
            0.74 + self.table_z_bias,
        ] + [0, 1, 0, 0]

    def play_once(self):
        self.last_gripper = None
        self.stage = 0
        self.task_success = [0, 0]

        arm_tag1 = self.pick_and_place_block(self.block1, self.block1_target_pose)
        self.update_progress()

        arm_tag2 = self.pick_and_place_block(self.block2, self.block2_target_pose)
        self.update_progress()


        self.info["info"] = {
            # "{A}": "tall block",
            # "{B}": "medium-height block",
            # "{a}": arm_tag1,
            # "{b}": arm_tag2,
        }
        return self.info

    def pick_and_place_block(self, block, target_pose=None):
        block_pose = block.get_pose().p
        arm_tag = ArmTag("left" if block_pose[0] < 0 else "right")

        if self.last_gripper is not None and (self.last_gripper != arm_tag):
            self.move(
                self.grasp_actor(block, arm_tag=arm_tag, pre_grasp_dis=0.09),
                self.back_to_origin(arm_tag=arm_tag.opposite),
            )
        else:
            self.move(self.grasp_actor(block, arm_tag=arm_tag, pre_grasp_dis=0.09))

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07))

        self.move(
            self.place_actor(
                block,
                target_pose=target_pose,
                arm_tag=arm_tag,
                functional_point_id=0,
                pre_dis=0.09,
                dis=0.02,
                constrain="align",
            )
        )
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07, move_axis="arm"))

        self.last_gripper = arm_tag
        return str(arm_tag)

    def update_progress(self):
        block1_pose = self.block1.get_pose().p
        block2_pose = self.block2.get_pose().p

        self.task_success[0] = 1
        self.task_success[1] = int(
            block2_pose[0] > block1_pose[0]
            and abs(block2_pose[1] - block1_pose[1]) <= 0.03
        )

        self.stage = sum(self.task_success)

    def check_success(self):
        self.update_progress()
        return self.task_success == [1, 1]