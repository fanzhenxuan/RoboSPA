from ._base_task import Base_Task
from .utils import *
import sapien
import math
import numpy as np
from copy import deepcopy


class Rank_Blocks_Color_3(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.stage_sum = 3
        self.stage = 0
        self.task_success = [0, 0, 0]

        # ===== while修改区开始 =====
        # while True:
        #     block_pose_lst = []
        #     for i in range(3):
        #         block_pose = rand_pose(
        #             xlim=[-0.28, 0.28],
        #             ylim=[-0.08, 0.05],
        #             zlim=[0.765],
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
        #                 zlim=[0.765],
        #                 qpos=[1, 0, 0, 0],
        #                 ylim_prop=True,
        #                 rotate_rand=True,
        #                 rotate_lim=[0, 0, 0.75],
        #             )
        #         block_pose_lst.append(deepcopy(block_pose))
        #
        #     eps = [0.12, 0.03]
        #     block1_pose = block_pose_lst[0].p
        #     block2_pose = block_pose_lst[1].p
        #     block3_pose = block_pose_lst[2].p
        #
        #     if ((
        #         np.all(abs(block1_pose[:2] - block2_pose[:2]) < eps)
        #         and np.all(abs(block2_pose[:2] - block3_pose[:2]) < eps))
        #         or (block1_pose[0] < block2_pose[0] < block3_pose[0])
        #     ):
        #         continue
        #     else:
        #         break
    
        outer_max_trials = 100
        outer_trials = 0
    
        while outer_trials < outer_max_trials:
            block_pose_lst = []
            for i in range(3):
                block_pose = rand_pose(
                    xlim=[-0.28, 0.28],
                    ylim=[-0.08, 0.05],
                    zlim=[0.765],
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
                        zlim=[0.765],
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
    
                block_pose_lst.append(deepcopy(block_pose))
    
            eps = [0.12, 0.03]
            block1_pose = block_pose_lst[0].p
            block2_pose = block_pose_lst[1].p
            block3_pose = block_pose_lst[2].p
    
            if ((
                np.all(abs(block1_pose[:2] - block2_pose[:2]) < eps)
                and np.all(abs(block2_pose[:2] - block3_pose[:2]) < eps))
                or (block1_pose[0] < block2_pose[0] < block3_pose[0])
            ):
                outer_trials += 1
                continue
            else:
                break
    
        if outer_trials >= outer_max_trials:
            raise RuntimeError("Failed to sample valid block poses arrangement within 100 tries.")
        # ===== while修改区结束 =====

        size = np.random.uniform(0.015, 0.025)
        half_size = (size, size, size)

        self.block1 = create_box(
            scene=self,
            pose=block_pose_lst[0],
            half_size=half_size,
            color=(1, 0, 0),
            name="box",
        )
        self.block2 = create_box(
            scene=self,
            pose=block_pose_lst[1],
            half_size=half_size,
            color=(0, 1, 0),
            name="box",
        )
        self.block3 = create_box(
            scene=self,
            pose=block_pose_lst[2],
            half_size=half_size,
            color=(0, 0, 1),
            name="box",
        )

        self.add_prohibit_area(self.block1, padding=0.05)
        self.add_prohibit_area(self.block2, padding=0.05)
        self.add_prohibit_area(self.block3, padding=0.05)

        self.prohibited_area.append([-0.17, -0.22, 0.17, -0.12])

        y_pose = np.random.uniform(-0.2, -0.1)

        # 从左到右：红 绿 蓝
        self.block1_target_pose = [
            np.random.uniform(-0.16, -0.14),
            y_pose,
            0.74 + self.table_z_bias,
        ] + [0, 1, 0, 0]

        self.block2_target_pose = [
            np.random.uniform(-0.08, -0.06),
            y_pose,
            0.74 + self.table_z_bias,
        ] + [0, 1, 0, 0]

        self.block3_target_pose = [
            np.random.uniform(-0.01, 0.01),
            y_pose,
            0.74 + self.table_z_bias,
        ] + [0, 1, 0, 0]

    def play_once(self):
        self.last_gripper = None
        self.stage = 0
        self.task_success = [0, 0, 0]

        arm_tag1 = self.pick_and_place_block(self.block1, self.block1_target_pose)
        self.update_progress()

        arm_tag2 = self.pick_and_place_block(self.block2, self.block2_target_pose)
        self.update_progress()

        arm_tag3 = self.pick_and_place_block(self.block3, self.block3_target_pose)
        self.update_progress()

        self.info["info"] = {
            # "{A}": "red block",
            # "{B}": "green block",
            # "{C}": "blue block",
            # "{a}": arm_tag1,
            # "{b}": arm_tag2,
            # "{c}": arm_tag3,
        }
        return self.info

    def pick_and_place_block(self, block, target_pose=None):
        block_pose = block.get_pose().p
        arm_tag = ArmTag("left" if block_pose[0] < 0 else "right")

        if self.last_gripper is not None and self.last_gripper != arm_tag:
            self.move(
                self.grasp_actor(block, arm_tag=arm_tag, pre_grasp_dis=0.09, grasp_dis=0.01),
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
        block3_pose = self.block3.get_pose().p

        self.task_success[0] = 1
        self.task_success[1] = int(
            block2_pose[0] > block1_pose[0]
            and abs(block2_pose[1] - block1_pose[1]) <= 0.03
        )
        self.task_success[2] = int(
            block3_pose[0] > block2_pose[0]
            and abs(block3_pose[1] - block2_pose[1]) <= 0.03
        )

        self.stage = sum(self.task_success)

    def check_success(self):
        self.update_progress()
        return self.task_success == [1, 1, 1]