from ._base_task import Base_Task
from .utils import *
import sapien
import math
import numpy as np
from copy import deepcopy


class Rank_Blocks_Size_5(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.stage_sum = 5
        self.stage = 0
        self.task_success = [0, 0, 0, 0, 0]

        # 所有 block 使用同一个随机颜色
        color = (
            np.random.random(),
            np.random.random(),
            np.random.random(),
        )

        # 从大到小
        halfsize_lst = [
            np.random.uniform(0.032, 0.034),  # largest
            np.random.uniform(0.027, 0.029),  # large
            np.random.uniform(0.022, 0.024),  # medium
            np.random.uniform(0.017, 0.019),  # small
            np.random.uniform(0.012, 0.014),  # extra small
        ]

        while True:
            block_pose_lst = []
            for i in range(5):
                block_pose = rand_pose(
                    xlim=[-0.28, 0.28],
                    ylim=[-0.08, 0.05],
                    zlim=[0.741 + halfsize_lst[i]],
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

                while (
                    abs(block_pose.p[0]) < 0.05
                    or np.sum((block_pose.p[:2] - np.array([0, -0.1])) ** 2) < 0.01
                    or not check_block_pose(block_pose)
                ):
                    block_pose = rand_pose(
                        xlim=[-0.28, 0.28],
                        ylim=[-0.08, 0.05],
                        zlim=[0.741 + halfsize_lst[i]],
                        qpos=[1, 0, 0, 0],
                        ylim_prop=True,
                        rotate_rand=True,
                        rotate_lim=[0, 0, 0.75],
                    )
                block_pose_lst.append(deepcopy(block_pose))

            eps = [0.12, 0.03]
            block1_pose = block_pose_lst[0].p
            block2_pose = block_pose_lst[1].p
            block3_pose = block_pose_lst[2].p
            block4_pose = block_pose_lst[3].p
            block5_pose = block_pose_lst[4].p

            # 避免初始状态已经接近满足目标排序：最大 < 大 < 中 < 小 < 最小（从左到右）
            # 避免初始状态已经接近满足目标排序：最大 < 大 < 中 < 小 < 最小（从左到右）
            aligned_close = (
                np.all(abs(block1_pose[:2] - block2_pose[:2]) < eps)
                and np.all(abs(block2_pose[:2] - block3_pose[:2]) < eps)
                and np.all(abs(block3_pose[:2] - block4_pose[:2]) < eps)
                and np.all(abs(block4_pose[:2] - block5_pose[:2]) < eps)
            )

            ordered_x = (
                block1_pose[0] < block2_pose[0] < block3_pose[0] < block4_pose[0] < block5_pose[0]
            )

            if aligned_close or ordered_x:
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

        self.block1 = create_block(block_pose_lst[0], halfsize_lst[0], color)  # largest
        self.block2 = create_block(block_pose_lst[1], halfsize_lst[1], color)  # large
        self.block3 = create_block(block_pose_lst[2], halfsize_lst[2], color)  # medium
        self.block4 = create_block(block_pose_lst[3], halfsize_lst[3], color)  # small
        self.block5 = create_block(block_pose_lst[4], halfsize_lst[4], color)  # extra small

        self.add_prohibit_area(self.block1, padding=0.1)
        self.add_prohibit_area(self.block2, padding=0.1)
        self.add_prohibit_area(self.block3, padding=0.1)
        self.add_prohibit_area(self.block4, padding=0.1)
        self.add_prohibit_area(self.block5, padding=0.1)
        self.prohibited_area.append([-0.27, -0.22, 0.27, -0.12])

        # 仅用于放置动作
        y_pose = np.random.uniform(-0.2, -0.1)

        # 从左到右：最大 大 中 小 最小
        self.block1_target_pose = [
            np.random.uniform(-0.18, -0.16),
            y_pose,
            0.74 + self.table_z_bias,
        ] + [0, 1, 0, 0]

        self.block2_target_pose = [
            np.random.uniform(-0.10, -0.08),
            y_pose,
            0.74 + self.table_z_bias,
        ] + [0, 1, 0, 0]

        self.block3_target_pose = [
            np.random.uniform(-0.01, 0.01),
            y_pose,
            0.74 + self.table_z_bias,
        ] + [0, 1, 0, 0]

        self.block4_target_pose = [
            np.random.uniform(0.08, 0.10),
            y_pose,
            0.74 + self.table_z_bias,
        ] + [0, 1, 0, 0]

        self.block5_target_pose = [
            np.random.uniform(0.16, 0.18),
            y_pose,
            0.74 + self.table_z_bias,
        ] + [0, 1, 0, 0]

    def play_once(self):
        self.last_gripper = None
        self.stage = 0
        self.task_success = [0, 0, 0, 0, 0]

        arm_tag1 = self.pick_and_place_block(self.block1, self.block1_target_pose)
        self.update_progress()

        arm_tag2 = self.pick_and_place_block(self.block2, self.block2_target_pose)
        self.update_progress()

        arm_tag3 = self.pick_and_place_block(self.block3, self.block3_target_pose)
        self.update_progress()

        arm_tag4 = self.pick_and_place_block(self.block4, self.block4_target_pose)
        self.update_progress()

        arm_tag5 = self.pick_and_place_block(self.block5, self.block5_target_pose)
        self.update_progress()

        self.info["info"] = {
            # "{A}": "largest block",
            # "{B}": "large block",
            # "{C}": "medium block",
            # "{D}": "small block",
            # "{E}": "extra small block",
            # "{a}": arm_tag1,
            # "{b}": arm_tag2,
            # "{c}": arm_tag3,
            # "{d}": arm_tag4,
            # "{e}": arm_tag5,
        }
        return self.info

    def pick_and_place_block(self, block, target_pose=None):
        block_pose = block.get_pose().p
        arm_tag = ArmTag("left" if block_pose[0] < 0 else "right")

        if self.last_gripper is not None and self.last_gripper != arm_tag:
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
        block3_pose = self.block3.get_pose().p
        block4_pose = self.block4.get_pose().p
        block5_pose = self.block5.get_pose().p

        self.task_success[0] = 1
        self.task_success[1] = int(
            block2_pose[0] > block1_pose[0]
            and abs(block2_pose[1] - block1_pose[1]) <= 0.03
        )
        self.task_success[2] = int(
            block3_pose[0] > block2_pose[0]
            and abs(block3_pose[1] - block2_pose[1]) <= 0.03
        )
        self.task_success[3] = int(
            block4_pose[0] > block3_pose[0]
            and abs(block4_pose[1] - block3_pose[1]) <= 0.03
        )
        self.task_success[4] = int(
            block5_pose[0] > block4_pose[0]
            and abs(block5_pose[1] - block4_pose[1]) <= 0.03
        )

        self.stage = sum(self.task_success)

    def check_success(self):
        self.update_progress()
        return (
            self.task_success == [1, 1, 1, 1, 1]
            and self.is_left_gripper_open()
            and self.is_right_gripper_open()
        )