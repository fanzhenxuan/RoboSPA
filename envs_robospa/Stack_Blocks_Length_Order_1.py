from ._base_task import Base_Task
from .utils import *
import sapien
import math
import numpy as np
from copy import deepcopy


class Stack_Blocks_Length_Order_1(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.stage_sum = 1
        self.stage = 0
        self.task_success = [0]

        max_trials = 100
        trials = 0

        while trials < max_trials:
            block_pose = rand_pose(
                xlim=[-0.28, 0.28],
                ylim=[-0.08, 0.05],
                zlim=[0.741],
                qpos=[1, 0, 0, 0],
                ylim_prop=True,
                rotate_rand=True,
                rotate_lim=[0, 0, 0.20],
            )

            if (
                abs(block_pose.p[0]) < 0.05
                or np.sum((block_pose.p[:2] - np.array([0, -0.1])) ** 2) < 0.01
            ):
                trials += 1
                continue
            else:
                break

        if (
            abs(block_pose.p[0]) < 0.05
            or np.sum((block_pose.p[:2] - np.array([0, -0.1])) ** 2) < 0.01
        ):
            raise RuntimeError("Failed to sample a valid block_pose within 100 tries.")

        # 按长度任务：x方向变化，y/z固定
        half_x = np.random.uniform(0.0240, 0.0250)
        half_y = np.random.uniform(0.0155, 0.0165)
        half_z = np.random.uniform(0.0155, 0.0165)
        half_size = (half_x, half_y, half_z)

        color = (
            np.random.random(),
            np.random.random(),
            np.random.random(),
        )

        # 修正初始z，使方块底面落在桌面上
        block_pose = sapien.Pose(
            p=[block_pose.p[0], block_pose.p[1], 0.741 + half_z],
            q=block_pose.q
        )

        self.block1 = create_box(
            scene=self,
            pose=deepcopy(block_pose),
            half_size=half_size,
            color=color,
            name="box",
        )

        self.add_prohibit_area(self.block1, padding=0.05)
        self.prohibited_area.append([-0.17, -0.22, 0.17, -0.12])

        y_pose = np.random.uniform(-0.2, -0.1)
        self.block1_target_pose = [
            np.random.uniform(-0.01, 0.01),
            y_pose,
            0.74 + half_z + self.table_z_bias,
        ] + [0, 1, 0, 0]

    def play_once(self):
        self.last_gripper = None
        self.stage = 0

        arm_tag1 = self.pick_and_place_block(self.block1, self.block1_target_pose)
        self.update_progress()

        self.info["info"] = {
            # "{A}": "long block",
            # "{a}": arm_tag1,
        }
        return self.info

    def pick_and_place_block(self, block, target_pose=None):
        block_pose = block.get_pose().p
        arm_tag = ArmTag("left" if block_pose[0] < 0 else "right")

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
        if block1_pose[1] < -0.08 and abs(block1_pose[0]) < 0.05:
            self.task_success[0] = 1

    def check_success(self):
        self.update_progress()
        return (self.task_success[0] == 1 and self.robot.is_left_gripper_open() and self.robot.is_right_gripper_open())