from ._base_task import Base_Task
from .utils import *
import sapien
import math
import numpy as np
from copy import deepcopy


class Stack_Blocks_Size_Order_1(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.stage_sum = 1
        self.stage = 0
        self.task_success = [0]

        # while True:
        #     block_pose = rand_pose(
        #         xlim=[-0.28, 0.28],
        #         ylim=[-0.08, 0.05],
        #         zlim=[0.765],
        #         qpos=[1, 0, 0, 0],
        #         ylim_prop=True,
        #         rotate_rand=True,
        #         rotate_lim=[0, 0, 0.75],
        #     )
        #
        #     if (
        #         abs(block_pose.p[0]) < 0.05
        #         or np.sum((block_pose.p[:2] - np.array([0, -0.1])) ** 2) < 0.01
        #     ):
        #         continue
        #     else:
        #         break
        
        max_trials = 100
        trials = 0
        
        while trials < max_trials:
            block_pose = rand_pose(
                xlim=[-0.28, 0.28],
                ylim=[-0.08, 0.05],
                zlim=[0.765],
                qpos=[1, 0, 0, 0],
                ylim_prop=True,
                rotate_rand=True,
                rotate_lim=[0, 0, 0.75],
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

        size = np.random.uniform(0.015, 0.025)
        half_size = (size, size, size)

        color = (
            np.random.random(),
            np.random.random(),
            np.random.random(),
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
            0.74 + self.table_z_bias,
        ] + [0, 1, 0, 0]

    def play_once(self):
        self.last_gripper = None
        self.stage = 0

        arm_tag1 = self.pick_and_place_block(self.block1, self.block1_target_pose)
        self.update_progress()

        self.info["info"] = {
            # "{A}": "red block",
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
        if block1_pose[1]<-0.08 and abs(block1_pose[0])<0.05:
            self.task_success[0] = 1

    def check_success(self):
        self.update_progress()
        return self.task_success[0] == 1