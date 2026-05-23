from ._base_task import Base_Task
from .utils import *
import numpy as np
import sapien
from ._GLOBAL_CONFIGS import *
from copy import deepcopy


class Hang_Mug_Stack_Blocks_2(Base_Task):
    def setup_demo(self, is_test=False, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        # =========================
        # 1. mug
        # =========================
        self.mug_id = np.random.choice([i for i in range(10)])
        self.task_success = [0, 0]
        self.stage_sum = 2

        mug_x = np.random.uniform(0.05, 0.2)
        mug_y = np.random.uniform(-0.2, -0.05)
        mug_z = 0.75
        mug_q = [1, 0, 0, 0]

        self.middle_pos = [mug_x, mug_y, mug_z, *mug_q]

        self.mug = rand_create_actor(
            self,
            xlim=[mug_x, mug_x],
            ylim=[mug_y, mug_y],
            ylim_prop=True,
            modelname="039_mug",
            rotate_rand=True,
            rotate_lim=[0, 1.57, 0],
            qpos=[0.707, 0.707, 0, 0],
            convex=True,
            model_id=self.mug_id,
        )

        # =========================
        # 2. rack
        # =========================
        rack_pose = rand_pose(
            xlim=[0.1, 0.3],
            ylim=[0.13, 0.17],
            rotate_rand=True,
            rotate_lim=[0, 0.2, 0],
            qpos=[-0.22, -0.22, 0.67, 0.67],
        )
        self.rack = create_actor(
            self,
            pose=rack_pose,
            modelname="040_rack",
            is_static=True,
            convex=True,
        )

        self.add_prohibit_area(self.mug, padding=0.1)
        self.add_prohibit_area(self.rack, padding=0.1)

        # =========================
        # 3. 左侧一个 block
        # =========================
        self.block_half_size = 0.015

        block_pose = rand_pose(
            xlim=[-0.28, -0.05],
            ylim=[0, 0.2],
            zlim=[0.741 + self.block_half_size + self.table_z_bias],
            qpos=[1, 0, 0, 0],
            ylim_prop=True,
            rotate_rand=True,
            rotate_lim=[0, 0, 0.75],
        )

        # while np.sum((block_pose.p[:2] - np.array([-0.16, -0.13])) ** 2) < 0.01:
        #     block_pose = rand_pose(
        #         xlim=[-0.28, -0.05],
        #         ylim=[0, 0.2],
        #         zlim=[0.741 + self.block_half_size + self.table_z_bias],
        #         qpos=[1, 0, 0, 0],
        #         ylim_prop=True,
        #         rotate_rand=True,
        #         rotate_lim=[0, 0, 0.75],
        #     )
        
        max_trials = 100
        trials = 0
        
        while np.sum((block_pose.p[:2] - np.array([-0.16, -0.13])) ** 2) < 0.01 and trials < max_trials:
            block_pose = rand_pose(
                xlim=[-0.28, -0.05],
                ylim=[0, 0.2],
                zlim=[0.741 + self.block_half_size + self.table_z_bias],
                qpos=[1, 0, 0, 0],
                ylim_prop=True,
                rotate_rand=True,
                rotate_lim=[0, 0, 0.75],
            )
            trials += 1
        
        if np.sum((block_pose.p[:2] - np.array([-0.16, -0.13])) ** 2) < 0.01:
            raise RuntimeError("Failed to sample a valid block_pose within 100 tries.")

        self.block1 = create_box(
            scene=self,
            pose=deepcopy(block_pose),
            half_size=(
                self.block_half_size,
                self.block_half_size,
                self.block_half_size,
            ),
            color=(1, 0, 0),
            name="box",
        )

        self.add_prohibit_area(self.block1, padding=0.05)

        # 左侧目标区
        self.prohibited_area.append([-0.21, -0.18, -0.11, -0.08])
        self.block1_target_pose = [
            -0.16,
            -0.13,
            0.741 + self.block_half_size + self.table_z_bias,
            0,
            1,
            0,
            0,
        ]

    def play_once(self):
        # =========================
        # Part 1: 先挂 mug
        # =========================
        mug_arm_tag = ArmTag("right")

        self.move(self.grasp_actor(self.mug, arm_tag=mug_arm_tag, pre_grasp_dis=0.05))
        self.move(
            self.move_by_displacement(
                arm_tag=mug_arm_tag,
                z=0.1,
                quat=GRASP_DIRECTION_DIC["front"],
            )
        )

        target_pose = self.rack.get_functional_point(0)
        self.move(
            self.place_actor(
                self.mug,
                arm_tag=mug_arm_tag,
                target_pose=target_pose,
                functional_point_id=0,
                constrain="align",
                pre_dis=0.05,
                dis=-0.05,
                pre_dis_axis="fp",
            )
        )
        self.move(self.move_by_displacement(arm_tag=mug_arm_tag, z=0.1, move_axis="arm"))
        # self.move(self.back_to_origin(mug_arm_tag))

        # =========================
        # Part 2: 再放左侧一个 block
        # =========================
        self.last_gripper = mug_arm_tag
        self.last_actor = None
        
        block_arm_tag = self.pick_and_place_block(self.block1)

        self.move(self.back_to_origin(ArmTag("left")))
        self.move(self.back_to_origin(ArmTag("right")))

        self.info["info"] = {
            "{A}": f"039_mug/base{self.mug_id}",
            "{B}": "040_rack/base0",
            "{C}": "red block",
            "{a}": str(mug_arm_tag),
            "{b}": str(block_arm_tag),
        }
        return self.info

    def pick_and_place_block(self, block: Actor):
        block_pose = block.get_pose().p
        arm_tag = ArmTag("left" if block_pose[0] < 0 else "right")

        if self.last_gripper is not None and (self.last_gripper != arm_tag):
            self.move(
                self.grasp_actor(block, arm_tag=arm_tag, pre_grasp_dis=0.09),
                self.back_to_origin(arm_tag=arm_tag.opposite),
            )
        else:
            self.move(self.grasp_actor(block, arm_tag=arm_tag, pre_grasp_dis=0.09))

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07, move_axis="arm"))

        if self.last_actor is None:
            target_pose = self.block1_target_pose
        else:
            target_pose = self.last_actor.get_functional_point(1)

        self.move(
            self.place_actor(
                block,
                target_pose=target_pose,
                arm_tag=arm_tag,
                functional_point_id=0,
                pre_dis=0.05,
                dis=0.0,
                pre_dis_axis="fp",
            )
        )
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07, move_axis="arm"))

        self.last_gripper = arm_tag
        self.last_actor = block
        return str(arm_tag)

    def update_progress(self):
        # =========================
        # 1. mug 是否挂成功
        # =========================
        mug_function_pose = self.mug.get_functional_point(0)[:3]
        rack_pose = self.rack.get_pose().p
        rack_function_pose = self.rack.get_functional_point(0)[:3]
        rack_middle_pose = (rack_pose + rack_function_pose) / 2

        mug_eps = 0.02
        hang_success = (
            np.all(np.abs((mug_function_pose - rack_middle_pose)[:2]) < mug_eps)
            and mug_function_pose[2] > 0.86
        )

        # =========================
        # 2. block1 是否到左下区域
        # =========================
        block1_pose = self.block1.get_pose().p
        block1_success = (block1_pose[0] < -0.05) and (block1_pose[1] < -0.1)

        # =========================
        # 3. 更新 task_success
        # =========================
        self.task_success[0] = int(hang_success)
        self.task_success[1] = int(block1_success)

    def check_success(self):
        self.update_progress()
        return self.task_success == [1, 1]