from ._base_task import Base_Task
from .utils import *
import numpy as np
import sapien
from ._GLOBAL_CONFIGS import *
from copy import deepcopy


class Hang_Mug_Stack_Blocks_5(Base_Task):
    def setup_demo(self, is_test=False, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        # =========================
        # 1. mug
        # =========================
        self.mug_id = np.random.choice([i for i in range(10)])
        self.task_success = [0, 0, 0, 0, 0]
        self.stage_sum = 5

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
        # 3. 左侧四个 block
        # =========================
        self.block_half_size = 0.015
        block_pose_lst = []

        for i in range(4):
            block_pose = rand_pose(
                xlim=[-0.28, -0.05],
                ylim=[-0.20, 0.2],
                zlim=[0.741 + self.block_half_size + self.table_z_bias],
                qpos=[1, 0, 0, 0],
                ylim_prop=True,
                rotate_rand=True,
                rotate_lim=[0, 0, 0.75],
            )

            def check_block_pose(cur_pose):
                for old_pose in block_pose_lst:
                    if np.sum((cur_pose.p[:2] - old_pose.p[:2]) ** 2) < 0.01:
                        return False
                return True

            # 避开左侧堆叠目标区，避免一开始就刷在目标位置附近
            # while (
            #     np.sum((block_pose.p[:2] - np.array([-0.16, -0.13])) ** 2) < 0.01
            #     or not check_block_pose(block_pose)
            # ):
            #     block_pose = rand_pose(
            #         xlim=[-0.28, -0.05],
            #         ylim=[-0.20, 0.2],
            #         zlim=[0.741 + self.block_half_size + self.table_z_bias],
            #         qpos=[1, 0, 0, 0],
            #         ylim_prop=True,
            #         rotate_rand=True,
            #         rotate_lim=[0, 0, 0.75],
            #     )
            
            max_trials = 100
            trials = 0
            
            while (
                np.sum((block_pose.p[:2] - np.array([-0.16, -0.13])) ** 2) < 0.01
                or not check_block_pose(block_pose)
            ) and trials < max_trials:
                block_pose = rand_pose(
                    xlim=[-0.28, -0.05],
                    ylim=[-0.20, 0.2],
                    zlim=[0.741 + self.block_half_size + self.table_z_bias],
                    qpos=[1, 0, 0, 0],
                    ylim_prop=True,
                    rotate_rand=True,
                    rotate_lim=[0, 0, 0.75],
                )
                trials += 1
            
            if (
                np.sum((block_pose.p[:2] - np.array([-0.16, -0.13])) ** 2) < 0.01
                or not check_block_pose(block_pose)
            ):
                raise RuntimeError(f"Failed to sample a valid block_pose for block {i} within 100 tries.")

            block_pose_lst.append(deepcopy(block_pose))

        def create_block(block_pose, color):
            return create_box(
                scene=self,
                pose=block_pose,
                half_size=(
                    self.block_half_size,
                    self.block_half_size,
                    self.block_half_size,
                ),
                color=color,
                name="box",
            )

        self.block1 = create_block(block_pose_lst[0], (1, 0, 0))
        self.block2 = create_block(block_pose_lst[1], (0, 1, 0))
        self.block3 = create_block(block_pose_lst[2], (0, 0, 1))
        self.block4 = create_block(block_pose_lst[3], (1, 1, 0))

        self.add_prohibit_area(self.block1, padding=0.05)
        self.add_prohibit_area(self.block2, padding=0.05)
        self.add_prohibit_area(self.block3, padding=0.05)
        self.add_prohibit_area(self.block4, padding=0.05)

        # 左侧堆叠目标区
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
        # Part 2: 再堆叠左侧四个 block
        # =========================
        self.last_gripper = mug_arm_tag
        self.last_actor = None

        arm_tag1 = self.pick_and_place_block(self.block1)
        arm_tag2 = self.pick_and_place_block(self.block2)
        arm_tag3 = self.pick_and_place_block(self.block3)
        arm_tag4 = self.pick_and_place_block(self.block4)

        self.move(self.back_to_origin(ArmTag("left")))
        self.move(self.back_to_origin(ArmTag("right")))

        self.info["info"] = {
            "{A}": f"039_mug/base{self.mug_id}",
            "{B}": "040_rack/base0",
            "{C}": "red block",
            "{D}": "green block",
            "{E}": "blue block",
            "{F}": "yellow block",
            "{a}": str(mug_arm_tag),
            "{b}": str(arm_tag1),
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
        # 2. 四个 block 的状态
        # =========================
        block1_pose = self.block1.get_pose().p
        block2_pose = self.block2.get_pose().p
        block3_pose = self.block3.get_pose().p
        block4_pose = self.block4.get_pose().p

        stack_gap = 2 * self.block_half_size
        eps = np.array([0.015, 0.015, 0.007])

        # 第一个块只要求在左侧下方区域
        block1_success = (block1_pose[0] < 0) and (block1_pose[1] < -0.05)

        block2_success = np.all(
            np.abs(
                block2_pose
                - np.array([block1_pose[0], block1_pose[1], block1_pose[2] + stack_gap])
            ) < eps
        )

        block3_success = np.all(
            np.abs(
                block3_pose
                - np.array([block2_pose[0], block2_pose[1], block2_pose[2] + stack_gap])
            ) < eps
        )

        block4_success = np.all(
            np.abs(
                block4_pose
                - np.array([block3_pose[0], block3_pose[1], block3_pose[2] + stack_gap])
            ) < eps
        )

        # =========================
        # 3. 更新 task_success
        # =========================
        self.task_success[0] = int(hang_success)
        self.task_success[1] = int(block1_success)
        self.task_success[2] = int(block2_success)
        self.task_success[3] = int(block3_success)
        self.task_success[4] = int(block4_success)

    def check_success(self):
        self.update_progress()
        return self.task_success == [1, 1, 1, 1, 1]