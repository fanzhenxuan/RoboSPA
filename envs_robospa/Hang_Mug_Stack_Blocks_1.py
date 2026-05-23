from ._base_task import Base_Task
from .utils import *
import numpy as np
import sapien
from ._GLOBAL_CONFIGS import *


class Hang_Mug_Stack_Blocks_1(Base_Task):
    def setup_demo(self, is_test=False, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        # =========================
        # 1. mug
        # =========================
        self.mug_id = np.random.choice([i for i in range(10)])
        self.task_success = [0]
        self.stage_sum = 1

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

    def play_once(self):
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

        self.info["info"] = {
            "{A}": f"039_mug/base{self.mug_id}",
            "{B}": "040_rack/base0",
            "{a}": str(mug_arm_tag),
        }
        return self.info

    def update_progress(self):
        mug_function_pose = self.mug.get_functional_point(0)[:3]
        rack_pose = self.rack.get_pose().p
        rack_function_pose = self.rack.get_functional_point(0)[:3]
        rack_middle_pose = (rack_pose + rack_function_pose) / 2

        mug_eps = 0.02
        hang_success = (
            np.all(np.abs((mug_function_pose - rack_middle_pose)[:2]) < mug_eps)
            and mug_function_pose[2] > 0.86
        )

        self.task_success[0] = int(hang_success)

    def check_success(self):
        self.update_progress()
        return self.task_success == [1]