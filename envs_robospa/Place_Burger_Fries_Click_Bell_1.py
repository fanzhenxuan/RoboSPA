from copy import deepcopy
from ._base_task import Base_Task
from .utils import *
import sapien
import math
import numpy as np
from ._GLOBAL_CONFIGS import *


class Place_Burger_Fries_Click_Bell_1(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        # -------------------------
        # task_success:
        # [0] hamburg placed
        # -------------------------
        self.task_success = [0]
        self.stage_sum = 1

        # -------------------------
        # tray
        # -------------------------
        rand_pos_1 = rand_pose(
            xlim=[-0.0, 0.0],
            ylim=[-0.15, -0.1],
            qpos=[0.706527, 0.706483, -0.0291356, -0.0291767],
            rotate_rand=True,
            rotate_lim=[0, 0, 0],
        )
        self.tray_id = np.random.choice([0, 1, 2, 3], 1)[0]
        self.tray = create_actor(
            scene=self,
            pose=rand_pos_1,
            modelname="008_tray",
            convex=True,
            model_id=self.tray_id,
            scale=(2.0, 2.0, 2.0),
            is_static=True,
        )
        self.tray.set_mass(0.05)

        # -------------------------
        # hamburg
        # -------------------------
        rand_pos_2 = rand_pose(
            xlim=[-0.3, -0.25],
            ylim=[-0.15, -0.07],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=True,
            rotate_lim=[0, 0, 0],
        )
        self.object1_id = np.random.choice([0, 1, 2, 3, 4, 5], 1)[0]
        self.hamburg = create_actor(
            scene=self,
            pose=rand_pos_2,
            modelname="006_hamburg",
            convex=True,
            model_id=self.object1_id,
        )
        self.hamburg.set_mass(0.05)

        # prohibit areas
        self.add_prohibit_area(self.tray, padding=0.1)
        self.add_prohibit_area(self.hamburg, padding=0.05)

    # -------------------------------------------------
    # helper: burger placement success
    # -------------------------------------------------
    def _hamburg_in_place(self):
        dis1 = np.linalg.norm(
            self.tray.get_functional_point(0, "pose").p[0:2]
            - self.hamburg.get_functional_point(0, "pose").p[0:2]
        )
        return dis1 < 0.08

    # -------------------------------------------------
    # progress update
    # -------------------------------------------------
    def update_progress(self):
        self.task_success[0] = int(self._hamburg_in_place())

    # -------------------------------------------------
    # action blocks
    # -------------------------------------------------
    def _do_place_hamburg(self):
        arm_tag_left = ArmTag("left")
        tray_place_pose_left = self.tray.get_functional_point(0)

        self.move(
            self.grasp_actor(
                self.hamburg,
                arm_tag=arm_tag_left,
                pre_grasp_dis=0.1,
            )
        )

        self.move(self.move_by_displacement(arm_tag=arm_tag_left, z=0.1))

        self.move(
            self.place_actor(
                self.hamburg,
                arm_tag=arm_tag_left,
                target_pose=tray_place_pose_left,
                functional_point_id=0,
                constrain="free",
                pre_dis=0.1,
                pre_dis_axis="fp",
            )
        )

        self.move(self.move_by_displacement(arm_tag=arm_tag_left, z=0.08))
        self.update_progress()

    def play_once(self):
        arm_tag_left = ArmTag("left")

        self._do_place_hamburg()
        self.update_progress()

        self.info["info"] = {
            "{A}": f"006_hamburg/base{self.object1_id}",
            "{B}": f"008_tray/base{self.tray_id}",
            "{a}": str(arm_tag_left),
        }
        return self.info

    def check_success(self):
        self.update_progress()
        return self.task_success == [1]