from ._base_task import Base_Task
from .utils import *
import sapien
import math
from copy import deepcopy
import numpy as np



class Click_Can_Place_Items_1(Base_Task):

    def setup_demo(self, **kwargs):
        super()._init_task_env_(**kwargs)

    def _load_can(self):
        can_dict = {"071_can": [0, 1, 2, 3, 5, 6]}
        self.can_name = "071_can"
        self.can_id = np.random.choice(can_dict[self.can_name])

        can_pose = rand_pose(
            xlim=[-0.25, -0.05],
            ylim=[-0.2, 0.1],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=False,
        )

        self.can = create_actor(
            scene=self,
            pose=can_pose,
            modelname=self.can_name,
            convex=True,
            model_id=self.can_id,
            is_static=True,
        )
        self.can.set_mass(0.01)


    def _is_can_click_success(self):
        can_pose = self.can.get_contact_point(8)[:3]
        positions = self.get_gripper_actor_contact_position("071_can")
        eps_xy = np.array([0.03, 0.03])

        for position in positions:
            if (
                np.all(np.abs(position[:2] - can_pose[:2]) < eps_xy)
                and abs(position[2] - can_pose[2]) < 0.04
            ):
                return True
        return False

    def _has_left_can_click_area(self):
        can_pose = self.can.get_contact_point(8)[:3]
        positions = self.get_gripper_actor_contact_position("071_can")

        if len(positions) == 0:
            return True

        for position in positions:
            if (
                np.all(np.abs(position[:2] - can_pose[:2]) < np.array([0.03, 0.03]))
                and abs(position[2] - can_pose[2]) < 0.04
            ):
                return False

        return True

    def _update_click_progress(self):
        if self.stage < self.click_stage_sum:
            if self._has_left_can_click_area():
                self.has_left_click_area = True

            if self.has_left_click_area and self._is_can_click_success():
                self.stage += 1
                self.has_left_click_area = False

    def _do_one_click(self, can_arm):
        can_hover_action = lambda: self.grasp_actor(
            self.can,
            arm_tag=can_arm,
            pre_grasp_dis=0.08,
            grasp_dis=0.08,
            contact_point_id=8,
        )

        can_press_action = lambda: self.grasp_actor(
            self.can,
            arm_tag=can_arm,
            pre_grasp_dis=0.02,
            grasp_dis=0.02,
            contact_point_id=8,
        )

        self.move(can_hover_action())
        self.update_progress()

        self.move(self.close_gripper(arm_tag=can_arm))
        self.move(can_press_action())
        self.update_progress()

        self.move(self.move_by_displacement(arm_tag=can_arm, z=0.05))
        self.update_progress()

    def load_actors(self):
        self._load_can()
        self.click_stage_sum = 1
        self.stage_sum = 1
        self.stage = 0
        self.has_left_click_area = True
        self.task_success = [0]

    def update_progress(self):
        self._update_click_progress()
        self.task_success[0] = int(self.stage >= 1)

    def play_once(self):
        can_arm = ArmTag("left" if self.can.get_pose().p[0] < 0 else "right")
        self._do_one_click(can_arm)

        self.info["info"] = {
            "{A}": f"{self.can_name}/base{self.can_id}",
            "{a}": str(can_arm),
        }

        return self.info

    def check_success(self):
        self.update_progress()
        return self.task_success == [1]
