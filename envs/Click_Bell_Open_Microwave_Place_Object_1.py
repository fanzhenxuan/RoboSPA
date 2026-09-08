import numpy as np
import sapien

from ._base_task import Base_Task
from .utils import *
from ._GLOBAL_CONFIGS import *
from copy import deepcopy


class Click_Bell_Open_Microwave_Place_Object_1(Base_Task):

    def setup_demo(self, is_test=False, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.task_success = [0]
        self.click_stage_sum = 1
        self.stage_sum = 1

        self.stage = 0
        self.has_left_press_area = True

        bell_pose = rand_pose(
            xlim=[-0.25, -0.15],
            ylim=[-0.20, -0.15],
            qpos=[0.5, 0.5, 0.5, 0.5],
        )
        while abs(bell_pose.p[0]) < 0.05:
            bell_pose = rand_pose(
                xlim=[-0.25, -0.15],
                ylim=[-0.20, -0.15],
                qpos=[0.5, 0.5, 0.5, 0.5],
            )

        self.bell_id = np.random.choice([0, 1], 1)[0]
        self.bell = create_actor(
            scene=self,
            pose=bell_pose,
            modelname="050_bell",
            convex=True,
            model_id=self.bell_id,
            is_static=True,
        )

        self.add_prohibit_area(self.bell, padding=0.07)
        self.check_arm_function = (
            self.is_left_gripper_close
            if self.bell.get_pose().p[0] < 0
            else self.is_right_gripper_close
        )

    def _is_click_success(self):
        if not self.check_arm_function():
            return False

        bell_pose = self.bell.get_contact_point(0)[:3]
        positions = self.get_gripper_actor_contact_position("050_bell")
        eps = [0.025, 0.025]

        for position in positions:
            if (
                np.all(np.abs(position[:2] - bell_pose[:2]) < eps)
                and abs(position[2] - bell_pose[2]) < 0.03
            ):
                return True
        return False

    def _has_left_click_area(self):
        bell_pose = self.bell.get_contact_point(0)[:3]
        positions = self.get_gripper_actor_contact_position("050_bell")

        if len(positions) == 0:
            return True

        for position in positions:
            if (
                np.all(np.abs(position[:2] - bell_pose[:2]) < np.array([0.025, 0.025]))
                and abs(position[2] - bell_pose[2]) < 0.03
            ):
                return False

        return True

    def update_progress(self):
        if self.stage < self.click_stage_sum:
            if self._has_left_click_area():
                self.has_left_press_area = True

            if self.has_left_press_area and self._is_click_success():
                self.stage += 1
                self.has_left_press_area = False

        self.task_success[0] = int(self.stage >= 1)

    def _do_one_click(self, arm_tag):
        self.move(
            self.grasp_actor(
                self.bell,
                arm_tag=arm_tag,
                pre_grasp_dis=0.1,
                grasp_dis=0.1,
                contact_point_id=0,
            )
        )

        self.move(self.move_by_displacement(arm_tag, z=-0.045))
        self.update_progress()

        self.move(self.move_by_displacement(arm_tag, z=0.045))
        self.update_progress()

    def play_once(self):
        self.last_gripper = None

        bell_arm = ArmTag("right" if self.bell.get_pose().p[0] > 0 else "left")
        self.move(self.close_gripper(arm_tag=bell_arm, pos=0))
        self._do_one_click(bell_arm)
        self.last_gripper = bell_arm

        self.info["info"] = {
            "{C}": f"050_bell/base{self.bell_id}",
            "{a}": str(bell_arm),
        }

        return self.info

    def check_success(self):
        self.update_progress()
        return self.task_success == [1]
