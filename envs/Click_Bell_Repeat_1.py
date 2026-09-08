from copy import deepcopy
from ._base_task import Base_Task
from .utils import *
import sapien
import math
import numpy as np


class Click_Bell_Repeat_1(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.task_success = [0]

        # How many bell presses are required in total
        self.stage_sum = 1
        self.stage = 0

        # Whether it has already left the press region
        # Avoid counting the same contact twice
        self.has_left_press_area = True

        rand_pos = rand_pose(
            xlim=[-0.25, 0.25],
            ylim=[-0.2, 0.0],
            qpos=[0.5, 0.5, 0.5, 0.5],
        )
        max_trials = 100
        trials = 0
        
        while abs(rand_pos.p[0]) < 0.05 and trials < max_trials:
            rand_pos = rand_pose(
                xlim=[-0.25, 0.25],
                ylim=[-0.2, 0.0],
                qpos=[0.5, 0.5, 0.5, 0.5],
            )
            trials += 1
        
        if abs(rand_pos.p[0]) < 0.05:
            raise RuntimeError("Failed to sample a valid rand_pos within 100 tries.")

        self.bell_id = np.random.choice([0, 1], 1)[0]
        self.bell = create_actor(
            scene=self,
            pose=rand_pos,
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
        """
        Check whether this is a valid bell press:
        1. The active gripper is closed
        2. The gripper is close enough to the bell-top contact
        """
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
        """
        Check whether the gripper has left the bell-top press region.
        Prevents one contact from being counted as multiple clicks.
        """
        bell_pose = self.bell.get_contact_point(0)[:3]
        positions = self.get_gripper_actor_contact_position("050_bell")

        # No contacts means it has already left
        if len(positions) == 0:
            return True

        # If a contact still exists near the bell top, it has not left yet
        for position in positions:
            if (
                np.all(np.abs(position[:2] - bell_pose[:2]) < np.array([0.025, 0.025]))
                and abs(position[2] - bell_pose[2]) < 0.03
            ):
                return False

        return True

    def update_progress(self):
        """
        Stage-advance logic:
        - A new click counts only after leaving the press region and then pressing successfully again
        """
        if self.stage >= self.stage_sum:
            return

        # First check whether it has already left the press region
        if self._has_left_click_area():
            self.has_left_press_area = True

        # If it has already left and this press succeeds, increment the count
        if self.has_left_press_area and self._is_click_success():
            self.stage += 1
            # Must leave again before the next click can count
            self.has_left_press_area = False

    def _do_one_click(self, arm_tag):
        """
        Run one full click:
        go above -> press down -> check count -> lift up -> update leave state
        """
        # Move above the bell
        self.move(
            self.grasp_actor(
                self.bell,
                arm_tag=arm_tag,
                pre_grasp_dis=0.1,
                grasp_dis=0.1,
                contact_point_id=0,
            )
        )

        # Press down
        self.move(self.move_by_displacement(arm_tag, z=-0.045))
        self.update_progress()

        # Lift
        self.move(self.move_by_displacement(arm_tag, z=0.045))
        self.update_progress()

    def play_once(self):
        # Use the right hand if the bell is on the right, else the left
        arm_tag = ArmTag("right" if self.bell.get_pose().p[0] > 0 else "left")

        # Close the gripper first to mimic pressing the bell
        self.move(self.close_gripper(arm_tag=arm_tag, pos=0))

        # Perform 1 click
        for _ in range(self.stage_sum):
            self._do_one_click(arm_tag)

        self.info["info"] = {
            "{A}": f"050_bell/base{self.bell_id}",
            # "{B}": str(self.stage_sum),
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        for i in range(self.stage_sum):
            self.task_success[i] = int(self.stage >= i + 1)
        return self.stage >= self.stage_sum