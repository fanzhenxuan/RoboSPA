from copy import deepcopy
from ._base_task import Base_Task
from .utils import *
import sapien
import numpy as np
import math


class Click_Objects_Order_1(Base_Task):
    """
    Stage 0: click (press) the alarm clock top button
    Success: stage >= 1
    """

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        # ====== stage stats ======
        self.stage_sum = 1
        self.stage = 0
        self.task_success = [0]

        # ====== sample poses (keep away from x≈0) ======
        def sample_pose(xlim=[-0.25, 0.25], ylim=[-0.2, 0.0], rotate=False):
            return rand_pose(
                xlim=xlim,
                ylim=ylim,
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=rotate,
                rotate_lim=[0, 3.14, 0] if rotate else None,
            )

        # Alarm clock pose
        alarm_pose = sample_pose(rotate=True)
        # while abs(alarm_pose.p[0]) < 0.05:
        #     alarm_pose = sample_pose(rotate=True)
        
        max_trials = 100
        trials = 0
        
        while abs(alarm_pose.p[0]) < 0.05 and trials < max_trials:
            alarm_pose = sample_pose(rotate=True)
            trials += 1
        
        if abs(alarm_pose.p[0]) < 0.05:
            raise RuntimeError("Failed to sample a valid alarm_pose within 100 tries.")

        # ====== create actor ======
        self.alarmclock_id = np.random.choice([1, 3], 1)[0]
        self.alarm = create_actor(
            scene=self,
            pose=alarm_pose,
            modelname="046_alarm-clock",
            convex=True,
            model_id=self.alarmclock_id,
            is_static=True,
        )

        # ====== prohibit area ======
        self.add_prohibit_area(self.alarm, padding=0.05)

        # ====== per-stage flags ======
        self._clicked_alarm = False

    # -------------------------
    # success checking helpers
    # -------------------------
    def _check_clicked_alarm(self, arm_tag: ArmTag):
        check_arm_fn = self.is_left_gripper_close if str(arm_tag) == "left" else self.is_right_gripper_close
        if not check_arm_fn():
            return False

        alarm_pose = self.alarm.get_contact_point(0)[:3]
        positions = self.get_gripper_actor_contact_position("046_alarm-clock")
        eps = [0.03, 0.03]
        for position in positions:
            if (
                np.all(np.abs(position[:2] - alarm_pose[:2]) < eps)
                and abs(position[2] - alarm_pose[2]) < 0.03
            ):
                return True
        return False

    # -------------------------
    # stage progress
    # -------------------------
    def update_progress(self):
        if self.stage == 0:
            arm_tag = ArmTag("right" if self.alarm.get_pose().p[0] > 0 else "left")
            if self._check_clicked_alarm(arm_tag):
                self._clicked_alarm = True
                self.stage = 1

    # -------------------------
    # main episode
    # -------------------------
    def play_once(self):
        # ====== Stage 0: click alarm clock ======
        alarm_arm = ArmTag("right" if self.alarm.get_pose().p[0] > 0 else "left")

        # self.move(
        #     self.grasp_actor(
        #         self.alarm,
        #         arm_tag=alarm_arm,
        #         pre_grasp_dis=0.1,
        #         grasp_dis=0.1,
        #         contact_point_id=0,
        #     )
        # )

        self.move((
            ArmTag(alarm_arm),
            [
                Action(
                    alarm_arm,
                    "move",
                    self.get_grasp_pose(self.alarm, pre_dis=0.1, contact_point_id=0, arm_tag=alarm_arm)[:3] +
                    [0.5, -0.5, 0.5, 0.5],
                ),
                Action(alarm_arm, "close", target_gripper_pos=0.0),
            ],
        ))


        # press down
        self.move(self.move_by_displacement(alarm_arm, z=-0.065))
        self.update_progress()

        # move back up
        self.move(self.move_by_displacement(alarm_arm, z=0.065))
        self.update_progress()

        # ====== info ======
        self.info["info"] = {
            "{A}": f"046_alarm-clock/base{self.alarmclock_id}",
            "{a}": str(alarm_arm),
        }
        return self.info

    def check_success(self):
        for i in range(self.stage_sum):
            self.task_success[i] = int(self.stage >= i + 1)
        return self.stage >= self.stage_sum