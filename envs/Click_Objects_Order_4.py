from copy import deepcopy
from ._base_task import Base_Task
from .utils import *
import sapien
import numpy as np
import math


class Click_Objects_Order_4(Base_Task):
    """
    Stage 0: click (press) the alarm clock top button
    Stage 1: click (press) the stapler
    Stage 2: click (press) the can (top contact point)
    Stage 3: click (press) the playing cards
    Success: stage >= 4
    """

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        # ====== stage stats ======
        self.stage_sum = 4
        self.stage = 0
        self.task_success = [0, 0, 0, 0]

        # ====== helpers ======
        def sample_pose(xlim=[-0.25, 0.25], ylim=[-0.2, 0.0], rotate=False):
            return rand_pose(
                xlim=xlim,
                ylim=ylim,
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=rotate,
                rotate_lim=[0, 3.14, 0] if rotate else None,
            )

        def far_enough(p0, p1, min_dist=0.14):
            return np.linalg.norm(p0.p[:2] - p1.p[:2]) >= min_dist

        # ====== alarm clock pose ======
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

        # ====== stapler pose ======
        stapler_pose = sample_pose(rotate=True)
        # while abs(stapler_pose.p[0]) < 0.05 or (not far_enough(stapler_pose, alarm_pose, min_dist=0.14)):
        #     stapler_pose = sample_pose(rotate=True)
        
        max_trials = 100
        trials = 0
        
        while (
            abs(stapler_pose.p[0]) < 0.05
            or (not far_enough(stapler_pose, alarm_pose, min_dist=0.14))
        ) and trials < max_trials:
            stapler_pose = sample_pose(rotate=True)
            trials += 1
        
        if (
            abs(stapler_pose.p[0]) < 0.05
            or (not far_enough(stapler_pose, alarm_pose, min_dist=0.14))
        ):
            raise RuntimeError("Failed to sample a valid stapler_pose within 100 tries.")

        # ====== can pose ======
        can_pose = sample_pose(rotate=False)
        # while (
        #     abs(can_pose.p[0]) < 0.05
        #     or (not far_enough(can_pose, alarm_pose, min_dist=0.14))
        #     or (not far_enough(can_pose, stapler_pose, min_dist=0.14))
        # ):
        #     can_pose = sample_pose(rotate=False)
        
        max_trials = 100
        trials = 0
        
        while (
            abs(can_pose.p[0]) < 0.05
            or (not far_enough(can_pose, alarm_pose, min_dist=0.14))
            or (not far_enough(can_pose, stapler_pose, min_dist=0.14))
        ) and trials < max_trials:
            can_pose = sample_pose(rotate=False)
            trials += 1
        
        if (
            abs(can_pose.p[0]) < 0.05
            or (not far_enough(can_pose, alarm_pose, min_dist=0.14))
            or (not far_enough(can_pose, stapler_pose, min_dist=0.14))
        ):
            raise RuntimeError("Failed to sample a valid can_pose within 100 tries.")

        # ====== playingcard pose ======
        playingcard_pose = sample_pose(rotate=True)
        # while (
        #     abs(playingcard_pose.p[0]) < 0.05
        #     or (not far_enough(playingcard_pose, alarm_pose, min_dist=0.14))
        #     or (not far_enough(playingcard_pose, stapler_pose, min_dist=0.14))
        #     or (not far_enough(playingcard_pose, can_pose, min_dist=0.14))
        # ):
        #     playingcard_pose = sample_pose(rotate=True)
        
        max_trials = 100
        trials = 0
        
        while (
            abs(playingcard_pose.p[0]) < 0.05
            or (not far_enough(playingcard_pose, alarm_pose, min_dist=0.14))
            or (not far_enough(playingcard_pose, stapler_pose, min_dist=0.14))
            or (not far_enough(playingcard_pose, can_pose, min_dist=0.14))
        ) and trials < max_trials:
            playingcard_pose = sample_pose(rotate=True)
            trials += 1
        
        if (
            abs(playingcard_pose.p[0]) < 0.05
            or (not far_enough(playingcard_pose, alarm_pose, min_dist=0.14))
            or (not far_enough(playingcard_pose, stapler_pose, min_dist=0.14))
            or (not far_enough(playingcard_pose, can_pose, min_dist=0.14))
        ):
            raise RuntimeError("Failed to sample a valid playingcard_pose within 100 tries.")

        # ====== create actors ======
        self.alarmclock_id = np.random.choice([1, 3], 1)[0]
        self.alarm = create_actor(
            scene=self,
            pose=alarm_pose,
            modelname="046_alarm-clock",
            convex=True,
            model_id=self.alarmclock_id,
            is_static=True,
        )

        self.stapler_id = np.random.choice([0, 1, 2, 3, 4, 5, 6], 1)[0]
        self.stapler = create_actor(
            scene=self,
            pose=stapler_pose,
            modelname="048_stapler",
            convex=True,
            model_id=self.stapler_id,
            is_static=True,
        )

        can_dict = {"071_can": [0, 1, 2, 3, 5, 6]}
        self.can_name = "071_can"
        self.can_id = can_dict[self.can_name][np.random.randint(0, len(can_dict[self.can_name]))]
        self.can = create_actor(
            scene=self,
            pose=can_pose,
            modelname=self.can_name,
            convex=True,
            model_id=self.can_id,
            is_static=True,
        )
        self.can.set_mass(0.01)

        self.playingcards_id = np.random.choice([0, 1, 2], 1)[0]
        self.playingcards = create_actor(
            scene=self,
            pose=playingcard_pose,
            modelname="081_playingcards",
            convex=True,
            model_id=self.playingcards_id,
            is_static=True,
        )

        # ====== prohibit areas ======
        self.add_prohibit_area(self.alarm, padding=0.05)
        self.add_prohibit_area(self.stapler, padding=0.05)
        self.add_prohibit_area(self.can, padding=0.10)
        self.add_prohibit_area(self.playingcards, padding=0.10)

        # ====== per-stage flags ======
        self._clicked_alarm = False
        self._clicked_stapler = False
        self._clicked_can = False
        self._clicked_playingcard = False

    # -------------------------
    # stage contact checks
    # -------------------------
    def _arm_closed(self, arm_tag: ArmTag):
        return self.is_left_gripper_close() if str(arm_tag) == "left" else self.is_right_gripper_close()

    def _check_clicked_alarm(self, arm_tag: ArmTag):
        if not self._arm_closed(arm_tag):
            return False
        alarm_pose = self.alarm.get_contact_point(0)[:3]
        positions = self.get_gripper_actor_contact_position("046_alarm-clock")
        eps = [0.03, 0.03]
        for position in positions:
            if np.all(np.abs(position[:2] - alarm_pose[:2]) < eps) and abs(position[2] - alarm_pose[2]) < 0.03:
                return True
        return False

    def _check_clicked_stapler(self, arm_tag: ArmTag):
        if not self._arm_closed(arm_tag):
            return False
        stapler_pose = self.stapler.get_contact_point(2)[:3]
        positions = self.get_gripper_actor_contact_position("048_stapler")
        eps = [0.03, 0.03]
        for position in positions:
            if np.all(np.abs(position[:2] - stapler_pose[:2]) < eps) and abs(position[2] - stapler_pose[2]) < 0.03:
                return True
        return False

    def _check_clicked_can(self, arm_tag: ArmTag):
        if not self._arm_closed(arm_tag):
            return False
        can_pose = self.can.get_contact_point(8)[:3]
        positions = self.get_gripper_actor_contact_position("071_can")
        eps = [0.03, 0.03]
        for position in positions:
            if np.all(np.abs(position[:2] - can_pose[:2]) < eps) and abs(position[2] - can_pose[2]) < 0.04:
                return True
        return False

    def _check_clicked_playingcard(self, arm_tag: ArmTag):
        if not self._arm_closed(arm_tag):
            return False
        playingcard_pose = self.playingcards.get_contact_point(0)[:3]
        positions = self.get_gripper_actor_contact_position("081_playingcards")
        eps = [0.03, 0.03]
        for position in positions:
            if np.all(np.abs(position[:2] - playingcard_pose[:2]) < eps) and abs(position[2] - playingcard_pose[2]) < 0.03:
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

        if self.stage == 1:
            arm_tag = ArmTag("right" if self.stapler.get_pose().p[0] > 0 else "left")
            if self._check_clicked_stapler(arm_tag):
                self._clicked_stapler = True
                self.stage = 2

        if self.stage == 2:
            arm_tag = ArmTag("right" if self.can.get_pose().p[0] > 0 else "left")
            if self._check_clicked_can(arm_tag):
                self._clicked_can = True
                self.stage = 3

        if self.stage == 3:
            arm_tag = ArmTag("right" if self.playingcards.get_pose().p[0] > 0 else "left")
            if self._check_clicked_playingcard(arm_tag):
                self._clicked_playingcard = True
                self.stage = 4

    # -------------------------
    # main episode: 4 actions
    # -------------------------
    def play_once(self):
        # 1) click alarm
        alarm_arm = ArmTag("right" if self.alarm.get_pose().p[0] > 0 else "left")

        self.move((
            alarm_arm,
            [
                Action(
                    alarm_arm,
                    "move",
                    self.get_grasp_pose(self.alarm, pre_dis=0.1, contact_point_id=0, arm_tag=alarm_arm)[:3]
                    + [0.5, -0.5, 0.5, 0.5],
                ),
                Action(alarm_arm, "close", target_gripper_pos=0.0),
            ],
        ))

        self.move(self.move_by_displacement(alarm_arm, z=-0.065))
        self.update_progress()
        self.move(self.move_by_displacement(alarm_arm, z=0.065))
        self.update_progress()

        # 2) click stapler
        stapler_arm = ArmTag("right" if self.stapler.get_pose().p[0] > 0 else "left")
        if stapler_arm != alarm_arm:
            self.move(
                self.grasp_actor(
                    self.stapler,
                    arm_tag=stapler_arm,
                    pre_grasp_dis=0.1,
                    grasp_dis=0.1,
                    contact_point_id=2,
                ),
                self.back_to_origin(arm_tag=stapler_arm.opposite),
            )
        else:
            self.move(
                self.grasp_actor(
                    self.stapler,
                    arm_tag=stapler_arm,
                    pre_grasp_dis=0.1,
                    grasp_dis=0.1,
                    contact_point_id=2,
                )
            )
        self.move(self.move_by_displacement(stapler_arm, z=-0.075))
        self.update_progress()
        self.move(self.move_by_displacement(stapler_arm, z=0.075))
        self.update_progress()

        # 3) click can
        can_arm = ArmTag("right" if self.can.get_pose().p[0] > 0 else "left")
        if can_arm != stapler_arm:
            self.move(
                self.grasp_actor(
                    self.can,
                    arm_tag=can_arm,
                    pre_grasp_dis=0.08,
                    grasp_dis=0.08,
                    contact_point_id=8,
                ),
                self.back_to_origin(arm_tag=can_arm.opposite),
            )
        else:
            self.move(
                self.grasp_actor(
                    self.can,
                    arm_tag=can_arm,
                    pre_grasp_dis=0.08,
                    grasp_dis=0.08,
                    contact_point_id=8,
                )
            )
        self.move(self.move_by_displacement(can_arm, z=-0.05))
        self.update_progress()
        self.move(self.move_by_displacement(can_arm, z=0.05))
        self.update_progress()

        # 4) click playingcard
        card_arm = ArmTag("right" if self.playingcards.get_pose().p[0] > 0 else "left")
        if card_arm != can_arm:
            self.move(
                self.grasp_actor(
                    self.playingcards,
                    arm_tag=card_arm,
                    pre_grasp_dis=0.08,
                    grasp_dis=0.08,
                    contact_point_id=0,
                ),
                self.back_to_origin(arm_tag=card_arm.opposite),
            )
        else:
            self.move(
                self.grasp_actor(
                    self.playingcards,
                    arm_tag=card_arm,
                    pre_grasp_dis=0.08,
                    grasp_dis=0.08,
                    contact_point_id=0,
                )
            )
        self.move(self.move_by_displacement(card_arm, z=-0.05))
        self.update_progress()
        self.move(self.move_by_displacement(card_arm, z=0.05))
        self.update_progress()

        self.info["info"] = {
            "{A}": f"046_alarm-clock/base{self.alarmclock_id}",
            "{B}": f"048_stapler/base{self.stapler_id}",
            "{C}": f"{self.can_name}/base{self.can_id}",
            "{D}": f"081_playingcards/base{self.playingcards_id}",
            "{a}": str(alarm_arm),
            "{b}": str(stapler_arm),
            "{c}": str(can_arm),
            "{d}": str(card_arm),
        }
        return self.info

    def check_success(self):
        for i in range(self.stage_sum):
            self.task_success[i] = int(self.stage >= i + 1)
        return self.stage >= self.stage_sum