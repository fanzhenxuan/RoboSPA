from ._base_task import Base_Task
from .utils import *
import sapien
import math
from copy import deepcopy
import numpy as np


class Lift_Fan_Repeat_3(Base_Task):

    def setup_demo(self, is_test=False, **kwargs):
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        self.task_success = [0, 0, 0]

        self.stage_sum = 3
        self.stage = 0

        self.left_after_stage1 = False
        self.left_after_stage2 = False
        self.left_after_stage3 = False

        rand_pos = rand_pose(
            xlim=[-0.1, 0.1],
            ylim=[-0.15, -0.05],
            # qpos=[0.0, 0.0, 0.707, 0.707],
            qpos=[0.707, 0.707, 0.0, 0.0],
            rotate_rand=False,
            # rotate_lim=[0, 2 * np.pi, 0],
        )
        id_list = [4, 5]
        self.fan_id = np.random.choice(id_list)
        self.fan = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="099_fan",
            convex=True,
            model_id=self.fan_id,
        )
        self.fan.set_mass(0.01)

        self.add_prohibit_area(self.fan, padding=0.07)

        # 记录风扇初始位姿，后续“放下”时回到这里
        self.init_pose_p = self.fan.get_pose().p.copy()
        self.init_pose_q = self.fan.get_pose().q.copy()
        self.target_pose = self.init_pose_p.tolist() + self.init_pose_q.tolist()

    def _fan_at_origin(self):
        fan_pose = self.fan.get_pose().p
        fan_qpose = self.fan.get_pose().q.copy()
        target_pose = np.array(self.init_pose_p)
        target_qpose = np.array(self.init_pose_q)

        if np.dot(fan_qpose, target_qpose) < 0:
            fan_qpose *= -1

        return (
            np.all(np.abs(fan_pose - target_pose) < np.array([0.04, 0.04, 0.04]))
            and np.all(np.abs(fan_qpose - target_qpose) < np.array([0.05, 0.05, 0.05, 0.05]))
        )

    def _fan_lifted(self):
        fan_z = self.fan.get_pose().p[2]
        origin_z = self.init_pose_p[2]
        return fan_z > origin_z + 0.05

    def update_progress(self):
        if self.stage == 0:
            if self._fan_lifted():
                self.left_after_stage1 = True
            elif self.left_after_stage1 and self._fan_at_origin():
                self.stage = 1
            return

        if self.stage == 1:
            if self._fan_lifted():
                self.left_after_stage2 = True
            elif self.left_after_stage2 and self._fan_at_origin():
                self.stage = 2
            return

        if self.stage == 2:
            if self._fan_lifted():
                self.left_after_stage3 = True
            elif self.left_after_stage3 and self._fan_at_origin():
                self.stage = 3
            return

    def _lift_and_put_back_once(self, arm_tag, lift_z=0.08):
        self.move(self.grasp_actor(self.fan, arm_tag=arm_tag, pre_grasp_dis=0.05))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=lift_z, move_axis="arm"))
        self.update_progress()

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=-lift_z + 0.005, move_axis="world"))  # 放下
        self.move(self.open_gripper(arm_tag))                                                         # 松手
        self.update_progress()

    def play_once(self):
        arm_tag = ArmTag("right" if self.fan.get_pose().p[0] > 0 else "left")

        # 原地举起放下 3 次
        self._lift_and_put_back_once(arm_tag, lift_z=0.08)
        self._lift_and_put_back_once(arm_tag, lift_z=0.08)
        self._lift_and_put_back_once(arm_tag, lift_z=0.08)

        self.info["info"] = {
            "{A}": f"099_fan/base{self.fan_id}",
            "{B}": str(self.stage_sum),
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        self.update_progress()

        # 前几个成功就置 1，比如 stage=2 -> [1,1,0]
        for i in range(self.stage_sum):
            self.task_success[i] = int(self.stage >= i + 1)

        return self.stage >= self.stage_sum