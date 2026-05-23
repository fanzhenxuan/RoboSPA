from copy import deepcopy
from ._base_task import Base_Task
from .utils import *
import sapien
import math
import numpy as np


class Click_Bell_Repeat_5(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.task_success = [0, 0, 0, 0, 0]


        # 总共需要按多少次铃
        self.stage_sum = 5
        self.stage = 0

        # 当前是否已经离开按压区域
        # 用来避免同一次接触被重复计数
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
        判断当前是否形成一次有效按铃：
        1. 当前使用的夹爪是闭合状态
        2. 夹爪与铃铛顶部接触点足够接近
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
        判断夹爪是否已经离开铃铛顶部按压区域。
        用于避免一次接触被重复记成多次点击。
        """
        bell_pose = self.bell.get_contact_point(0)[:3]
        positions = self.get_gripper_actor_contact_position("050_bell")

        # 没有任何接触，说明已经离开
        if len(positions) == 0:
            return True

        # 如果仍有接触点在铃铛顶部附近，说明还没离开
        for position in positions:
            if (
                np.all(np.abs(position[:2] - bell_pose[:2]) < np.array([0.025, 0.025]))
                and abs(position[2] - bell_pose[2]) < 0.03
            ):
                return False

        return True

    def update_progress(self):
        """
        阶段推进逻辑：
        - 只有“已经离开按压区域”之后，再次按下成功，才算新的一次点击
        """
        if self.stage >= self.stage_sum:
            return

        # 先检测是否已经离开按压区域
        if self._has_left_click_area():
            self.has_left_press_area = True

        # 如果已经离开过，再次按压成功，则计数 +1
        if self.has_left_press_area and self._is_click_success():
            self.stage += 1
            # 进入下一次点击前，必须再次离开
            self.has_left_press_area = False

    def _do_one_click(self, arm_tag):
        """
        执行一次完整点击动作：
        到上方 -> 向下按 -> 检查是否计数 -> 向上抬起 -> 更新离开状态
        """
        # 到铃铛上方
        self.move(
            self.grasp_actor(
                self.bell,
                arm_tag=arm_tag,
                pre_grasp_dis=0.1,
                grasp_dis=0.1,
                contact_point_id=0,
            )
        )

        # 向下按
        self.move(self.move_by_displacement(arm_tag, z=-0.045))
        self.update_progress()

        # 抬起
        self.move(self.move_by_displacement(arm_tag, z=0.045))
        self.update_progress()

    def play_once(self):
        # 铃铛在右边用右手，否则用左手
        arm_tag = ArmTag("right" if self.bell.get_pose().p[0] > 0 else "left")

        # 先确保夹爪闭合，模拟按铃
        self.move(self.close_gripper(arm_tag=arm_tag, pos=0))

        # 顺序执行 stage_sum 次点击
        for _ in range(self.stage_sum):
            self._do_one_click(arm_tag)

        self.info["info"] = {
            "{A}": f"050_bell/base{self.bell_id}",
            "{B}": str(self.stage_sum),
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):

        for i in range(self.stage_sum):
            self.task_success[i] = int(self.stage >= i + 1)

        return self.stage >= self.stage_sum