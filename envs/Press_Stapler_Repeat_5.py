from ._base_task import Base_Task
from .utils import *
from ._GLOBAL_CONFIGS import *
import numpy as np


class Press_Stapler_Repeat_5(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.task_success = [0, 0, 0, 0, 0]


        self.stage_sum = 5
        self.stage = 0
        self.has_left_press_area_after_stage1 = False
        self.has_left_press_area_after_stage2 = False
        self.has_left_press_area_after_stage3 = False
        self.has_left_press_area_after_stage4 = False

        rand_pos = rand_pose(
            xlim=[-0.2, 0.2],
            ylim=[-0.1, 0.05],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=True,
            rotate_lim=[0, np.pi, 0],
        )

        self.stapler_id = np.random.choice([0, 1, 2, 3, 4, 5, 6], 1)[0]
        self.stapler = create_actor(
            self,
            pose=rand_pos,
            modelname="048_stapler",
            convex=True,
            model_id=self.stapler_id,
            is_static=True,
        )

        self.add_prohibit_area(self.stapler, padding=0.05)

    def _is_press_success(self):
        stapler_pose = self.stapler.get_contact_point(2)[:3]
        positions = self.get_gripper_actor_contact_position("048_stapler")
        eps_xy = np.array([0.03, 0.03])

        for position in positions:
            if (
                np.all(np.abs(position[:2] - stapler_pose[:2]) < eps_xy)
                and abs(position[2] - stapler_pose[2]) < 0.03
            ):
                return True
        return False

    def _has_left_press_area(self):
        stapler_pose = self.stapler.get_contact_point(2)[:3]
        positions = self.get_gripper_actor_contact_position("048_stapler")
        eps_xy = np.array([0.03, 0.03])
        if len(positions) == 0:
            return True

        for position in positions:
            if (
                np.all(np.abs(position[:2] - stapler_pose[:2]) < eps_xy)
                and abs(position[2] - stapler_pose[2]) < 0.03
            ):
                return False

        return True

    def update_progress(self):
        # stage 0 -> 1：第一次按压成功
        if self.stage == 0:
            if self._is_press_success():
                self.stage = 1
            return

        # stage 1 -> 2：先离开，再第二次按压成功
        if self.stage == 1:
            if self._has_left_press_area():
                self.has_left_press_area_after_stage1 = True
            elif self.has_left_press_area_after_stage1 and self._is_press_success():
                self.stage = 2
            return

        # stage 2 -> 3：再离开，再第三次按压成功
        if self.stage == 2:
            if self._has_left_press_area():
                self.has_left_press_area_after_stage2 = True
            elif self.has_left_press_area_after_stage2 and self._is_press_success():
                self.stage = 3
            return

        # stage 3 -> 4：再离开，再第四次按压成功
        if self.stage == 3:
            if self._has_left_press_area():
                self.has_left_press_area_after_stage3 = True
            elif self.has_left_press_area_after_stage3 and self._is_press_success():
                self.stage = 4
            return

        # stage 4 -> 5：再离开，再第五次按压成功
        if self.stage == 4:
            if self._has_left_press_area():
                self.has_left_press_area_after_stage4 = True
            elif self.has_left_press_area_after_stage4 and self._is_press_success():
                self.stage = 5
            return

    def play_once(self):
        arm_tag = ArmTag("left" if self.stapler.get_pose().p[0] < 0 else "right")

        # 移动到订书机上方并闭合夹爪
        self.move(
            self.grasp_actor(
                self.stapler,
                arm_tag=arm_tag,
                pre_grasp_dis=0.1,
                grasp_dis=0.1,
                contact_point_id=2,
            )
        )
        self.move(self.close_gripper(arm_tag=arm_tag))

        # 第一次按压
        self.move(
            self.grasp_actor(
                self.stapler,
                arm_tag=arm_tag,
                pre_grasp_dis=0.02,
                grasp_dis=0.02,
                contact_point_id=2,
            )
        )
        self.update_progress()

        # 第一次抬起
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.05, move_axis="arm"))
        self.update_progress()

        # 第二次按压
        self.move(
            self.grasp_actor(
                self.stapler,
                arm_tag=arm_tag,
                pre_grasp_dis=0.02,
                grasp_dis=0.02,
                contact_point_id=2,
            )
        )
        self.update_progress()

        # 第二次抬起
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.05, move_axis="arm"))
        self.update_progress()

        # 第三次按压
        self.move(
            self.grasp_actor(
                self.stapler,
                arm_tag=arm_tag,
                pre_grasp_dis=0.02,
                grasp_dis=0.02,
                contact_point_id=2,
            )
        )
        self.update_progress()

        # 第三次抬起
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.05, move_axis="arm"))
        self.update_progress()

        # 第四次按压
        self.move(
            self.grasp_actor(
                self.stapler,
                arm_tag=arm_tag,
                pre_grasp_dis=0.02,
                grasp_dis=0.02,
                contact_point_id=2,
            )
        )
        self.update_progress()

        # 第四次抬起
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.05, move_axis="arm"))
        self.update_progress()

        # 第五次按压
        self.move(
            self.grasp_actor(
                self.stapler,
                arm_tag=arm_tag,
                pre_grasp_dis=0.02,
                grasp_dis=0.02,
                contact_point_id=2,
            )
        )
        self.update_progress()

        self.info["info"] = {
            "{A}": f"048_stapler/base{self.stapler_id}",
            "{B}": str(self.stage_sum),
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        for i in range(self.stage_sum):
            self.task_success[i] = int(self.stage >= i + 1)
        
        return self.stage >= self.stage_sum