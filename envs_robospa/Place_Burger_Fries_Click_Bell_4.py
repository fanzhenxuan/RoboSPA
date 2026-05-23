from copy import deepcopy
from ._base_task import Base_Task
from .utils import *
import sapien
import math
import numpy as np
from ._GLOBAL_CONFIGS import *


class Place_Burger_Fries_Click_Bell_4(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        # -------------------------
        # task_success:
        # [0] hamburg placed
        # [1] fries placed
        # [2] bell click 1
        # [3] hamburg returned
        # -------------------------
        self.task_success = [0, 0, 0, 0]
        self.click_stage_sum = 1
        self.stage_sum = 4

        # self.stage only counts bell clicks
        self.stage = 0
        self.has_left_click_area = True
        self.hamburg_was_placed = False
        self.fries_was_placed = False

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
        # hamburg on left side
        # -------------------------
        rand_pos_2 = rand_pose(
            xlim=[-0.3, -0.25],
            ylim=[-0.15, -0.07],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=True,
            rotate_lim=[0, 0, 0],
        )
        self.hamburg_init_pose = deepcopy(rand_pos_2)
        self.object1_id = np.random.choice([0, 1, 2, 3, 4, 5], 1)[0]
        self.hamburg = create_actor(
            scene=self,
            pose=rand_pos_2,
            modelname="006_hamburg",
            convex=True,
            model_id=self.object1_id,
        )
        self.hamburg.set_mass(0.05)

        # -------------------------
        # fries on right side
        # -------------------------
        rand_pos_3 = rand_pose(
            xlim=[0.2, 0.3],
            ylim=[-0.15, -0.07],
            qpos=[1.0, 0.0, 0.0, 0.0],
            rotate_rand=True,
            rotate_lim=[0, 0, 0],
        )
        self.fries_init_pose = deepcopy(rand_pos_3)
        self.object2_id = np.random.choice([0, 1], 1)[0]
        self.frenchfries = create_actor(
            scene=self,
            pose=rand_pos_3,
            modelname="005_french-fries",
            convex=True,
            model_id=self.object2_id,
        )
        self.frenchfries.set_mass(0.05)

        # -------------------------
        # bell
        # -------------------------
        bell_rand_pose = rand_pose(
            xlim=[-0.25, 0.25],
            ylim=[0.02, 0.12],
            qpos=[0.5, 0.5, 0.5, 0.5],
        )
        # while abs(bell_rand_pose.p[0]) < 0.05:
        #     bell_rand_pose = rand_pose(
        #         xlim=[-0.25, 0.25],
        #         ylim=[0.02, 0.12],
        #         qpos=[0.5, 0.5, 0.5, 0.5],
        #     )
        
        max_trials = 100
        trials = 0
        
        while abs(bell_rand_pose.p[0]) < 0.05 and trials < max_trials:
            bell_rand_pose = rand_pose(
                xlim=[-0.25, 0.25],
                ylim=[0.02, 0.12],
                qpos=[0.5, 0.5, 0.5, 0.5],
            )
            trials += 1
        
        if abs(bell_rand_pose.p[0]) < 0.05:
            raise RuntimeError("Failed to sample a valid bell_rand_pose within 100 tries.")

        self.bell_id = np.random.choice([0, 1], 1)[0]
        self.bell = create_actor(
            scene=self,
            pose=bell_rand_pose,
            modelname="050_bell",
            convex=True,
            model_id=self.bell_id,
            is_static=True,
        )

        self.check_arm_function = (
            self.is_left_gripper_close
            if self.bell.get_pose().p[0] < 0
            else self.is_right_gripper_close
        )

        # prohibit areas
        self.add_prohibit_area(self.tray, padding=0.1)
        self.add_prohibit_area(self.hamburg, padding=0.05)
        self.add_prohibit_area(self.frenchfries, padding=0.05)
        self.add_prohibit_area(self.bell, padding=0.07)

    # -------------------------------------------------
    # helper: burger/fries placement success
    # -------------------------------------------------
    def _hamburg_in_place(self):
        dis1 = np.linalg.norm(
            self.tray.get_functional_point(0, "pose").p[0:2]
            - self.hamburg.get_functional_point(0, "pose").p[0:2]
        )
        return dis1 < 0.08

    def _fries_in_place(self):
        dis2 = np.linalg.norm(
            self.tray.get_functional_point(1, "pose").p[0:2]
            - self.frenchfries.get_functional_point(0, "pose").p[0:2]
        )
        return dis2 < 0.08

    def _hamburg_back_in_place(self):
        target_pose = getattr(self, "hamburg_home_pose", self.hamburg_init_pose)
        current_pose = self.hamburg.get_pose()
        xy_dis = np.linalg.norm(current_pose.p[0:2] - target_pose.p[0:2])
        z_dis = abs(current_pose.p[2] - target_pose.p[2])
        not_on_tray = not self.check_actors_contact(self.hamburg.get_name(), self.tray.get_name())
        return xy_dis < 0.04 and z_dis < 0.005 and not_on_tray

    def _hamburg_released(self):
        return self.is_left_gripper_open() and len(self.get_gripper_actor_contact_position("006_hamburg")) == 0

    def _record_home_poses(self):
        self.hamburg_home_pose = deepcopy(self.hamburg.get_pose())
        self.hamburg_home_fp_pose = deepcopy(self.hamburg.get_functional_point(0))
        self.fries_home_pose = deepcopy(self.frenchfries.get_pose())
        self.fries_home_fp_pose = deepcopy(self.frenchfries.get_functional_point(0))

    def _settle_scene(self, steps=12):
        for _ in range(steps):
            self.scene.step()
        self._update_render()

    # -------------------------------------------------
    # helper: bell click success
    # -------------------------------------------------
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

        if len(positions) == 0:
            return True

        for position in positions:
            if (
                np.all(np.abs(position[:2] - bell_pose[:2]) < np.array([0.025, 0.025]))
                and abs(position[2] - bell_pose[2]) < 0.03
            ):
                return False

        return True

    # -------------------------------------------------
    # progress update
    # -------------------------------------------------
    def update_progress(self):
        hamburg_done = int(self._hamburg_in_place())
        fries_done = int(self._fries_in_place())
        if hamburg_done:
            self.hamburg_was_placed = True
        if fries_done:
            self.fries_was_placed = True
        hamburg_returned = int(
            self.hamburg_was_placed and self._hamburg_back_in_place() and self._hamburg_released()
        )

        # bell 一次点击进度更新
        if self.stage < self.click_stage_sum:
            if self._has_left_click_area():
                self.has_left_click_area = True

            if self.has_left_click_area and self._is_click_success():
                self.stage += 1
                self.has_left_click_area = False

        # 更新 task_success：记录子任务是否完成，不随之后的搬运动作回退
        self.task_success[0] = int(self.hamburg_was_placed)
        self.task_success[1] = int(self.fries_was_placed)
        self.task_success[2] = int(self.stage >= 1)
        self.task_success[3] = hamburg_returned

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

        self.move(self.open_gripper(arm_tag=arm_tag_left))
        self.move(self.move_by_displacement(arm_tag=arm_tag_left, z=0.08))
        self._settle_scene()
        self.update_progress()

    def _do_place_fries(self, retreat_arm_tag=None):
        arm_tag_right = ArmTag("right")
        tray_place_pose_right = self.tray.get_functional_point(1)

        if retreat_arm_tag is not None and retreat_arm_tag != arm_tag_right:
            self.move(
                self.grasp_actor(
                    self.frenchfries,
                    arm_tag=arm_tag_right,
                    pre_grasp_dis=0.1,
                ),
                self.back_to_origin(arm_tag=retreat_arm_tag),
            )
        else:
            self.move(
                self.grasp_actor(
                    self.frenchfries,
                    arm_tag=arm_tag_right,
                    pre_grasp_dis=0.1,
                )
            )

        self.move(self.move_by_displacement(arm_tag=arm_tag_right, z=0.1))

        self.move(
            self.place_actor(
                self.frenchfries,
                arm_tag=arm_tag_right,
                target_pose=tray_place_pose_right,
                functional_point_id=0,
                constrain="free",
                pre_dis=0.1,
                pre_dis_axis="fp",
            )
        )

        self.move(self.open_gripper(arm_tag=arm_tag_right))
        self.move(self.move_by_displacement(arm_tag=arm_tag_right, z=0.08))
        self._settle_scene()
        self.update_progress()

    def _do_return_hamburg(self, retreat_arm_tag=None):
        arm_tag_left = ArmTag("left")

        if retreat_arm_tag is not None and retreat_arm_tag != arm_tag_left:
            self.move(
                self.grasp_actor(
                    self.hamburg,
                    arm_tag=arm_tag_left,
                    pre_grasp_dis=0.1,
                ),
                self.back_to_origin(arm_tag=retreat_arm_tag),
            )
        else:
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
                target_pose=self.hamburg_home_fp_pose,
                functional_point_id=0,
                constrain="free",
                pre_dis=0.1,
                pre_dis_axis="fp",
            )
        )

        self.move(self.open_gripper(arm_tag=arm_tag_left))
        self.move(self.move_by_displacement(arm_tag=arm_tag_left, z=0.08))
        self._settle_scene()
        self.update_progress()

    def _do_one_click(self, arm_tag):
        """
        执行一次完整点击动作：到上方 -> 向下按 -> 检查是否计数 -> 向上抬起 -> 更新离开状态
        """
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
        self.move(self.open_gripper(arm_tag=arm_tag))

    def play_once(self):
        arm_tag_left = ArmTag("left")
        arm_tag_right = ArmTag("right")
        bell_arm_tag = ArmTag("right" if self.bell.get_pose().p[0] > 0 else "left")

        self._record_home_poses()
        # print(
        #     "[burger_return_play]",
        #     f"bell_arm={bell_arm_tag}",
        #     f"left_origin={getattr(self.robot, 'left_original_pose', None)}",
        #     f"right_origin={getattr(self.robot, 'right_original_pose', None)}",
        #     f"burger_home={self.hamburg_home_pose}",
        #     f"fries_home={self.fries_home_pose}",
        #     flush=True,
        # )
        prev_arm_tag = arm_tag_left

        # 1. place hamburg
        # print("[burger_return_play] stage=place_burger", flush=True)
        self._do_place_hamburg()
        self.update_progress()

        # 2. place fries
        # print("[burger_return_play] stage=place_fries", flush=True)
        self._do_place_fries(retreat_arm_tag=prev_arm_tag if prev_arm_tag != arm_tag_right else None)
        self.update_progress()
        prev_arm_tag = arm_tag_right

        # 3. click bell once
        # print("[burger_return_play] stage=click_once", flush=True)
        if bell_arm_tag != prev_arm_tag:
            self.move(
                self.close_gripper(arm_tag=bell_arm_tag, pos=0),
                self.back_to_origin(arm_tag=prev_arm_tag)
            )
        else:
            self.move(self.close_gripper(arm_tag=bell_arm_tag, pos=0))

        self._do_one_click(bell_arm_tag)

        self.update_progress()
        prev_arm_tag = bell_arm_tag

        # 4. return hamburg to the original position
        # print("[burger_return_play] stage=return_burger", flush=True)
        self._do_return_hamburg(retreat_arm_tag=prev_arm_tag if prev_arm_tag != arm_tag_left else None)
        self.update_progress()

        self.info["info"] = {
            # "{A}": f"006_hamburg/base{self.object1_id}",
            # "{B}": f"008_tray/base{self.tray_id}",
            # "{C}": f"005_french-fries/base{self.object2_id}",
            # "{D}": f"050_bell/base{self.bell_id}",
            # "{E}": str(self.click_stage_sum),
            # "{a}": str(arm_tag_left),
            # "{b}": str(arm_tag_right),
            # "{d}": str(bell_arm_tag),
            "{A}": f"006_hamburg/base{self.object1_id}",
            "{B}": f"008_tray/base{self.tray_id}",
            "{C}": f"005_french-fries/base{self.object2_id}",
            "{D}": f"050_bell/base{self.bell_id}",
            "{a}": str(arm_tag_left),
            "{b}": str(arm_tag_right),
            "{d}": str(bell_arm_tag),
        }
        return self.info

    def check_success(self):
        self.update_progress()
        success = self.task_success == [1, 1, 1, 1]
        # if not success:
        #     print(
        #         "[burger_return_check]",
        #         f"burger_placed={self.task_success[0]}",
        #         f"fries_placed={self.task_success[1]}",
        #         f"click={self.task_success[2]}",
        #         f"burger_returned={self.task_success[3]}",
        #         f"burger_xy={np.linalg.norm(self.hamburg.get_pose().p[0:2] - self.hamburg_home_pose.p[0:2]):.4f}",
        #         f"burger_z={abs(self.hamburg.get_pose().p[2] - self.hamburg_home_pose.p[2]):.4f}",
        #         f"fries_xy_to_tray={np.linalg.norm(self.tray.get_functional_point(1, 'pose').p[0:2] - self.frenchfries.get_functional_point(0, 'pose').p[0:2]):.4f}",
        #         flush=True,
        #     )
        return success
