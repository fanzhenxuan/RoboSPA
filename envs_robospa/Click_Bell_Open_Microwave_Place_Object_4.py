import glob
import os
import math
import numpy as np
import sapien

from ._base_task import Base_Task
from .utils import *
from ._GLOBAL_CONFIGS import *
from copy import deepcopy


class Click_Bell_Open_Microwave_Place_Object_4(Base_Task):

    def setup_demo(self, is_test=False, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.task_success = [0, 0, 0, 0]
        self.click_stage_sum = 2
        self.stage_sum = 4

        self.stage = 0
        self.has_left_press_area = True

        # =========================
        # 1. 创建微波炉
        # =========================
        self.microwave_name = "044_microwave"
        self.microwave_id = np.random.randint(0, 2)

        self.microwave = rand_create_sapien_urdf_obj(
            scene=self,
            modelname=self.microwave_name,
            modelid=self.microwave_id,
            xlim=[-0.12, -0.02],
            ylim=[0.15, 0.2],
            zlim=[0.8, 0.8],
            qpos=[0.707, 0, 0, 0.707],
            fix_root_link=True,
        )
        self.microwave.set_mass(0.01)
        self.microwave.set_properties(0.0, 0.0)

        self.add_prohibit_area(self.microwave)
        self.prohibited_area.append([-0.25, -0.25, 0.25, 0.1])

        # =========================
        # 2. 创建 bell
        # =========================
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

        # =========================
        # 3. 两个前方物体 A / B
        # =========================
        def get_available_model_ids(modelname):
            asset_path = os.path.join("assets/objects", modelname)
            json_files = glob.glob(os.path.join(asset_path, "model_data*.json"))

            available_ids = []
            for file in json_files:
                base = os.path.basename(file)
                try:
                    idx = int(base.replace("model_data", "").replace(".json", ""))
                    available_ids.append(idx)
                except ValueError:
                    continue
            return available_ids

        object_list = [
            "047_mouse",
            "048_stapler",
            "057_toycar",
            "073_rubikscube",
            "075_bread",
            "077_phone",
            "081_playingcards",
            "086_woodenblock",
            "112_tea-box",
            "113_coffee-box",
            "107_soap",
        ]

        try_num, try_lim = 0, 100
        while try_num <= try_lim:
            rand_pos = rand_pose(
                xlim=[-0.10, 0.25],
                ylim=[-0.2, 0.0],
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=True,
                rotate_lim=[0, 3.14, 0],
            )

            if rand_pos.p[0] > 0:
                xlim = [0.18, 0.23]
            else:
                xlim = [-0.1, 0.1]

            target_rand_pose = rand_pose(
                xlim=xlim,
                ylim=[-0.2, 0.0],
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=True,
                rotate_lim=[0, 3.14, 0],
            )

            while (
                np.sqrt((target_rand_pose.p[0] - rand_pos.p[0]) ** 2 + (target_rand_pose.p[1] - rand_pos.p[1]) ** 2) < 0.1
            ) or (np.abs(target_rand_pose.p[1] - rand_pos.p[1]) < 0.1):
                target_rand_pose = rand_pose(
                    xlim=xlim,
                    ylim=[-0.2, 0.0],
                    qpos=[0.5, 0.5, 0.5, 0.5],
                    rotate_rand=True,
                    rotate_lim=[0, 3.14, 0],
                )

            try_num += 1
            distance = np.sqrt(np.sum((rand_pos.p[:2] - target_rand_pose.p[:2]) ** 2))

            if distance > 0.19 and rand_pos.p[0] < target_rand_pose.p[0]:
                mw_pos = self.microwave.get_pose().p
                bell_pos = self.bell.get_pose().p

                cond_mw = (
                    np.linalg.norm(rand_pos.p[:2] - mw_pos[:2]) > 0.18
                    and np.linalg.norm(target_rand_pose.p[:2] - mw_pos[:2]) > 0.18
                )
                # cond_bell = (
                #     np.linalg.norm(rand_pos.p[:2] - bell_pos[:2]) > 0.12
                #     and np.linalg.norm(target_rand_pose.p[:2] - bell_pos[:2]) > 0.12
                # )

                # if cond_mw and cond_bell:
                if cond_mw:
                    break

        if try_num > try_lim:
            raise RuntimeError("Actor create limit!")

        self.selected_modelname_A = np.random.choice(object_list)
        available_model_ids_A = get_available_model_ids(self.selected_modelname_A)
        if not available_model_ids_A:
            raise ValueError(f"No available model_data.json files found for {self.selected_modelname_A}")
        self.selected_model_id_A = np.random.choice(available_model_ids_A)

        self.object = create_actor(
            scene=self,
            pose=rand_pos,
            modelname=self.selected_modelname_A,
            convex=True,
            model_id=self.selected_model_id_A,
        )

        self.selected_modelname_B = np.random.choice(object_list)
        while self.selected_modelname_B == self.selected_modelname_A:
            self.selected_modelname_B = np.random.choice(object_list)

        available_model_ids_B = get_available_model_ids(self.selected_modelname_B)
        if not available_model_ids_B:
            raise ValueError(f"No available model_data.json files found for {self.selected_modelname_B}")
        self.selected_model_id_B = np.random.choice(available_model_ids_B)

        self.target_object = create_actor(
            scene=self,
            pose=target_rand_pose,
            modelname=self.selected_modelname_B,
            convex=True,
            model_id=self.selected_model_id_B,
        )

        self.object.set_mass(0.05)
        self.target_object.set_mass(0.05)

        self.add_prohibit_area(self.object, padding=0.05)
        self.add_prohibit_area(self.target_object, padding=0.1)

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
        microwave_done = int(self.check_microwave_open(target=0.6))

        object_pose = self.object.get_pose().p
        target_pose = self.target_object.get_pose().p
        distance = np.linalg.norm(object_pose[:2] - target_pose[:2])

        place_done = int(
            0.08 < distance < 0.2
            and object_pose[0] < target_pose[0]
            and abs(object_pose[1] - target_pose[1]) < 0.05
        )

        if self.stage < self.click_stage_sum:
            if self._has_left_click_area():
                self.has_left_press_area = True

            if self.has_left_press_area and self._is_click_success():
                self.stage += 1
                self.has_left_press_area = False

        self.task_success[0] = int(self.stage >= 1)
        self.task_success[1] = microwave_done
        self.task_success[2] = int(self.stage >= 2)
        self.task_success[3] = place_done

    def _do_one_click(self, arm_tag):
        if self.last_gripper is not None and self.last_gripper != arm_tag:
            self.move(
                self.grasp_actor(
                    self.bell,
                    arm_tag=arm_tag,
                    pre_grasp_dis=0.1,
                    grasp_dis=0.1,
                    contact_point_id=0,
                ),
                self.back_to_origin(arm_tag=arm_tag.opposite),  
            )
        else:
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
        self.move(self.open_gripper(arm_tag=arm_tag))
        self.update_progress()

    def play_once(self):
        self.last_gripper = None

        # ==================================================
        # Stage 1: 第一次 click bell
        # ==================================================
        bell_arm = ArmTag("right" if self.bell.get_pose().p[0] > 0 else "left")
        self._do_one_click(bell_arm)
        self.last_gripper = bell_arm

        # ==================================================
        # Stage 2: open microwave
        # ==================================================
        microwave_arm = ArmTag("left")

        if self.last_gripper is not None and self.last_gripper != microwave_arm:
            self.move(
                self.grasp_actor(
                    self.microwave,
                    arm_tag=microwave_arm,
                    pre_grasp_dis=0.08,
                    contact_point_id=0,
                ),
                self.back_to_origin(arm_tag=microwave_arm.opposite),
            )
        else:
            self.move(
                self.grasp_actor(
                    self.microwave,
                    arm_tag=microwave_arm,
                    pre_grasp_dis=0.08,
                    contact_point_id=0,
                )
            )

        start_qpos = self.microwave.get_qpos()[0]
        for _ in range(50):
            self.move(
                self.grasp_actor(
                    self.microwave,
                    arm_tag=microwave_arm,
                    pre_grasp_dis=0.0,
                    grasp_dis=0.0,
                    contact_point_id=4,
                )
            )

            new_qpos = self.microwave.get_qpos()[0]
            if new_qpos - start_qpos <= 0.001:
                break
            start_qpos = new_qpos

            if not self.plan_success:
                break
            if self.check_microwave_open(target=0.7):
                break

        if not self.check_microwave_open(target=0.7):
            self.plan_success = True

            self.move(self.open_gripper(arm_tag=microwave_arm))
            self.move(self.move_by_displacement(arm_tag=microwave_arm, y=-0.05, z=0.05))

            self.move(
                self.grasp_actor(
                    self.microwave,
                    arm_tag=microwave_arm,
                    pre_grasp_dis=0.02,
                    contact_point_id=1,
                )
            )

            start_qpos = self.microwave.get_qpos()[0]
            for _ in range(30):
                self.move(
                    self.grasp_actor(
                        self.microwave,
                        arm_tag=microwave_arm,
                        pre_grasp_dis=0.0,
                        grasp_dis=0.0,
                        contact_point_id=2,
                    )
                )

                new_qpos = self.microwave.get_qpos()[0]
                if new_qpos - start_qpos <= 0.001:
                    break
                start_qpos = new_qpos

                if not self.plan_success:
                    break
                if self.check_microwave_open(target=0.7):
                    break

        self.move(self.open_gripper(arm_tag=microwave_arm))
        self.move(self.move_by_displacement(arm_tag=microwave_arm, y=-0.06, z=0.06))
        self.last_gripper = microwave_arm

        # ==================================================
        # Stage 3: 第二次 click bell
        # ==================================================

        self._do_one_click(bell_arm)
        self.last_gripper = bell_arm

        # ==================================================
        # Stage 4: place A to B left
        # ==================================================
        arm_tag = ArmTag("right" if self.object.get_pose().p[0] > 0 else "left")

        if self.last_gripper is not None and self.last_gripper != arm_tag:
            self.move(
                self.grasp_actor(self.object, arm_tag=arm_tag, pre_grasp_dis=0.1),
                self.back_to_origin(arm_tag=arm_tag.opposite),
            )
        else:
            self.move(self.grasp_actor(self.object, arm_tag=arm_tag, pre_grasp_dis=0.1))

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1, move_axis="arm"))

        target_pose = self.target_object.get_pose().p.tolist()
        target_pose[0] -= 0.1

        self.move(self.place_actor(self.object, arm_tag=arm_tag, target_pose=target_pose))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.06, move_axis="arm"))

        object_pose = self.object.get_pose().p.tolist()
        # print("target_pose =", target_pose)
        # print("object_pose =", object_pose)

        self.last_gripper = arm_tag
        self.update_progress()

        self.info["info"] = {
            "{A}": f"{self.selected_modelname_A}/base{self.selected_model_id_A}",
            "{B}": f"{self.selected_modelname_B}/base{self.selected_model_id_B}",
            "{C}": f"050_bell/base{self.bell_id}",
            "{M}": f"{self.microwave_name}/base{self.microwave_id}",
            "{a}": str(bell_arm),
            "{b}": str(microwave_arm),
            "{c}": str(bell_arm),
            "{d}": str(arm_tag),
        }

        return self.info

    def check_microwave_open(self, target=0.6):
        limits = self.microwave.get_qlimits()
        qpos = self.microwave.get_qpos()
        return qpos[0] >= limits[0][1] * target

    def check_success(self):
        self.update_progress()
        return (self.task_success == [1, 1, 1, 1] and self.robot.is_left_gripper_open() and self.robot.is_right_gripper_open())