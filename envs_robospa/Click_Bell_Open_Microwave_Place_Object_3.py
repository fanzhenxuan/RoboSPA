import numpy as np
import sapien

from ._base_task import Base_Task
from .utils import *
from ._GLOBAL_CONFIGS import *
from copy import deepcopy


class Click_Bell_Open_Microwave_Place_Object_3(Base_Task):

    def setup_demo(self, is_test=False, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.task_success = [0, 0, 0]
        self.click_stage_sum = 2
        self.stage_sum = 3

        self.stage = 0
        self.has_left_press_area = True

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
        microwave_done = int(self.check_microwave_open(target=0.6))

        if self.stage < self.click_stage_sum:
            if self._has_left_click_area():
                self.has_left_press_area = True

            if self.has_left_press_area and self._is_click_success():
                self.stage += 1
                self.has_left_press_area = False

        self.task_success[0] = int(self.stage >= 1)
        self.task_success[1] = microwave_done
        self.task_success[2] = int(self.stage >= 2)

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
        self.move(self.open_gripper(arm_tag=arm_tag))
        self.update_progress()

    def _open_microwave(self, microwave_arm):
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
                    contact_point_id=1,
                )
            )

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

    def play_once(self):
        self.last_gripper = None

        bell_arm = ArmTag("right" if self.bell.get_pose().p[0] > 0 else "left")
        self.move(self.close_gripper(arm_tag=bell_arm, pos=0))
        self._do_one_click(bell_arm)
        self.last_gripper = bell_arm

        microwave_arm = ArmTag("left")
        self._open_microwave(microwave_arm)
        self.last_gripper = microwave_arm

        if self.last_gripper is not None and self.last_gripper != bell_arm:
            self.move(
                self.close_gripper(arm_tag=bell_arm, pos=0),
                self.back_to_origin(arm_tag=bell_arm.opposite),
            )
        else:
            self.move(self.close_gripper(arm_tag=bell_arm, pos=0))

        self._do_one_click(bell_arm)
        self.last_gripper = bell_arm

        self.info["info"] = {
            "{C}": f"050_bell/base{self.bell_id}",
            "{M}": f"{self.microwave_name}/base{self.microwave_id}",
            "{a}": str(bell_arm),
            "{b}": str(microwave_arm),
            "{c}": str(bell_arm),
        }

        return self.info

    def check_microwave_open(self, target=0.6):
        limits = self.microwave.get_qlimits()
        qpos = self.microwave.get_qpos()
        return qpos[0] >= limits[0][1] * target

    def check_success(self):
        self.update_progress()
        return self.task_success == [1, 1, 1]
