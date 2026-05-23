from ._base_task import Base_Task
from .utils import *
import sapien
import math
import numpy as np


class Lift_Pot_Repeat_1(Base_Task):

    def setup_demo(self, is_test=False, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.model_name = "060_kitchenpot"
        self.model_id = np.random.randint(0, 2)

        self.pot = rand_create_sapien_urdf_obj(
            scene=self,
            modelname=self.model_name,
            modelid=self.model_id,
            xlim=[-0.05, 0.05],
            ylim=[-0.05, 0.05],
            rotate_rand=True,
            rotate_lim=[0, 0, np.pi / 8],
            qpos=[0.704141, 0, 0, 0.71006],
        )

        self.init_pot_pose = self.pot.get_pose()
        self.init_pot_z = self.init_pot_pose.p[2]

        self.stage_sum = 1
        self.stage = 0
        self.task_success = [0]

        self.lifted_after_stage1 = False

        x, y = self.pot.get_pose().p[0], self.pot.get_pose().p[1]
        self.prohibited_area.append([x - 0.3, y - 0.1, x + 0.3, y + 0.1])

    def _is_lift_success(self):
        pot_pose = self.pot.get_pose()
        left_end = np.array(self.robot.get_left_tcp_pose()[:3])
        right_end = np.array(self.robot.get_right_tcp_pose()[:3])
        left_grasp = np.array(self.pot.get_contact_point(0)[:3])
        right_grasp = np.array(self.pot.get_contact_point(1)[:3])
        pot_dir = get_face_prod(pot_pose.q, [0, 0, 1], [0, 0, 1])

        return (
            pot_pose.p[2] > 0.82
            and np.linalg.norm(left_end - left_grasp) < 0.03
            and np.linalg.norm(right_end - right_grasp) < 0.03
            and pot_dir > 0.8
        )

    def _is_pot_upright(self):
        pot_pose = self.pot.get_pose()
        pot_dir = get_face_prod(pot_pose.q, [0, 0, 1], [0, 0, 1])
        return pot_dir > 0.8

    def _is_putdown_release_success(self):
        pot_pose = self.pot.get_pose()

        height_ok = abs(pot_pose.p[2] - (self.init_pot_z + 0.02)) < 0.05
        upright_ok = self._is_pot_upright()
        pos_xy_ok = np.linalg.norm(
            np.array(pot_pose.p[:2]) - np.array(self.init_pot_pose.p[:2])
        ) < 0.08

        return height_ok and upright_ok and pos_xy_ok

    def update_progress(self):
        if self.stage == 0:
            if self._is_lift_success():
                self.lifted_after_stage1 = True
            elif self.lifted_after_stage1 and self._is_putdown_release_success():
                self.stage = 1
            return

    def _do_one_stage(self, left_arm_tag, right_arm_tag):
        self.move(
            self.close_gripper(left_arm_tag, pos=0.5),
            self.close_gripper(right_arm_tag, pos=0.5),
        )

        self.move(
            self.grasp_actor(
                self.pot,
                left_arm_tag,
                pre_grasp_dis=0.035,
                contact_point_id=0,
            ),
            self.grasp_actor(
                self.pot,
                right_arm_tag,
                pre_grasp_dis=0.035,
                contact_point_id=1,
            ),
        )

        lift_target_z = 0.88
        lift_dis = lift_target_z - self.pot.get_pose().p[2]
        self.move(
            self.move_by_displacement(left_arm_tag, z=lift_dis),
            self.move_by_displacement(right_arm_tag, z=lift_dis),
        )
        self.update_progress()

        putdown_target_z = self.init_pot_z + 0.01
        putdown_dis = putdown_target_z - self.pot.get_pose().p[2]
        self.move(
            self.move_by_displacement(left_arm_tag, z=putdown_dis),
            self.move_by_displacement(right_arm_tag, z=putdown_dis),
        )

        self.move(
            self.open_gripper(left_arm_tag),
            self.open_gripper(right_arm_tag),
        )

        self.move(
            self.move_by_displacement(left_arm_tag, z=0.08, move_axis="arm"),
            self.move_by_displacement(right_arm_tag, z=0.08, move_axis="arm"),
        )

        self.move(
            self.move_by_displacement(left_arm_tag, y=0.05, move_axis="arm"),
            self.move_by_displacement(right_arm_tag, y=0.05, move_axis="arm"),
        )

        self.update_progress()

    def play_once(self):
        left_arm_tag = ArmTag("left")
        right_arm_tag = ArmTag("right")

        self._do_one_stage(left_arm_tag, right_arm_tag)

        self.info["info"] = {
            "{A}": f"{self.model_name}/base{self.model_id}",
            # "{B}": str(self.stage_sum),
        }
        return self.info

    def check_success(self):
        self.update_progress()

        for i in range(self.stage_sum):
            self.task_success[i] = int(self.stage >= i + 1)

        return self.stage >= self.stage_sum