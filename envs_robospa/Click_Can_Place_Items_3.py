from ._base_task import Base_Task
from .utils import *
import sapien
import math
from copy import deepcopy
import numpy as np



class Click_Can_Place_Items_3(Base_Task):

    def setup_demo(self, **kwargs):
        super()._init_task_env_(**kwargs)

    def _load_breadbasket_and_bread(self):
        rand_pos = rand_pose(
            xlim=[0.10, 0.0],
            ylim=[-0.2, -0.2],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=True,
            rotate_lim=[0, 3.14, 0],
        )
        id_list = [0, 1, 2, 3, 4]
        self.basket_id = np.random.choice(id_list)
        self.breadbasket = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="076_breadbasket",
            convex=True,
            model_id=self.basket_id,
            is_static=True,
        )

        breadbasket_pose = self.breadbasket.get_pose()
        self.bread: list[Actor] = []
        self.bread_id = []

        for i in range(1):
            rand_pos = rand_pose(
                xlim=[0.10, 0.24],
                ylim=[-0.10, 0.05],
                qpos=[0.707, 0.707, 0.0, 0.0],
                rotate_rand=True,
                rotate_lim=[0, np.pi / 4, 0],
            )
            try_num = 0
            while True:
                try_num += 1
                if try_num > 50:
                    raise RuntimeError("Failed to place the only bread after 50 tries.")

                max_trials = 100
                trials = 0

                while (
                    abs(rand_pos.p[0]) < 0.15
                    or (
                        (rand_pos.p[0] - breadbasket_pose.p[0]) ** 2
                        + (rand_pos.p[1] - breadbasket_pose.p[1]) ** 2
                    ) < 0.01
                ) and trials < max_trials:
                    rand_pos = rand_pose(
                        xlim=[0.10, 0.27],
                        ylim=[-0.10, 0.05],
                        qpos=[0.707, 0.707, 0.0, 0.0],
                        rotate_rand=True,
                        rotate_lim=[0, np.pi / 4, 0],
                    )
                    trials += 1

                if (
                    abs(rand_pos.p[0]) < 0.15
                    or (
                        (rand_pos.p[0] - breadbasket_pose.p[0]) ** 2
                        + (rand_pos.p[1] - breadbasket_pose.p[1]) ** 2
                    ) < 0.01
                ):
                    raise RuntimeError("Failed to sample a valid bread rand_pos within 100 tries.")
                break

            id_list = [0, 1, 3, 5, 6]
            self.bread_id.append(np.random.choice(id_list))
            bread_actor = create_actor(
                scene=self,
                pose=rand_pos,
                modelname="075_bread",
                convex=True,
                model_id=self.bread_id[i],
            )
            self.bread.append(bread_actor)


    def _load_can(self):
        can_dict = {"071_can": [0, 1, 2, 3, 5, 6]}
        self.can_name = "071_can"
        self.can_id = np.random.choice(can_dict[self.can_name])

        can_pose = rand_pose(
            xlim=[-0.25, -0.05],
            ylim=[-0.2, -0.15],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=False,
        )

        max_trials = 100
        trials = 0

        while (
            np.sqrt((can_pose.p[0] - self.breadbasket.get_pose().p[0]) ** 2 + (can_pose.p[1] - self.breadbasket.get_pose().p[1]) ** 2) < 0.18
            or np.sqrt((can_pose.p[0] - self.bread[0].get_pose().p[0]) ** 2 + (can_pose.p[1] - self.bread[0].get_pose().p[1]) ** 2) < 0.18
            or (hasattr(self, "pillbottle") and np.sqrt((can_pose.p[0] - self.pillbottle.get_pose().p[0]) ** 2 + (can_pose.p[1] - self.pillbottle.get_pose().p[1]) ** 2) < 0.1)
            or (hasattr(self, "pad") and np.sqrt((can_pose.p[0] - self.pad.get_pose().p[0]) ** 2 + (can_pose.p[1] - self.pad.get_pose().p[1]) ** 2) < 0.1)
        ) and trials < max_trials:
            can_pose = rand_pose(
                xlim=[-0.25, -0.05],
                ylim=[-0.2, 0.1],
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=False,
            )
            trials += 1

        if (
            np.sqrt((can_pose.p[0] - self.breadbasket.get_pose().p[0]) ** 2 + (can_pose.p[1] - self.breadbasket.get_pose().p[1]) ** 2) < 0.18
            or np.sqrt((can_pose.p[0] - self.bread[0].get_pose().p[0]) ** 2 + (can_pose.p[1] - self.bread[0].get_pose().p[1]) ** 2) < 0.18
            or (hasattr(self, "pillbottle") and np.sqrt((can_pose.p[0] - self.pillbottle.get_pose().p[0]) ** 2 + (can_pose.p[1] - self.pillbottle.get_pose().p[1]) ** 2) < 0.1)
            or (hasattr(self, "pad") and np.sqrt((can_pose.p[0] - self.pad.get_pose().p[0]) ** 2 + (can_pose.p[1] - self.pad.get_pose().p[1]) ** 2) < 0.1)
        ):
            raise RuntimeError("Failed to place can after 100 tries.")

        self.can = create_actor(
            scene=self,
            pose=can_pose,
            modelname=self.can_name,
            convex=True,
            model_id=self.can_id,
            is_static=True,
        )
        self.can.set_mass(0.01)


    def _is_can_click_success(self):
        can_pose = self.can.get_contact_point(8)[:3]
        positions = self.get_gripper_actor_contact_position("071_can")
        eps_xy = np.array([0.03, 0.03])

        for position in positions:
            if (
                np.all(np.abs(position[:2] - can_pose[:2]) < eps_xy)
                and abs(position[2] - can_pose[2]) < 0.04
            ):
                return True
        return False

    def _has_left_can_click_area(self):
        can_pose = self.can.get_contact_point(8)[:3]
        positions = self.get_gripper_actor_contact_position("071_can")

        if len(positions) == 0:
            return True

        for position in positions:
            if (
                np.all(np.abs(position[:2] - can_pose[:2]) < np.array([0.03, 0.03]))
                and abs(position[2] - can_pose[2]) < 0.04
            ):
                return False

        return True

    def _update_click_progress(self):
        if self.stage < self.click_stage_sum:
            if self._has_left_can_click_area():
                self.has_left_click_area = True

            if self.has_left_click_area and self._is_can_click_success():
                self.stage += 1
                self.has_left_click_area = False

    def _do_one_click(self, can_arm, back_arm=None):
        can_hover_action = lambda: self.grasp_actor(
            self.can,
            arm_tag=can_arm,
            pre_grasp_dis=0.08,
            grasp_dis=0.08,
            contact_point_id=8,
        )

        can_press_action = lambda: self.grasp_actor(
            self.can,
            arm_tag=can_arm,
            pre_grasp_dis=0.02,
            grasp_dis=0.02,
            contact_point_id=8,
        )

        if back_arm is None or back_arm == can_arm:
            self.move(can_hover_action())
        else:
            self.move(can_hover_action(), self.back_to_origin(arm_tag=back_arm))
            # self
        self.update_progress()

        self.move(self.close_gripper(arm_tag=can_arm))
        self.move(can_press_action())
        self.update_progress()

        self.move(self.move_by_displacement(arm_tag=can_arm, z=0.05))
        self.move(self.open_gripper(arm_tag=can_arm))
        self.update_progress()


    def _do_place_bread(self, prev_arm=None):
        arm_tag = ArmTag("right" if self.bread[0].get_pose().p[0] > 0 else "left")

        if prev_arm is not None and arm_tag != prev_arm:
            self.move(
                self.grasp_actor(self.bread[0], arm_tag=arm_tag, pre_grasp_dis=0.07),
                self.back_to_origin(arm_tag=prev_arm)
            )
        else:
            self.move(self.grasp_actor(self.bread[0], arm_tag=arm_tag, pre_grasp_dis=0.07))

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1, move_axis="arm"))

        breadbasket_pose = self.breadbasket.get_functional_point(0)

        self.move(
            self.place_actor(
                self.bread[0],
                arm_tag=arm_tag,
                target_pose=breadbasket_pose,
                constrain="free",
                pre_dis=0.12,
            )
        )

        self.move(self.open_gripper(arm_tag=arm_tag))
        return arm_tag

    def _do_place_pill(self, prev_arm=None):
        arm_tag = ArmTag("right" if self.pillbottle.get_pose().p[0] > 0 else "left")

        if prev_arm is not None and arm_tag != prev_arm:
            self.move(
                self.grasp_actor(
                    self.pillbottle,
                    arm_tag=arm_tag,
                    pre_grasp_dis=0.06,
                    gripper_pos=0
                ),
                self.back_to_origin(arm_tag=prev_arm)
            )
        else:
            self.move(
                self.grasp_actor(
                    self.pillbottle,
                    arm_tag=arm_tag,
                    pre_grasp_dis=0.06,
                    gripper_pos=0
                )
            )

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.05))
        target_pose = self.pad.get_functional_point(1)

        self.move(
            self.place_actor(
                self.pillbottle,
                arm_tag=arm_tag,
                target_pose=target_pose,
                pre_dis=0.05,
                dis=0,
                functional_point_id=0,
                pre_dis_axis='fp'
            )
        )

        self.move(self.open_gripper(arm_tag=arm_tag))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.10))
        return arm_tag

    def load_actors(self):
        self._load_breadbasket_and_bread()
        self._load_can()

        self.add_prohibit_area(self.bread[0], padding=0.03)
        self.add_prohibit_area(self.breadbasket, padding=0.05)

        self.click_stage_sum = 2
        self.stage_sum = 3
        self.stage = 0
        self.has_left_click_area = True
        self.task_success = [0, 0, 0]

    def update_progress(self):
        self._update_click_progress()

        breadbasket_pose = self.breadbasket.get_pose().p
        bread_pose = self.bread[0].get_pose().p
        eps1 = np.array([0.05, 0.05])

        bread_done = int(
            np.all(np.abs(bread_pose[:2] - breadbasket_pose[:2]) < eps1)
            and bread_pose[2] > 0.73 + self.table_z_bias
        )

        self.task_success[0] = int(self.stage >= 1)
        self.task_success[1] = bread_done
        self.task_success[2] = int(self.stage >= 2)

    def play_once(self):
        can_arm = ArmTag("left" if self.can.get_pose().p[0] < 0 else "right")
        self._do_one_click(can_arm)

        bread_arm = self._do_place_bread(prev_arm=can_arm)

        # if can_arm != bread_arm:
        #     self.move(self.back_to_origin(arm_tag=bread_arm))
        self._do_one_click(can_arm, back_arm=bread_arm)

        self.info["info"] = {
            "{A}": f"{self.can_name}/base{self.can_id}",
            "{B}": f"076_breadbasket/base{self.basket_id}",
            "{C}": f"075_bread/base{self.bread_id[0]}",
            "{a}": str(can_arm),
            "{b}": str(bread_arm),
            "{c}": str(can_arm),
        }

        return self.info

    def check_success(self):
        self.update_progress()
        return self.task_success == [1, 1, 1]
