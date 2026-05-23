from ._base_task import Base_Task
from .utils import *
import sapien
import math
from copy import deepcopy
import numpy as np


class Click_Can_Place_Items_5(Base_Task):

    def setup_demo(self, **kwargs):
        super()._init_task_env_(**kwargs)

    def load_actors(self):
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

                # while (
                #     abs(rand_pos.p[0]) < 0.15
                #     or (
                #         (rand_pos.p[0] - breadbasket_pose.p[0]) ** 2
                #         + (rand_pos.p[1] - breadbasket_pose.p[1]) ** 2
                #     ) < 0.01
                # ):
                #     rand_pos = rand_pose(
                #         xlim=[0.10, 0.27],
                #         ylim=[-0.10, 0.05],
                #         qpos=[0.707, 0.707, 0.0, 0.0],
                #         rotate_rand=True,
                #         rotate_lim=[0, np.pi / 4, 0],
                #     )
                
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

        pill_pose = rand_pose(
            xlim=[-0.25, -0.10],
            ylim=[-0.1, 0.1],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=False,
        )

        self.pillbottle_id = np.random.choice([1, 2, 3, 4, 5], 1)[0]
        self.pillbottle = create_actor(
            scene=self,
            pose=pill_pose,
            modelname="080_pillbottle",
            convex=True,
            model_id=self.pillbottle_id,
        )
        self.pillbottle.set_mass(0.05)

        pad_pose = rand_pose(
            xlim=[-0.25, -0.05],
            ylim=[-0.1, 0.1],
            qpos=[1, 0, 0, 0],
            rotate_rand=False,
        )
        # while np.sqrt(
        #     (pad_pose.p[0] - pill_pose.p[0]) ** 2
        #     + (pad_pose.p[1] - pill_pose.p[1]) ** 2
        # ) < 0.1:
        #     pad_pose = rand_pose(
        #         xlim=[-0.25, -0.05],
        #         ylim=[-0.1, 0.1],
        #         qpos=[1, 0, 0, 0],
        #         rotate_rand=False,
        #     )
        
        max_trials = 100
        trials = 0
        
        while np.sqrt(
            (pad_pose.p[0] - pill_pose.p[0]) ** 2
            + (pad_pose.p[1] - pill_pose.p[1]) ** 2
        ) < 0.1 and trials < max_trials:
            pad_pose = rand_pose(
                xlim=[-0.25, -0.05],
                ylim=[-0.1, 0.1],
                qpos=[1, 0, 0, 0],
                rotate_rand=False,
            )
            trials += 1
        
        if np.sqrt(
            (pad_pose.p[0] - pill_pose.p[0]) ** 2
            + (pad_pose.p[1] - pill_pose.p[1]) ** 2
        ) < 0.1:
            raise RuntimeError("Failed to sample a valid pad_pose within 100 tries.")

        half_size = [0.04, 0.04, 0.0005]
        self.pad = create_box(
            scene=self,
            pose=pad_pose,
            half_size=half_size,
            color=(0, 0, 1),
            name="box",
            is_static=True,
        )

        can_dict = {"071_can": [0, 1, 2, 3, 5, 6]}
        self.can_name = "071_can"
        self.can_id = np.random.choice(can_dict[self.can_name])

        can_pose = rand_pose(
            xlim=[-0.25, -0.05],
            ylim=[-0.2, -0.15],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=False,
        )
        try_num = 0
        # while (
        #     np.sqrt((can_pose.p[0] - pill_pose.p[0]) ** 2 + (can_pose.p[1] - pill_pose.p[1]) ** 2) < 0.1
        #     or np.sqrt((can_pose.p[0] - pad_pose.p[0]) ** 2 + (can_pose.p[1] - pad_pose.p[1]) ** 2) < 0.1
        #     or np.sqrt((can_pose.p[0] - breadbasket_pose.p[0]) ** 2 + (can_pose.p[1] - breadbasket_pose.p[1]) ** 2) < 0.18
        #     or np.sqrt((can_pose.p[0] - self.bread[0].get_pose().p[0]) ** 2 + (can_pose.p[1] - self.bread[0].get_pose().p[1]) ** 2) < 0.18
        # ):
        #     try_num += 1
        #     if try_num > 50:
        #         raise RuntimeError("Failed to place can after 50 tries.")
        #     can_pose = rand_pose(
        #         xlim=[-0.25, -0.05],
        #         ylim=[-0.2, 0.1],
        #         qpos=[0.5, 0.5, 0.5, 0.5],
        #         rotate_rand=False,
        #     )
        
        max_trials = 100
        trials = 0
        
        while (
            np.sqrt((can_pose.p[0] - pill_pose.p[0]) ** 2 + (can_pose.p[1] - pill_pose.p[1]) ** 2) < 0.1
            or np.sqrt((can_pose.p[0] - pad_pose.p[0]) ** 2 + (can_pose.p[1] - pad_pose.p[1]) ** 2) < 0.1
            or np.sqrt((can_pose.p[0] - breadbasket_pose.p[0]) ** 2 + (can_pose.p[1] - breadbasket_pose.p[1]) ** 2) < 0.18
            or np.sqrt((can_pose.p[0] - self.bread[0].get_pose().p[0]) ** 2 + (can_pose.p[1] - self.bread[0].get_pose().p[1]) ** 2) < 0.18
        ) and trials < max_trials:
            can_pose = rand_pose(
                xlim=[-0.25, -0.05],
                ylim=[-0.2, 0.1],
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=False,
            )
            trials += 1
        
        if (
            np.sqrt((can_pose.p[0] - pill_pose.p[0]) ** 2 + (can_pose.p[1] - pill_pose.p[1]) ** 2) < 0.1
            or np.sqrt((can_pose.p[0] - pad_pose.p[0]) ** 2 + (can_pose.p[1] - pad_pose.p[1]) ** 2) < 0.1
            or np.sqrt((can_pose.p[0] - breadbasket_pose.p[0]) ** 2 + (can_pose.p[1] - breadbasket_pose.p[1]) ** 2) < 0.18
            or np.sqrt((can_pose.p[0] - self.bread[0].get_pose().p[0]) ** 2 + (can_pose.p[1] - self.bread[0].get_pose().p[1]) ** 2) < 0.18
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

        self.add_prohibit_area(self.bread[0], padding=0.03)
        self.add_prohibit_area(self.breadbasket, padding=0.05)
        self.add_prohibit_area(self.pillbottle, padding=0.05)
        self.add_prohibit_area(self.pad, padding=0.05)

        self.click_stage_sum = 3
        self.stage_sum = 5
        self.stage = 0
        self.has_left_click_area = True

        # click1, bread, click2, pillbottle, click3
        self.task_success = [0, 0, 0, 0, 0]

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

    def update_progress(self):
        breadbasket_pose = self.breadbasket.get_pose().p
        bread_pose = self.bread[0].get_pose().p
        eps1 = np.array([0.05, 0.05])

        bread_done = int(
            np.all(np.abs(bread_pose[:2] - breadbasket_pose[:2]) < eps1)
            and bread_pose[2] > 0.73 + self.table_z_bias
        )

        pill_pos = self.pillbottle.get_pose().p
        pad_pos = self.pad.get_pose().p
        eps2 = np.array([0.03, 0.03])

        pill_done = int(
            np.all(np.abs(pill_pos[:2] - pad_pos[:2]) < eps2)
            and np.abs(pill_pos[2] - (0.741 + self.table_z_bias)) < 0.005
        )

        if self.stage < self.click_stage_sum:
            if self._has_left_can_click_area():
                self.has_left_click_area = True

            if self.has_left_click_area and self._is_can_click_success():
                self.stage += 1
                self.has_left_click_area = False

        # click1, bread, click2, pillbottle
        self.task_success[0] = int(self.stage >= 1)
        self.task_success[1] = bread_done
        self.task_success[2] = int(self.stage >= 2)
        self.task_success[3] = pill_done
        self.task_success[4] = int(self.stage >= 3)

    def play_once(self):
        can_arm = ArmTag("left" if self.can.get_pose().p[0] < 0 else "right")
        bread_arm = ArmTag("right" if self.bread[0].get_pose().p[0] > 0 else "left")
        pill_arm = ArmTag("right" if self.pillbottle.get_pose().p[0] > 0 else "left")

        def can_hover_action():
            return self.grasp_actor(
                self.can,
                arm_tag=can_arm,
                pre_grasp_dis=0.08,
                grasp_dis=0.08,
                contact_point_id=8,
            )

        def can_press_action():
            return self.grasp_actor(
                self.can,
                arm_tag=can_arm,
                pre_grasp_dis=0.02,
                grasp_dis=0.02,
                contact_point_id=8,
            )

        def do_click(back_arm=None):
            if back_arm is None or back_arm == can_arm:
                self.move(can_hover_action())
            else:
                self.move(can_hover_action(), self.back_to_origin(arm_tag=back_arm))
            self.update_progress()
            self.move(self.close_gripper(arm_tag=can_arm))
            self.move(can_press_action())
            self.update_progress()
            self.move(self.move_by_displacement(arm_tag=can_arm, z=0.05))
            self.move(self.open_gripper(arm_tag=can_arm))
            self.update_progress()

        # 1. click
        do_click()

        # 2. place bread
        if bread_arm != can_arm:
            self.move(
                self.grasp_actor(self.bread[0], arm_tag=bread_arm, pre_grasp_dis=0.07),
                self.back_to_origin(arm_tag=can_arm)
            )
        else:
            # self.move(self.back_to_origin(arm_tag=can_arm))
            self.move(self.grasp_actor(self.bread[0], arm_tag=bread_arm, pre_grasp_dis=0.07))
        self.move(self.move_by_displacement(arm_tag=bread_arm, z=0.1, move_axis="arm"))

        breadbasket_pose = self.breadbasket.get_functional_point(0)
        self.move(
            self.place_actor(
                self.bread[0],
                arm_tag=bread_arm,
                target_pose=breadbasket_pose,
                constrain="free",
                pre_dis=0.12,
            )
        )
        self.move(self.open_gripper(arm_tag=bread_arm))
        self.move(self.move_by_displacement(arm_tag=bread_arm, z=0.06))

        # 3. click
        do_click(back_arm=bread_arm if bread_arm != can_arm else None)
        self.move(self.open_gripper(arm_tag=can_arm))

        # 4. place pillbottle
        arm_tag = ArmTag("right" if self.pillbottle.get_pose().p[0] > 0 else "left")
        if arm_tag != can_arm :
        # Grasp pillbottle
            self.move(
                self.grasp_actor(
                    self.pillbottle,
                    arm_tag=arm_tag,
                    pre_grasp_dis=0.06,
                    gripper_pos=0
                ),
                self.back_to_origin(arm_tag=arm_tag.opposite)
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


        # Lift pillbottle
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.05))

        # Target pad pose
        target_pose = self.pad.get_functional_point(1)

        # Place pillbottle
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
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07))

        do_click(back_arm=arm_tag if arm_tag != can_arm else None)

        self.move(self.open_gripper(arm_tag=arm_tag))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.10))
        self.update_progress()
        self.info["info"] = {
            "{A}": f"{self.can_name}/base{self.can_id}",
            "{B}": f"076_breadbasket/base{self.basket_id}",
            "{C}": f"075_bread/base{self.bread_id[0]}",
            "{D}": f"080_pillbottle/base{self.pillbottle_id}",
            "{a}": str(can_arm),
            "{b}": str(bread_arm),
            "{c}": str(can_arm),
            "{d}": str(pill_arm),
            "{e}": str(can_arm),
        }

        return self.info

    def check_success(self):
        # print("== check_success ==")
        self.update_progress()
        # print("== task_success: ", self.task_success)
        # return True
        return self.task_success == [1, 1, 1, 1, 1]