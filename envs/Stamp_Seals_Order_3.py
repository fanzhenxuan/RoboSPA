from ._base_task import Base_Task
from .utils import *
import sapien
import math
from ._GLOBAL_CONFIGS import *
from copy import deepcopy
import time
import numpy as np

class Stamp_Seals_Order_3(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.stage_sum = 3
        self.stage = 0
        self._placed_stages = [False, False, False]

        self.task_success = [0, 0, 0]

        # =========================
        # 1. 创建三个 pad
        # =========================
        half_size = [0.035, 0.035, 0.0005]

        shared_y = np.random.uniform(-0.18, 0.08)

        pad_pose3 = rand_pose(
            xlim=[-0.25, -0.15],
            ylim=[shared_y, shared_y],
            qpos=[1, 0, 0, 0],
            rotate_rand=False,
        )

        pad_pose2 = rand_pose(
            xlim=[-0.05, 0.05],
            ylim=[shared_y, shared_y],
            qpos=[1, 0, 0, 0],
            rotate_rand=False,
        )

        pad_pose1 = rand_pose(
            xlim=[0.15, 0.25],
            ylim=[shared_y, shared_y],
            qpos=[1, 0, 0, 0],
            rotate_rand=False,
        )

        colors = {
            "Red": (1, 0, 0),
            "Green": (0, 1, 0),
            "Blue": (0, 0, 1),
            "Yellow": (1, 1, 0),
            "Cyan": (0, 1, 1),
            "Magenta": (1, 0, 1),
            "Black": (0, 0, 0),
            "Gray": (0.5, 0.5, 0.5),
            "Orange": (1, 0.5, 0),
            "Purple": (0.5, 0, 0.5),
            "Brown": (0.65, 0.4, 0.16),
            "Pink": (1, 0.75, 0.8),
            "Lime": (0.5, 1, 0),
            "Olive": (0.5, 0.5, 0),
            "Teal": (0, 0.5, 0.5),
            "Maroon": (0.5, 0, 0),
            "Navy": (0, 0, 0.5),
            "Coral": (1, 0.5, 0.31),
            "Turquoise": (0.25, 0.88, 0.82),
            "Indigo": (0.29, 0, 0.51),
            "Beige": (0.96, 0.91, 0.81),
            "Tan": (0.82, 0.71, 0.55),
            "Silver": (0.75, 0.75, 0.75),
        }

        color_items = list(colors.items())
        idxs = np.random.choice(len(color_items), size=3, replace=False)

        self.color_name, self.color_value = color_items[idxs[0]]
        self.other_color_name, self.other_color_value = color_items[idxs[1]]
        self.third_color_name, self.third_color_value = color_items[idxs[2]]

        self.target = create_visual_box(
            scene=self,
            pose=pad_pose1,
            half_size=half_size,
            color=self.color_value,
            name="target_pad",
        )

        self.target2 = create_visual_box(
            scene=self,
            pose=pad_pose2,
            half_size=half_size,
            color=self.other_color_value,
            name="pad2",
        )

        self.target3 = create_visual_box(
            scene=self,
            pose=pad_pose3,
            half_size=half_size,
            color=self.third_color_value,
            name="pad3",
        )

        self.pads = [self.target, self.target2, self.target3]

        # =========================
        # 2. 创建 seal
        # =========================
        seal_pose = rand_pose(
            # xlim=[-0.25, 0.1],
            xlim=[0.00, 0.25],
            ylim=[-0.15, 0.05],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=False,
        )

        # while (
        #     np.linalg.norm(seal_pose.p[:2] - pad_pose1.p[:2]) < 0.12
        #     or np.linalg.norm(seal_pose.p[:2] - pad_pose2.p[:2]) < 0.12
        #     or np.linalg.norm(seal_pose.p[:2] - pad_pose3.p[:2]) < 0.12
        # ):
        #     seal_pose = rand_pose(
        #         # xlim=[-0.25, 0.1],
        #         xlim=[0.00, 0.25],
        #         ylim=[-0.15, 0.05],
        #         qpos=[0.5, 0.5, 0.5, 0.5],
        #         rotate_rand=False,
        #     )
        
        max_trials = 100
        trials = 0
        
        while (
            np.linalg.norm(seal_pose.p[:2] - pad_pose1.p[:2]) < 0.12
            or np.linalg.norm(seal_pose.p[:2] - pad_pose2.p[:2]) < 0.12
            or np.linalg.norm(seal_pose.p[:2] - pad_pose3.p[:2]) < 0.12
        ) and trials < max_trials:
            seal_pose = rand_pose(
                # xlim=[-0.25, 0.1],
                xlim=[0.00, 0.25],
                ylim=[-0.15, 0.05],
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=False,
            )
            trials += 1
        
        if (
            np.linalg.norm(seal_pose.p[:2] - pad_pose1.p[:2]) < 0.12
            or np.linalg.norm(seal_pose.p[:2] - pad_pose2.p[:2]) < 0.12
            or np.linalg.norm(seal_pose.p[:2] - pad_pose3.p[:2]) < 0.12
        ):
            raise RuntimeError("Failed to sample a valid seal_pose within 100 tries.")

        self.seal_id = np.random.choice([0, 2, 3, 4, 6], 1)[0]

        self.seal = create_actor(
            scene=self,
            pose=seal_pose,
            modelname="100_seal",
            convex=True,
            model_id=self.seal_id,
        )
        self.seal.set_mass(0.05)

        self.target_pose = self.target.get_pose()


        self.add_prohibit_area(self.seal, padding=0.01)
        self.add_prohibit_area(self.target, padding=0.01)
        self.add_prohibit_area(self.target2, padding=0.01)
        self.add_prohibit_area(self.target3, padding=0.01)


    def update_progress(self):
        idx = min(self.stage, self.stage_sum - 1)
        target_pad = self.pads[idx]

        seal_pose = self.seal.get_pose().p
        pad_pose = target_pad.get_pose().p

        eps1 = 0.01
        place_now = np.all(np.abs(seal_pose[:2] - pad_pose[:2]) < np.array([eps1, eps1]))

        if place_now:
            self._placed_stages[idx] = True
            if self.stage < self.stage_sum:
                self.stage += 1

        for i in range(self.stage_sum):
            self.task_success[i] = int(self.stage >= i + 1)


    def play_once(self):
        self.last_gripper = None

        # seal -> pad1
        arm_tag1 = self.pick_and_place_seal(
            self.target.get_pose(),
            pre_grasp_dis=0.1,
            pre_dis=0.1,
        )
        self.update_progress()

        # seal -> pad2
        arm_tag2 = self.pick_and_place_seal(
            self.target2.get_pose(),
            pre_grasp_dis=0.08,
            pre_dis=0.1,
        )
        self.update_progress()

        # seal -> pad3
        arm_tag3 = self.pick_and_place_seal(
            self.target3.get_pose(),
            pre_grasp_dis=0.08,
            pre_dis=0.1,
        )
        self.update_progress()

        self.info["info"] = {
            "{A}": f"100_seal/base{self.seal_id}",
            "{B}": f"{self.color_name}",
            "{C}": f"{self.other_color_name}",
            "{D}": f"{self.third_color_name}",
            # "{a}": arm_tag1,
            # "{b}": arm_tag2,
            # "{c}": arm_tag3,
        }

        return self.info

    def pick_and_place_seal(self, target_pose, pre_grasp_dis=0.1, pre_dis=0.1):
        seal_pose = self.seal.get_pose().p
        arm_tag = ArmTag("left" if seal_pose[0] < 0 else "right")

        if self.last_gripper is not None and (self.last_gripper != arm_tag):
            self.move(
                self.grasp_actor(
                    self.seal,
                    arm_tag=arm_tag,
                    pre_grasp_dis=pre_grasp_dis,
                    contact_point_id=[4, 5, 6, 7],
                ),
                self.back_to_origin(arm_tag=arm_tag.opposite),
            )
        else:
            self.move(
                self.grasp_actor(
                    self.seal,
                    arm_tag=arm_tag,
                    pre_grasp_dis=pre_grasp_dis,
                    contact_point_id=[4, 5, 6, 7],
                )
            )

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.05))

        self.move(
            self.place_actor(
                self.seal,
                arm_tag=arm_tag,
                target_pose=target_pose,
                pre_dis=pre_dis,
                constrain="auto",
            )
        )

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.05, move_axis="arm"))

        self.last_gripper = arm_tag
        return str(arm_tag)

    def check_success(self):
        return all(self.task_success)
        # seal_pose = self.seal.get_pose().p
        # target_pos = self.target5.get_pose().p
        # eps1 = 0.01

        # final_on_last_pad = np.all(
        #     np.abs(seal_pose[:2] - target_pos[:2]) < np.array([eps1, eps1])
        # )

        # return (
        #     self.stage >= self.stage_sum
        #     and final_on_last_pad
        # )