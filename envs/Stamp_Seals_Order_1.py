from ._base_task import Base_Task
from .utils import *
import sapien
import math
from ._GLOBAL_CONFIGS import *
from copy import deepcopy
import time
import numpy as np


class Stamp_Seals_Order_1(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.stage_sum = 1
        self.stage = 0
        self._placed_stages = [False]
        self.task_success = [0]

        rand_pos = rand_pose(
            xlim=[-0.25, 0.25],
            ylim=[-0.05, 0.05],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=False,
        )
        # while abs(rand_pos.p[0]) < 0.05:
        #     rand_pos = rand_pose(
        #         xlim=[-0.25, 0.25],
        #         ylim=[-0.05, 0.05],
        #         qpos=[0.5, 0.5, 0.5, 0.5],
        #         rotate_rand=False,
        #     )
        
        max_trials = 100
        trials = 0
        
        while abs(rand_pos.p[0]) < 0.05 and trials < max_trials:
            rand_pos = rand_pose(
                xlim=[-0.25, 0.25],
                ylim=[-0.05, 0.05],
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=False,
            )
            trials += 1
        
        if abs(rand_pos.p[0]) < 0.05:
            raise RuntimeError("Failed to sample a valid rand_pos within 100 tries.")

        self.seal_id = np.random.choice([0, 2, 3, 4, 6], 1)[0]

        self.seal = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="100_seal",
            convex=True,
            model_id=self.seal_id,
        )
        self.seal.set_mass(0.05)

        if rand_pos.p[0] > 0:
            xlim = [0.05, 0.25]
        else:
            xlim = [-0.25, -0.05]

        target_rand_pose = rand_pose(
            xlim=xlim,
            ylim=[-0.05, 0.05],
            qpos=[1, 0, 0, 0],
            rotate_rand=False,
        )
        # while np.sqrt(
        #     (target_rand_pose.p[0] - rand_pos.p[0]) ** 2
        #     + (target_rand_pose.p[1] - rand_pos.p[1]) ** 2
        # ) < 0.1:
        #     target_rand_pose = rand_pose(
        #         xlim=xlim,
        #         ylim=[-0.05, 0.1],
        #         qpos=[1, 0, 0, 0],
        #         rotate_rand=False,
        #     )
        
        max_trials = 100
        trials = 0
        
        while np.sqrt(
            (target_rand_pose.p[0] - rand_pos.p[0]) ** 2
            + (target_rand_pose.p[1] - rand_pos.p[1]) ** 2
        ) < 0.1 and trials < max_trials:
            target_rand_pose = rand_pose(
                xlim=xlim,
                ylim=[-0.05, 0.1],
                qpos=[1, 0, 0, 0],
                rotate_rand=False,
            )
            trials += 1
        
        if np.sqrt(
            (target_rand_pose.p[0] - rand_pos.p[0]) ** 2
            + (target_rand_pose.p[1] - rand_pos.p[1]) ** 2
        ) < 0.1:
            raise RuntimeError("Failed to sample a valid target_rand_pose within 100 tries.")

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
        idx = np.random.choice(len(color_items))
        self.color_name, self.color_value = color_items[idx]

        half_size = [0.035, 0.035, 0.0005]
        self.target = create_visual_box(
            scene=self,
            pose=target_rand_pose,
            half_size=half_size,
            color=self.color_value,
            name="box",
        )

        self.pads = [self.target]

        self.add_prohibit_area(self.seal, padding=0.1)
        self.add_prohibit_area(self.target, padding=0.1)
        self.target_pose = self.target.get_pose()

    def update_progress(self):
        idx = min(self.stage, self.stage_sum - 1)
        target_pad = self.pads[idx]

        seal_pose = self.seal.get_pose().p
        pad_pose = target_pad.get_pose().p

        eps1 = 0.01
        place_now = np.all(np.abs(seal_pose[:2] - pad_pose[:2]) < np.array([eps1, eps1]))

        if place_now and not self._placed_stages[idx]:
            self._placed_stages[idx] = True
            if self.stage < self.stage_sum:
                self.stage += 1

        for i in range(self.stage_sum):
            self.task_success[i] = int(self.stage >= i + 1)

    def play_once(self):
        self.last_gripper = None

        arm_tag1 = self.pick_and_place_seal(
            self.target.get_pose(),
            pre_grasp_dis=0.1,
            pre_dis=0.1,
        )
        self.update_progress()

        self.info["info"] = {
            "{A}": f"100_seal/base{self.seal_id}",
            "{B}": f"{self.color_name}",
            # "{a}": arm_tag1,
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