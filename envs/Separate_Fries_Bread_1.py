from ._base_task import Base_Task
from .utils import *
import sapien
import numpy as np
import math


class Separate_Fries_Bread_1(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        # One fixed slot
        pos_lims = [
            [-0.25, 0.25],
        ]

        # Share the same y
        shared_y = np.random.uniform(-0.15, -0.10)

        self.actors = []
        self.breads = []
        self.fries = []
        self.task_success = [0]

        # Store objects and categories in slot order for update_progress
        self.slot_actors = []
        self.slot_categories = []

        # Randomly choose the unique object's category
        self.category = np.random.choice(["bread", "fries"])

        # Randomly choose the move direction
        self.move_direction = np.random.choice(["forward", "backward"])

        pose = rand_pose(
            xlim=pos_lims[0],
            ylim=[shared_y, shared_y],
            rotate_rand=True,
            qpos=[0.707, 0.707, 0.0, 0.0] if self.category == "bread" else [1.0, 0.0, 0.0, 0.0],
            rotate_lim=[0, np.pi / 4, 0] if self.category == "bread" else [0, 0, 0],
        )

        if self.category == "bread":
            bread_id = np.random.choice([0, 1, 3, 5, 6])
            actor = create_actor(
                self,
                pose=pose,
                modelname="075_bread",
                convex=True,
                model_id=bread_id,
            )
            self.breads.append(actor)
            self.bread_1 = actor
        else:
            fries_id = np.random.choice([0, 1], 1)[0]
            actor = create_actor(
                self,
                pose=pose,
                modelname="005_french-fries",
                convex=True,
                model_id=fries_id,
            )
            self.fries.append(actor)
            self.frenchfries_1 = actor
            self.fries_id_1 = fries_id

        self.actor = actor
        self.actors.append(actor)
        self.slot_actors.append(actor)
        self.slot_categories.append(self.category)

        # prohibit area
        for actor in self.actors:
            self.add_prohibit_area(actor, padding=0.03)

        self.delay(2)

    def pick_and_place_block(
        self,
        actor,
        target_xy,
        is_last=True,
        lift_z=0.1,
        place_down_z=-0.07,
        pre_grasp_dis=0.1,
        grasp_dis=0.01,
    ):
        # Choose the arm from the target position
        arm_tag = ArmTag("right" if target_xy[0] > 0 else "left")

        self.move(
            self.grasp_actor(
                actor,
                arm_tag=arm_tag,
                pre_grasp_dis=pre_grasp_dis,
                grasp_dis=grasp_dis,
            )
        )

        # Lift
        self.move(
            self.move_by_displacement(
                arm_tag,
                z=lift_z,
            )
        )

        # Translate to the target xy
        cur_xy = actor.get_pose().p[:2]
        delta_xy = np.array(target_xy) - cur_xy

        self.move(
            self.move_by_displacement(
                arm_tag,
                x=float(delta_xy[0]),
                y=float(delta_xy[1]),
            )
        )

        # place down
        self.move(
            self.move_by_displacement(
                arm_tag,
                z=place_down_z,
            )
        )

        # release
        self.move(self.open_gripper(arm_tag))

        # If this is not the last one, lift again
        if not is_last:
            self.move(
                self.move_by_displacement(
                    arm_tag,
                    z=lift_z,
                )
            )

        return arm_tag

    def play_once(self):
        cur_pose = self.actor.get_pose().p

        if self.move_direction == "forward":
            target_xy = [cur_pose[0], cur_pose[1] + 0.10]
        else:
            target_xy = [cur_pose[0], cur_pose[1] - 0.10]

        arm_tag = self.pick_and_place_block(
            actor=self.actor,
            target_xy=target_xy,
            is_last=True,
        )

        self.info["info"] = {
            "{A}": self.category,
            "{B}": self.move_direction,
            "{a}": str(arm_tag),
        }
        return self.info

    def update_progress(self):
        actor = self.slot_actors[0]
        y = actor.get_pose().p[1]

        if self.move_direction == "forward":
            self.task_success[0] = 1 if y >= -0.08 else 0
        else:
            self.task_success[0] = 1 if y <= -0.17 else 0

    def check_success(self):
        self.update_progress()
        return (
            self.task_success == [1]
            and self.robot.is_left_gripper_open()
            and self.robot.is_right_gripper_open()
        )