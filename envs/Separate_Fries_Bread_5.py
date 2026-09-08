from ._base_task import Base_Task
from .utils import *
import sapien
import numpy as np
import math


class Separate_Fries_Bread_5(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        # Five fixed slots: left to right
        pos_lims = [
            [-0.25, -0.23],
            [-0.13, -0.11],
            [-0.01,  0.01],
            [ 0.11,  0.13],
            [ 0.23,  0.25],
        ]

        # Share the same y
        shared_y = np.random.uniform(-0.15, -0.10)

        self.actors = []
        self.breads = []
        self.fries = []
        self.task_success = [0, 0, 0, 0, 0]

        # Store objects and categories in slot order so update_progress can update by position
        self.slot_actors = []
        self.slot_categories = []

        # =========================
        # Randomly decide which category is first
        # First category has 2, second has 3
        # Then shuffle category order across the five slots
        # =========================
        if np.random.rand() < 0.5:
            self.first_category = "fries"
            self.second_category = "bread"
            category_order = ["fries", "fries", "bread", "bread", "bread"]
        else:
            self.first_category = "bread"
            self.second_category = "fries"
            category_order = ["bread", "bread", "fries", "fries", "fries"]

        np.random.shuffle(category_order)

        # Save first/second category object lists for play_once
        self.first_category_actors = []
        self.second_category_actors = []

        # =========================
        # Create objects in the five slots in order
        # =========================
        bread_count = 0
        fries_count = 0

        for i, category in enumerate(category_order):
            pose = rand_pose(
                xlim=pos_lims[i],
                ylim=[shared_y, shared_y],
                rotate_rand=True,
                qpos=[0.707, 0.707, 0.0, 0.0] if category == "bread" else [1.0, 0.0, 0.0, 0.0],
                rotate_lim=[0, np.pi / 4, 0] if category == "bread" else [0, 0, 0],
            )

            if category == "bread":
                bread_id = np.random.choice([0, 1, 3, 5, 6])
                actor = create_actor(
                    self,
                    pose=pose,
                    modelname="075_bread",
                    convex=True,
                    model_id=bread_id,
                )
                bread_count += 1
                setattr(self, f"bread_{bread_count}", actor)
                self.breads.append(actor)

                if self.first_category == "bread":
                    self.first_category_actors.append(actor)
                else:
                    self.second_category_actors.append(actor)

            else:
                fries_id = np.random.choice([0, 1], 1)[0]
                actor = create_actor(
                    self,
                    pose=pose,
                    modelname="005_french-fries",
                    convex=True,
                    model_id=fries_id,
                )
                fries_count += 1
                setattr(self, f"frenchfries_{fries_count}", actor)
                setattr(self, f"fries_id_{fries_count}", fries_id)
                self.fries.append(actor)

                if self.first_category == "fries":
                    self.first_category_actors.append(actor)
                else:
                    self.second_category_actors.append(actor)

            self.actors.append(actor)
            self.slot_actors.append(actor)
            self.slot_categories.append(category)

        # =========================
        # prohibit area
        # =========================
        for actor in self.actors:
            self.add_prohibit_area(actor, padding=0.03)

        self.delay(2)

    def pick_and_place_block(
        self,
        actor,
        target_xy,
        is_last=False,
        lift_z=0.1,
        place_down_z=-0.07,
        pre_grasp_dis=0.1,
        grasp_dis=0.01,
    ):
        # Choose the arm from the target position
        arm_tag = ArmTag("right" if target_xy[0] > 0 else "left")

        # If the arm switches, retract the other hand while grasping the current object
        if hasattr(self, "last_arm_tag") and self.last_arm_tag is not None and self.last_arm_tag != arm_tag:
            self.move(
                self.grasp_actor(
                    actor,
                    arm_tag=arm_tag,
                    pre_grasp_dis=pre_grasp_dis,
                    grasp_dis=grasp_dis,
                ),
                self.back_to_origin(arm_tag=arm_tag.opposite)
            )
        else:
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

        # Record which hand was used this time
        self.last_arm_tag = arm_tag

        return arm_tag

    def play_once(self):
        self.last_arm_tag = None

        # =========================
        # First category: move up by +0.10
        # =========================
        for actor in self.first_category_actors:
            cur_pose = actor.get_pose().p
            target_xy = [cur_pose[0], cur_pose[1] + 0.10]

            self.pick_and_place_block(
                actor=actor,
                target_xy=target_xy,
                is_last=False,
            )

        # =========================
        # Second category: move down by -0.10
        # =========================
        for i, actor in enumerate(self.second_category_actors):
            cur_pose = actor.get_pose().p
            target_xy = [cur_pose[0], cur_pose[1] - 0.10]

            is_last = (i == len(self.second_category_actors) - 1)
            self.pick_and_place_block(
                actor=actor,
                target_xy=target_xy,
                is_last=is_last,
            )

        self.info["info"] = {
            "{A}": self.first_category,
            "{B}": self.second_category,
        }
        return self.info

    def update_progress(self):
        # Update task_success in slot order, not first/second_category order
        for i, (actor, category) in enumerate(zip(self.slot_actors, self.slot_categories)):
            y = actor.get_pose().p[1]

            if category == self.first_category:
                # First category needs to move up
                self.task_success[i] = 1 if y >= -0.08 else 0
            else:
                # Second category needs to move down
                self.task_success[i] = 1 if y <= -0.17 else 0

    def check_success(self):
        self.update_progress()
        return (
            self.task_success == [1, 1, 1, 1, 1]
            and self.robot.is_left_gripper_open()
            and self.robot.is_right_gripper_open()
        )