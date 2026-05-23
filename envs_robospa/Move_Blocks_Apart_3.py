from ._base_task import Base_Task
from .utils import *
import sapien
import math
import numpy as np
from ._GLOBAL_CONFIGS import *
from copy import deepcopy


class Move_Blocks_Apart_3(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.task_success = [0, 0, 0]

        size = np.random.uniform(0.014, 0.016)
        half_size = (size, size, size)

        color_dict = {
            "red": (1, 0, 0),
            "green": (0, 1, 0),
            "blue": (0, 0, 1),
            "yellow": (1, 1, 0),
            "cyan": (0, 1, 1),
            "magenta": (1, 0, 1),
            "orange": (1, 0.5, 0),
            # "white": (1, 1, 1),
        }

        color_name_a, color_name_b = np.random.choice(list(color_dict.keys()), size=2, replace=False)
        color_a, color_b = color_dict[color_name_a], color_dict[color_name_b]

        self.block1_color_name = color_name_a
        self.block2_color_name = color_name_b
        self.block3_color_name = color_name_b

        def sample_pose(existing_poses):
            # ===== while修改区开始 =====
            # while True:
            #     pose = rand_pose(
            #         xlim=[-0.06, 0.06],
            #         ylim=[-0.25, 0.10],
            #         zlim=[0.765],
            #         qpos=[1.0, 0.0, 0.0, 0.0],
            #         rotate_rand=True,
            #         rotate_lim=[0, 0, 0.75],
            #     )
            #     if all(np.linalg.norm(pose.p[:2] - p.p[:2]) > 0.04 for p in existing_poses):
            #         return pose
        
            max_trials = 100
            trials = 0
        
            while trials < max_trials:
                pose = rand_pose(
                    xlim=[-0.06, 0.06],
                    ylim=[-0.25, 0.10],
                    zlim=[0.765],
                    qpos=[1.0, 0.0, 0.0, 0.0],
                    rotate_rand=True,
                    rotate_lim=[0, 0, 0.75],
                )
                if all(np.linalg.norm(pose.p[:2] - p.p[:2]) > 0.04 for p in existing_poses):
                    return pose
                trials += 1
        
            raise RuntimeError("Failed to sample a valid pose within 100 tries.")

        poses = []
        for _ in range(3):
            poses.append(sample_pose(poses))

        colors = [color_a, color_b, color_b]

        self.block1 = create_box(scene=self, pose=poses[0], half_size=half_size, color=colors[0], name="box")
        self.block2 = create_box(scene=self, pose=poses[1], half_size=half_size, color=colors[1], name="box")
        self.block3 = create_box(scene=self, pose=poses[2], half_size=half_size, color=colors[2], name="box")

        for block in [self.block1, self.block2, self.block3]:
            self.add_prohibit_area(block, padding=0.01)

        self.target_pose = self.block1.get_pose()

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
        arm_tag = ArmTag("right" if target_xy[0] > 0 else "left")

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

        self.move(
            self.move_by_displacement(
                arm_tag,
                z=lift_z,
            )
        )

        cur_xy = actor.get_pose().p[:2]
        delta_xy = np.array(target_xy) - cur_xy

        self.move(
            self.move_by_displacement(
                arm_tag,
                x=float(delta_xy[0]),
                y=float(delta_xy[1]),
            )
        )

        self.move(
            self.move_by_displacement(
                arm_tag,
                z=place_down_z,
            )
        )

        self.move(self.open_gripper(arm_tag))

        if not is_last:
            self.move(
                self.move_by_displacement(
                    arm_tag,
                    z=lift_z,
                )
            )

        self.last_arm_tag = arm_tag
        return arm_tag

    def play_once(self):
        self.last_arm_tag = None

        self.pick_and_place_block(
            actor=self.block1,
            target_xy=[0.2, 0.0],
            is_last=False,
        )

        last_two = [self.block2, self.block3]
        last_two_sorted = sorted(last_two, key=lambda b: b.get_pose().p[1], reverse=True)

        self.pick_and_place_block(
            actor=last_two_sorted[0],
            target_xy=[-0.2, 0.05],
            is_last=False,
        )

        self.pick_and_place_block(
            actor=last_two_sorted[1],
            target_xy=[-0.2, -0.08],
            is_last=True,
        )

        self.info["info"] = {
            # "{A}": f"{self.block1_color_name} block",
            # "{B}": f"{self.block2_color_name} block",
            # "{C}": f"{self.block3_color_name} block",
            "{A}": self.block1_color_name,
            "{B}": self.block2_color_name,
        }

        return self.info

    def update_progress(self):
        self.task_success[0] = 1 if self.block1.get_pose().p[0] >= 0.1 else 0
        self.task_success[1] = 1 if self.block2.get_pose().p[0] <= -0.1 else 0
        self.task_success[2] = 1 if self.block3.get_pose().p[0] <= -0.1 else 0

    def check_success(self):
        self.update_progress()
        return self.task_success == [1, 1, 1]