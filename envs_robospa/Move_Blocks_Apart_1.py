from ._base_task import Base_Task
from .utils import *
import sapien
import math
import numpy as np
from ._GLOBAL_CONFIGS import *
from copy import deepcopy


class Move_Blocks_Apart_1(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.task_success = [0]

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

        color_name = np.random.choice(list(color_dict.keys()))
        color = color_dict[color_name]

        self.block1_color_name = color_name

        pose = rand_pose(
            xlim=[-0.06, 0.06],
            ylim=[-0.25, 0.10],
            zlim=[0.765],
            qpos=[1.0, 0.0, 0.0, 0.0],
            rotate_rand=True,
            rotate_lim=[0, 0, 0.75],
        )

        self.block1 = create_box(
            scene=self,
            pose=pose,
            half_size=half_size,
            color=color,
            name="box",
        )

        self.add_prohibit_area(self.block1, padding=0.01)

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

        # 随机决定移动到左边还是右边
        self.target_side = np.random.choice(["left", "right"])
        target_x = -0.2 if self.target_side == "left" else 0.2

        self.pick_and_place_block(
            actor=self.block1,
            target_xy=[target_x, 0.0],
            is_last=True,
        )

        self.info["info"] = {
            "{A}": f"{self.block1_color_name} block",
            "{B}": self.target_side,
        }

        return self.info

    def update_progress(self):
        x = self.block1.get_pose().p[0]
        if self.target_side == "left":
            self.task_success[0] = 1 if x <= -0.1 else 0
        else:
            self.task_success[0] = 1 if x >= 0.1 else 0

    def check_success(self):
        self.update_progress()
        return self.task_success == [1]