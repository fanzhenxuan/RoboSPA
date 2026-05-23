from ._base_task import Base_Task
from .utils import *

import sapien
import numpy as np


class Pick_Blocks_Canonical_3(Base_Task):

    def setup_demo(self, **kwargs):
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        self.block_color = (
            np.random.random(),
            np.random.random(),
            np.random.random(),
        )

        size = np.random.uniform(0.014, 0.020)
        self.half_size = (size, size, size)

        while True:
            center_x = np.random.uniform(-0.04, 0.00)
            center_y = np.random.uniform(-0.135, 0.010)

            col_gap = 0.125
            row_gap = 0.095
            xs = [
                center_x - 1.5 * col_gap,
                center_x - 0.5 * col_gap,
                center_x + 0.5 * col_gap,
                center_x + 1.5 * col_gap,
            ]
            ys = [
                center_y - row_gap / 2,
                center_y + row_gap / 2,
            ]

            z = 0.74 + size + self.table_z_bias

            def rand_block_pose(x, y):
                return sapien.Pose(
                    p=[
                        x + np.random.uniform(-0.010, 0.010),
                        y + np.random.uniform(-0.010, 0.015),
                        z,
                    ],
                    q=[1, 0, 0, 0],
                )

            poses = []
            for y in ys:
                for x in xs:
                    poses.append(rand_block_pose(x, y))

            valid = True
            for pose in poses:
                if np.sum((pose.p[:2] - np.array([0, -0.10])) ** 2) < 0.004:
                    valid = False
                    break

            rows = [
                poses[0:4],
                poses[4:8]
            ]

            for row in rows:
                if not (row[0].p[0] < row[1].p[0] < row[2].p[0] < row[3].p[0]):
                    valid = False

            row_y_means = [np.mean([p.p[1] for p in row]) for row in rows]
            if not (row_y_means[0] < row_y_means[1]):
                valid = False

            if abs(row_y_means[0] - row_y_means[1]) < 0.080:
                valid = False
            for row in rows:
                for i in range(3):
                    if abs(row[i].p[0] - row[i + 1].p[0]) < 0.10:
                        valid = False

            if valid:
                break

        def create_block(block_pose):
            return create_box(
                scene=self,
                pose=block_pose,
                half_size=self.half_size,
                color=self.block_color,
                name="box",
            )

        self.blocks = [create_block(pose) for pose in poses]
        self.block_grid = [
            self.blocks[0:4],
            self.blocks[4:8]
        ]

        for block in self.blocks:
            self.add_prohibit_area(block, padding=0.038)
        self.prohibited_area.append([-0.17, -0.22, 0.17, -0.12])

        self.target_row = np.random.randint(2)
        self.target_col = np.random.randint(4)
        self.target_block = self.block_grid[self.target_row][self.target_col]

        self.init_block_z = [block.get_pose().p[2] for block in self.blocks]
        self.target_index = self.target_row * 4 + self.target_col

        self.row_from_far_to_near = bool(np.random.randint(2))
        self.col_from_left_to_right = bool(np.random.randint(2))

    def play_once(self):
        self.last_gripper = None
        arm_tag = self.pick_target_block(self.target_block)

        if self.row_from_far_to_near:
            row_text = ['1st row', '2nd row']
            row_dir_text = "from near to far"
        else:
            row_text = ['2nd row', '1st row']
            row_dir_text = "from far to near"

        if self.col_from_left_to_right:
            col_text = ['1st', '2nd', '3rd', '4th']
            col_dir_text = "from left to right"
        else:
            col_text = ['4th', '3rd', '2nd', '1st']
            col_dir_text = "from right to left"

        self.info["info"] = {
            "{R}": row_text[self.target_row],
            "{C}": col_text[self.target_col],
            "{row_dir}": row_dir_text,
            "{col_dir}": col_dir_text,
            "{a}": str(arm_tag),
        }

        return self.info

    def pick_target_block(self, block):
        block_pose = block.get_pose().p
        arm_tag = ArmTag("left" if block_pose[0] < 0 else "right")

        if self.last_gripper is not None and self.last_gripper != arm_tag:
            self.move(
                self.grasp_actor(block, arm_tag=arm_tag, pre_grasp_dis=0.09),
                self.back_to_origin(arm_tag=self.last_gripper),
            )
        else:
            self.move(
                self.grasp_actor(block, arm_tag=arm_tag, pre_grasp_dis=0.09)
            )

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.08))
        self.last_gripper = arm_tag
        return str(arm_tag)

    def check_success(self):
        cur_z = [block.get_pose().p[2] for block in self.blocks]
        target_lifted = cur_z[self.target_index] > self.init_block_z[self.target_index] + 0.04

        other_stable = True
        for i in range(len(self.blocks)):
            if i == self.target_index:
                continue
            if abs(cur_z[i] - self.init_block_z[i]) > 0.025:
                other_stable = False
                break

        return target_lifted and other_stable
