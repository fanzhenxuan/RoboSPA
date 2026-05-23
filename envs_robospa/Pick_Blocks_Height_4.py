from ._base_task import Base_Task
from .utils import *

import sapien
import numpy as np
from copy import deepcopy


class Pick_Blocks_Height_4(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        shared_color = (
            np.random.random(),
            np.random.random(),
            np.random.random(),
        )

        # 固定底面尺寸，只改变高度(z方向)
        fixed_half_x = np.random.uniform(0.0155, 0.0165)
        fixed_half_y = np.random.uniform(0.0155, 0.0165)

        # 保证相邻高度区间至少相差0.004
        halfsize_lst = [
            (fixed_half_x, fixed_half_y, np.random.uniform(0.0770, 0.0780)),  # 1st tallest
            (fixed_half_x, fixed_half_y, np.random.uniform(0.0600, 0.0610)),  # 2nd
            (fixed_half_x, fixed_half_y, np.random.uniform(0.0450, 0.0460)),  # 3rd
            (fixed_half_x, fixed_half_y, np.random.uniform(0.0320, 0.0330)),  # 4th
            (fixed_half_x, fixed_half_y, np.random.uniform(0.0210, 0.0220)),  # 5th
        ]

        n_block = 5

        outer_max_try = 50
        inner_max_try = 300

        xlim = [-0.25, 0.25]
        ylim = [-0.15, 0.05]
        rotate_lim = [0, 0, 0.35]

        extra_margin = 0.08

        success_flag = False
        block_pose_lst = None

        def get_xy_radius(half_size):
            hx, hy, _ = half_size
            return np.sqrt(hx ** 2 + hy ** 2)

        for outer_try in range(outer_max_try):
            cur_block_pose_lst = []
            failed = False

            for i in range(n_block):

                def check_block_pose(cur_pose, cur_size):
                    cur_radius = get_xy_radius(cur_size)
                    for j in range(len(cur_block_pose_lst)):
                        prev_pose = cur_block_pose_lst[j]
                        prev_size = halfsize_lst[j]
                        prev_radius = get_xy_radius(prev_size)
                        min_dist = cur_radius + prev_radius + extra_margin
                        if np.linalg.norm(cur_pose.p[:2] - prev_pose.p[:2]) < min_dist:
                            return False
                    return True

                found_valid_pose = False

                for inner_try in range(inner_max_try):
                    block_pose = rand_pose(
                        xlim=xlim,
                        ylim=ylim,
                        zlim=[0.741 + halfsize_lst[i][2]],
                        qpos=[1, 0, 0, 0],
                        ylim_prop=True,
                        rotate_rand=True,
                        rotate_lim=rotate_lim,
                    )

                    cond_center = abs(block_pose.p[0]) < 0.020
                    cond_front_forbidden = np.sum((block_pose.p[:2] - np.array([0.0, -0.12])) ** 2) < 0.015
                    cond_collision = not check_block_pose(block_pose, halfsize_lst[i])

                    if cond_center or cond_front_forbidden or cond_collision:
                        continue

                    cur_block_pose_lst.append(deepcopy(block_pose))
                    found_valid_pose = True
                    break

                if not found_valid_pose:
                    failed = True
                    break

            if failed:
                continue

            xy = np.array([pose.p[:2] for pose in cur_block_pose_lst])

            x_span = xy[:, 0].max() - xy[:, 0].min()
            y_span = xy[:, 1].max() - xy[:, 1].min()
            x_abs_max = np.max(np.abs(xy[:, 0]))
            y_max = np.max(xy[:, 1])

            if x_span < 0.15:
                continue
            if y_span < 0.04:
                continue
            if x_abs_max > 0.275:
                continue
            if y_max > 0.125:
                continue

            block_pose_lst = cur_block_pose_lst
            success_flag = True
            break

        if not success_flag:
            block_pose_lst = []
            fallback_xy = [
                [-0.20, 0.03],
                [-0.08, 0.09],
                [ 0.00, 0.03],
                [ 0.12, 0.09],
                [ 0.22, 0.03],
            ]

            for i in range(n_block):
                x, y = fallback_xy[i]
                pose = sapien.Pose(
                    p=[x, y, 0.741 + halfsize_lst[i][2]],
                    q=[1, 0, 0, 0]
                )
                block_pose_lst.append(pose)

        def create_block(block_pose, half_size, color):
            return create_box(
                scene=self,
                pose=block_pose,
                half_size=half_size,
                color=color,
                name="box",
            )

        self.block_lst = []
        for i in range(n_block):
            block = create_block(block_pose_lst[i], halfsize_lst[i], shared_color)
            self.block_lst.append(block)
            self.add_prohibit_area(block, padding=0.045)

        self.block1 = self.block_lst[0]
        self.block2 = self.block_lst[1]
        self.block3 = self.block_lst[2]
        self.block4 = self.block_lst[3]
        self.block5 = self.block_lst[4]

        self.target_idx = np.random.randint(0, n_block)
        self.target_block = self.block_lst[self.target_idx]

        self.rank_str_lst = ["1st", "2nd", "3rd", "4th", "5th"]
        self.target_rank_str = self.rank_str_lst[self.target_idx]

    def play_once(self):
        self.last_gripper = None

        arm_tag = self.pick_block(self.target_block)

        self.info["info"] = {
            "{A}": self.target_rank_str,
            "{a}": str(arm_tag),
        }

        return self.info

    def pick_block(self, block):
        block_pose = block.get_pose().p

        arm_tag = ArmTag("left" if block_pose[0] < 0 else "right")

        if self.last_gripper is not None and (self.last_gripper != arm_tag):
            self.move(
                self.grasp_actor(block, arm_tag=arm_tag, pre_grasp_dis=0.25),
                self.back_to_origin(arm_tag=arm_tag.opposite),
            )
        else:
            self.move(
                self.grasp_actor(block, arm_tag=arm_tag, pre_grasp_dis=0.25)
            )

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.25))

        self.last_gripper = arm_tag
        return arm_tag

    def check_success(self):
        z = self.target_block.get_pose().p[2]
        return z > 0.82