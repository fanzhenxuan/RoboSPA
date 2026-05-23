from ._base_task import Base_Task
from .utils import *
import sapien
import math
import numpy as np
from copy import deepcopy


class Stack_Blocks_Length_Order_2(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.num_blocks = 2

        # 进度记录：最底层默认已满足
        self.task_success = [1] + [0] * (self.num_blocks - 1)

        # 共享随机颜色
        shared_color = (
            np.random.random(),
            np.random.random(),
            np.random.random(),
        )

        # 固定宽度(y)和高度(z)，只改变长度(x方向)
        fixed_half_y = np.random.uniform(0.0155, 0.0165)
        fixed_half_z = np.random.uniform(0.0155, 0.0165)

        # 顺序：从最长到最短
        self.halfsize_lst = [
            (np.random.uniform(0.0440, 0.0450), fixed_half_y, fixed_half_z),  # 1st longest
            # (np.random.uniform(0.0330, 0.0340), fixed_half_y, fixed_half_z),  # 2nd
            (np.random.uniform(0.0240, 0.0250), fixed_half_y, fixed_half_z),  # 3rd
            # (np.random.uniform(0.0170, 0.0180), fixed_half_y, fixed_half_z),  # 4th
            # (np.random.uniform(0.0120, 0.0130), fixed_half_y, fixed_half_z),  # 5th
        ]

        outer_max_try = 50
        inner_max_try = 300

        xlim = [-0.25, 0.25]
        ylim = [-0.15, 0.05]
        rotate_lim = [0, 0, 0.20]
        extra_margin = 0.045

        success_flag = False
        block_pose_lst = None

        def get_xy_radius(half_size):
            hx, hy, _ = half_size
            return np.sqrt(hx ** 2 + hy ** 2)

        for outer_try in range(outer_max_try):
            cur_block_pose_lst = []
            failed = False

            for i in range(self.num_blocks):

                def check_block_pose(cur_pose, cur_size):
                    cur_radius = get_xy_radius(cur_size)
                    for j in range(len(cur_block_pose_lst)):
                        prev_pose = cur_block_pose_lst[j]
                        prev_size = self.halfsize_lst[j]
                        prev_radius = get_xy_radius(prev_size)
                        min_dist = cur_radius + prev_radius + extra_margin
                        if np.linalg.norm(cur_pose.p[:2] - prev_pose.p[:2]) < min_dist:
                            return False
                    return True

                found_valid_pose = False

                for inner_try in range(inner_max_try):
                    hz = self.halfsize_lst[i][2]

                    block_pose = rand_pose(
                        xlim=xlim,
                        ylim=ylim,
                        zlim=[0.741 + hz],
                        qpos=[1, 0, 0, 0],
                        ylim_prop=True,
                        rotate_rand=True,
                        rotate_lim=rotate_lim,
                    )

                    cond_center = abs(block_pose.p[0]) < 0.020
                    cond_front_forbidden = (
                        np.sum((block_pose.p[:2] - np.array([0.0, -0.12])) ** 2) < 0.015
                    )
                    cond_collision = not check_block_pose(block_pose, self.halfsize_lst[i])

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
            x_abs_max = np.max(np.abs(xy[:, 0]))
            y_max = np.max(xy[:, 1])

            if x_span < 0.06:
                continue
            if x_abs_max > 0.275:
                continue
            if y_max > 0.125:
                continue

            block_pose_lst = cur_block_pose_lst
            success_flag = True
            break

        # fallback：如果随机采样失败，则使用固定布局
        if not success_flag:
            block_pose_lst = []
            fallback_xy = [
                [-0.12, 0.05],
                [0.14, 0.05],
            ]

            for i in range(self.num_blocks):
                x, y = fallback_xy[i]
                hz = self.halfsize_lst[i][2]
                pose = sapien.Pose(
                    p=[x, y, 0.741 + hz],
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

        self.blocks = []
        for i in range(self.num_blocks):
            block = create_block(block_pose_lst[i], self.halfsize_lst[i], shared_color)
            self.blocks.append(block)
            setattr(self, f"block{i + 1}", block)

        for block in self.blocks:
            self.add_prohibit_area(block, padding=0.045)

        target_x = np.random.uniform(-0.03, 0.03)
        target_y = np.random.uniform(-0.15, -0.11)

        self.prohibited_area.append([
            target_x - 0.05,
            target_y - 0.025,
            target_x + 0.05,
            target_y + 0.025,
        ])

        self.base_target_pose = [
            target_x,
            target_y,
            0.74 + self.halfsize_lst[0][2] + self.table_z_bias,
            0, 1, 0, 0
        ]

    def play_once(self):
        self.last_gripper = None
        self.last_actor = None

        arm_tags = []
        for i in range(self.num_blocks):
            arm_tag = self.pick_and_place_block(self.blocks[i])
            arm_tags.append(arm_tag)

        info_dict = {}
        self.info["info"] = info_dict
        return self.info

    def pick_and_place_block(self, block):
        block_pose = block.get_pose().p
        arm_tag = ArmTag("left" if block_pose[0] < 0 else "right")

        if self.last_gripper is not None and self.last_gripper != arm_tag:
            self.move(
                self.grasp_actor(block, arm_tag=arm_tag, pre_grasp_dis=0.10),
                self.back_to_origin(arm_tag=arm_tag.opposite),
            )
        else:
            self.move(
                self.grasp_actor(block, arm_tag=arm_tag, pre_grasp_dis=0.10)
            )

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.10))

        if self.last_actor is None:
            target_pose = self.base_target_pose
        else:
            target_pose = self.last_actor.get_functional_point(1)

        self.move(
            self.place_actor(
                block,
                target_pose=target_pose,
                arm_tag=arm_tag,
                functional_point_id=0,
                pre_dis=0.06,
                dis=0.0,
                pre_dis_axis="fp",
            )
        )

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.10))

        self.last_gripper = arm_tag
        self.last_actor = block
        return str(arm_tag)

    def update_progress(self):
        eps_z = 0.02

        poses = [block.get_pose().p for block in self.blocks]

        base_z = poses[0][2]

        self.task_success[0] = 1

        cur_z = base_z
        for i in range(1, self.num_blocks):
            cur_z += self.halfsize_lst[i - 1][2] + self.halfsize_lst[i][2]
            is_aligned = abs(poses[i][2] - cur_z) <= eps_z
            self.task_success[i] = 1 if is_aligned else 0

    def check_success(self):
        self.update_progress()
        return (self.task_success == [1] * self.num_blocks and self.robot.is_left_gripper_open() and self.robot.is_right_gripper_open())