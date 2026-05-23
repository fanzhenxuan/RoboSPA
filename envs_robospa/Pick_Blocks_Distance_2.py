from ._base_task import Base_Task
from .utils import *

import sapien
import numpy as np
from copy import deepcopy


class Pick_Blocks_Distance_2(Base_Task):
    """
    任务说明：
    - 桌面上有6个不同颜色的block
    - 不再按颜色直接pick，而是按“相对参照物的距离排序”进行pick
    - 每轮随机选择一个参照block，再在其余5个block中随机选择“第k远”的block作为目标
    - 距离排序按“从远到近”定义，k的范围为1st ~ 5th
    - 为了让距离推理更稳定，要求目标rank附近的距离差足够明显
    - 目标block靠近哪边，就自动使用哪只arm抓取
    - 任务是pick：只抓起并抬高，不放下
    """

    def setup_demo(self, **kwargs):
        super()._init_task_env_(**kwargs)

    def _ordinal_number(self, n):
        if 10 <= n % 100 <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suffix}"

    def load_actors(self):
        self.color_specs = [
            ("red", (1.0, 0.0, 0.0)),
            ("green", (0.0, 1.0, 0.0)),
            ("blue", (0.0, 0.0, 1.0)),
            ("yellow", (1.0, 1.0, 0.0)),
            ("purple", (0.55, 0.0, 0.85)),
            ("orange", (1.0, 0.5, 0.0))
        ]

        size = np.random.uniform(0.0165, 0.0185)
        half_size = (size, size, size)

        min_center_dist = 0.09
        clear_dist_gap = 0.08
        table_z = 0.745
        block_center_z = table_z + size

        while True:
            block_pose_lst = []

            for _ in range(6):
                block_pose = rand_pose(
                    xlim=[-0.27, 0.27],
                    ylim=[-0.18, 0.08],
                    zlim=[block_center_z],
                    qpos=[1, 0, 0, 0],
                    ylim_prop=True,
                    rotate_rand=True,
                    rotate_lim=[0, 0, 0.75],
                )

                def check_block_pose(candidate_pose):
                    for old_pose in block_pose_lst:
                        if np.linalg.norm(candidate_pose.p[:2] - old_pose.p[:2]) < min_center_dist:
                            return False
                    return True
                    
                # while (
                #     abs(block_pose.p[0]) < 0.05
                #     or np.sum((block_pose.p[:2] - np.array([0, -0.1])) ** 2) < 0.01
                #     or not check_block_pose(block_pose)
                # ):
                while not check_block_pose(block_pose):
                    block_pose = rand_pose(
                        xlim=[-0.28, 0.28],
                        ylim=[-0.08, 0.05],
                        zlim=[block_center_z],
                        qpos=[1, 0, 0, 0],
                        ylim_prop=True,
                        rotate_rand=True,
                        rotate_lim=[0, 0, 0.75],
                    )

                block_pose_lst.append(deepcopy(block_pose))

            xs = [pose.p[0] for pose in block_pose_lst]
            ys = [pose.p[1] for pose in block_pose_lst]

            if max(xs) - min(xs) < 0.22 or max(ys) - min(ys) < 0.08:
                continue

            valid_reference_rank_pairs = []
            for ref_idx in range(6):
                ref_xy = block_pose_lst[ref_idx].p[:2]
                dist_info = []

                for obj_idx in range(6):
                    if obj_idx == ref_idx:
                        continue
                    dist = np.linalg.norm(block_pose_lst[obj_idx].p[:2] - ref_xy)
                    dist_info.append((obj_idx, dist))

                dist_info.sort(key=lambda x: x[1], reverse=True)

                for rank_idx in range(len(dist_info)):
                    enough_gap = True

                    if rank_idx > 0:
                        if abs(dist_info[rank_idx][1] - dist_info[rank_idx - 1][1]) < clear_dist_gap:
                            enough_gap = False

                    if rank_idx < len(dist_info) - 1:
                        if abs(dist_info[rank_idx][1] - dist_info[rank_idx + 1][1]) < clear_dist_gap:
                            enough_gap = False

                    if enough_gap:
                        valid_reference_rank_pairs.append((ref_idx, rank_idx, dist_info))

            if len(valid_reference_rank_pairs) == 0:
                continue

            chosen_ref_idx, chosen_rank_idx, chosen_dist_info = valid_reference_rank_pairs[
                np.random.randint(len(valid_reference_rank_pairs))
            ]

            self.reference_idx = chosen_ref_idx
            self.distance_rank = self._ordinal_number(chosen_rank_idx + 1)
            self.target_idx = chosen_dist_info[chosen_rank_idx][0]
            self.target_distance = chosen_dist_info[chosen_rank_idx][1]

            if chosen_rank_idx == 0:
                self.distance_gap = abs(chosen_dist_info[0][1] - chosen_dist_info[1][1])
            elif chosen_rank_idx == len(chosen_dist_info) - 1:
                self.distance_gap = abs(
                    chosen_dist_info[chosen_rank_idx][1] - chosen_dist_info[chosen_rank_idx - 1][1]
                )
            else:
                self.distance_gap = min(
                    abs(chosen_dist_info[chosen_rank_idx][1] - chosen_dist_info[chosen_rank_idx - 1][1]),
                    abs(chosen_dist_info[chosen_rank_idx][1] - chosen_dist_info[chosen_rank_idx + 1][1]),
                )

            break

        self.red_block = create_box(
            scene=self,
            pose=block_pose_lst[0],
            half_size=half_size,
            color=(1.0, 0.0, 0.0),
            name="red_block",
        )

        self.green_block = create_box(
            scene=self,
            pose=block_pose_lst[1],
            half_size=half_size,
            color=(0.0, 1.0, 0.0),
            name="green_block",
        )

        self.blue_block = create_box(
            scene=self,
            pose=block_pose_lst[2],
            half_size=half_size,
            color=(0.0, 0.0, 1.0),
            name="blue_block",
        )

        self.yellow_block = create_box(
            scene=self,
            pose=block_pose_lst[3],
            half_size=half_size,
            color=(1.0, 1.0, 0.0),
            name="yellow_block",
        )

        self.purple_block = create_box(
            scene=self,
            pose=block_pose_lst[4],
            half_size=half_size,
            color=(0.55, 0.0, 0.85),
            name="purple_block",
        )

        self.orange_block = create_box(
            scene=self,
            pose=block_pose_lst[5],
            half_size=half_size,
            color=(1.0, 0.5, 0.0),
            name="orange_block",
        )

        self.blocks = {
            "red": self.red_block,
            "green": self.green_block,
            "blue": self.blue_block,
            "yellow": self.yellow_block,
            "purple": self.purple_block,
            "orange": self.orange_block,
        }

        self.block_names = [color_name for color_name, _ in self.color_specs]
        self.block_list = [self.blocks[color_name] for color_name in self.block_names]

        for block in self.blocks.values():
            self.add_prohibit_area(block, padding=0.04)
        self.prohibited_area.append([-0.17, -0.22, 0.17, -0.12])

        self.reference_color = self.block_names[self.reference_idx]
        self.reference_block = self.block_list[self.reference_idx]
        self.target_color = self.block_names[self.target_idx]
        self.target_block = self.block_list[self.target_idx]

    def play_once(self):
        self.last_gripper = None
        arm_tag = self.pick_target_block(self.target_block)

        self.info["info"] = {
            "{A}": f"{self.reference_color} block",
            "{B}": self.distance_rank,
            "{a}": arm_tag,
        }
        return self.info

    def pick_target_block(self, block):
        block_pose = block.get_pose().p
        arm_tag = ArmTag("left" if block_pose[0] < 0 else "right")

        if self.last_gripper is not None and self.last_gripper != arm_tag:
            self.move(
                self.grasp_actor(
                    block,
                    arm_tag=arm_tag,
                    pre_grasp_dis=0.09,
                    grasp_dis=0.01,
                ),
                self.back_to_origin(arm_tag=arm_tag.opposite),
            )
        else:
            self.move(
                self.grasp_actor(
                    block,
                    arm_tag=arm_tag,
                    pre_grasp_dis=0.09,
                    grasp_dis=0.01,
                )
            )

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.08))
        self.last_gripper = arm_tag
        return str(arm_tag)

    def check_success(self):
        block_pose = self.target_block.get_pose().p
        return block_pose[2] > 0.82
