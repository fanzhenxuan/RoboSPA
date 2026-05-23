from ._base_task import Base_Task
from .utils import *

import sapien
import numpy as np
from copy import deepcopy


class Pick_Blocks_Distance_1(Base_Task):
    """
    任务说明：
    - 桌面上有3个block：red / green / blue
    - 保持原有3个颜色设定，但不再按颜色直接pick
    - 每轮随机选择一个参照block，再在其余2个block中随机选择“第1远”或“第2远”的block作为目标
    - 由于总共只有3个block，因此距离排序只会出现1st / 2nd
    - 为了让距离推理更稳定，要求两个候选block到参照block的距离差足够明显
    - 目标block靠近哪边，就自动使用哪只arm抓取
    - 任务是pick：只抓起并抬高，不放下
    """

    def setup_demo(self, **kwargs):
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        clear_dist_gap = 0.10  # 要求两个候选物体到参照物的距离差明显

        while True:
            block_pose_lst = []

            for _ in range(3):
                block_pose = rand_pose(
                    xlim=[-0.27, 0.27],
                    ylim=[-0.18, 0.08],
                    zlim=[0.765],
                    qpos=[1, 0, 0, 0],
                    ylim_prop=True,
                    rotate_rand=True,
                    rotate_lim=[0, 0, 0.75],
                )

                def check_block_pose(candidate_pose):
                    for old_pose in block_pose_lst:
                        if np.sum((candidate_pose.p[:2] - old_pose.p[:2]) ** 2) < 0.01:
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
                        zlim=[0.765],
                        qpos=[1, 0, 0, 0],
                        ylim_prop=True,
                        rotate_rand=True,
                        rotate_lim=[0, 0, 0.75],
                    )

                block_pose_lst.append(deepcopy(block_pose))

            xs = [pose.p[0] for pose in block_pose_lst]
            ys = [pose.p[1] for pose in block_pose_lst]
            xs_sorted = sorted(xs)

            too_ordered = True
            for i in range(2):
                if abs(xs_sorted[i + 1] - xs_sorted[i]) > 0.12:
                    too_ordered = False
                    break
            if max(ys) - min(ys) > 0.04:
                too_ordered = False

            if too_ordered:
                continue

            # 随机选参照物之前，先检查哪些参照物能形成“明显距离差”
            valid_reference_candidates = []
            for ref_idx in range(3):
                ref_xy = block_pose_lst[ref_idx].p[:2]
                dist_info = []

                for obj_idx in range(3):
                    if obj_idx == ref_idx:
                        continue
                    dist = np.linalg.norm(block_pose_lst[obj_idx].p[:2] - ref_xy)
                    dist_info.append((obj_idx, dist))

                # 按“距离参照物从远到近”排序
                dist_info.sort(key=lambda x: x[1], reverse=True)

                # 只有当两个候选物体的距离差足够明显时，才允许作为有效参照物
                if abs(dist_info[0][1] - dist_info[1][1]) >= clear_dist_gap:
                    valid_reference_candidates.append((ref_idx, dist_info))

            if len(valid_reference_candidates) == 0:
                continue

            # 随机选择一个有效参照物
            chosen_ref_idx, chosen_dist_info = valid_reference_candidates[
                np.random.randint(len(valid_reference_candidates))
            ]

            # 在其余两个物体中，随机选择第1远或第2远
            chosen_rank_idx = np.random.randint(2)  # 0 -> 1st, 1 -> 2nd

            self.reference_idx = chosen_ref_idx
            self.distance_rank = ["1st", "2nd"][chosen_rank_idx]
            self.target_idx = chosen_dist_info[chosen_rank_idx][0]
            self.target_distance = chosen_dist_info[chosen_rank_idx][1]
            self.distance_gap = abs(chosen_dist_info[0][1] - chosen_dist_info[1][1])

            break

        size = np.random.uniform(0.018, 0.022)
        half_size = (size, size, size)

        self.red_block = create_box(
            scene=self,
            pose=block_pose_lst[0],
            half_size=half_size,
            color=(1, 0, 0),
            name="red_block",
        )
        self.green_block = create_box(
            scene=self,
            pose=block_pose_lst[1],
            half_size=half_size,
            color=(0, 1, 0),
            name="green_block",
        )
        self.blue_block = create_box(
            scene=self,
            pose=block_pose_lst[2],
            half_size=half_size,
            color=(0, 0, 1),
            name="blue_block",
        )

        self.blocks = {
            "red": self.red_block,
            "green": self.green_block,
            "blue": self.blue_block,
        }
        self.block_names = ["red", "green", "blue"]
        self.block_list = [self.red_block, self.green_block, self.blue_block]

        for block in self.blocks.values():
            self.add_prohibit_area(block, padding=0.05)
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