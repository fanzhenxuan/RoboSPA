from ._base_task import Base_Task
from .utils import *

import sapien
import numpy as np
from copy import deepcopy


class Pick_Blocks_Distance_1(Base_Task):
    """
    Task description:
    - The table has 3 blocks: red / green / blue
    - Keep the original 3 colors, but do not pick by color directly
    - Each episode samples a reference block, then randomly picks the 1st- or 2nd-farthest remaining block as the target
    - With only 3 blocks, distance rank is only 1st / 2nd
    - For stable distance reasoning, the two candidates must have a clear distance gap to the reference block
    - Use the arm on the side closer to the target block
    - This is a pick task: lift only, do not place down
    """

    def setup_demo(self, **kwargs):
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        clear_dist_gap = 0.10  # Require a clear distance gap from both candidates to the reference

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

            # Before sampling a reference, check which ones yield a clear distance gap
            valid_reference_candidates = []
            for ref_idx in range(3):
                ref_xy = block_pose_lst[ref_idx].p[:2]
                dist_info = []

                for obj_idx in range(3):
                    if obj_idx == ref_idx:
                        continue
                    dist = np.linalg.norm(block_pose_lst[obj_idx].p[:2] - ref_xy)
                    dist_info.append((obj_idx, dist))

                # Sort farthest-to-nearest from the reference
                dist_info.sort(key=lambda x: x[1], reverse=True)

                # A reference is valid only when the two candidates have a clear distance gap
                if abs(dist_info[0][1] - dist_info[1][1]) >= clear_dist_gap:
                    valid_reference_candidates.append((ref_idx, dist_info))

            if len(valid_reference_candidates) == 0:
                continue

            # Randomly pick a valid reference object
            chosen_ref_idx, chosen_dist_info = valid_reference_candidates[
                np.random.randint(len(valid_reference_candidates))
            ]

            # Among the other two objects, randomly pick 1st- or 2nd-farthest
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
