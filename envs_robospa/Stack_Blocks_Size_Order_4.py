from ._base_task import Base_Task
from .utils import *
import sapien
import math
import numpy as np
from copy import deepcopy


class Stack_Blocks_Size_Order_4(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.num_blocks = 4

        # 进度记录：最底下方块默认已满足
        self.task_success = [1] + [0] * (self.num_blocks - 1)

        self.halfsize_lst =[0.02025, 0.0165, 0.01275, 0.009]
        
        # 随机颜色
        color = (
            np.random.random(),
            np.random.random(),
            np.random.random(),
        )

        while True:
            block_pose_lst = []

            for i in range(self.num_blocks):
                block_pose = rand_pose(
                    xlim=[-0.28, 0.28],
                    ylim=[-0.08, 0.05],
                    zlim=[0.741 + self.halfsize_lst[i]],
                    qpos=[1, 0, 0, 0],
                    ylim_prop=True,
                    rotate_rand=True,
                    rotate_lim=[0, 0, 0.75],
                )

                def check_block_pose(cur_pose):
                    for old_pose in block_pose_lst:
                        if np.sum((cur_pose.p[:2] - old_pose.p[:2]) ** 2) < 0.01:
                            return False
                    return True

                # while (
                #     abs(block_pose.p[0]) < 0.05
                #     or np.sum((block_pose.p[:2] - np.array([0, -0.1])) ** 2) < 0.0225
                #     or not check_block_pose(block_pose)
                # ):
                #     block_pose = rand_pose(
                #         xlim=[-0.28, 0.28],
                #         ylim=[-0.08, 0.05],
                #         zlim=[0.741 + self.halfsize_lst[i]],
                #         qpos=[1, 0, 0, 0],
                #         ylim_prop=True,
                #         rotate_rand=True,
                #         rotate_lim=[0, 0, 0.75],
                #     )
                
                max_trials = 100
                trials = 0
                
                while (
                    abs(block_pose.p[0]) < 0.05
                    or np.sum((block_pose.p[:2] - np.array([0, -0.1])) ** 2) < 0.0225
                    or not check_block_pose(block_pose)
                ) and trials < max_trials:
                    block_pose = rand_pose(
                        xlim=[-0.28, 0.28],
                        ylim=[-0.08, 0.05],
                        zlim=[0.741 + self.halfsize_lst[i]],
                        qpos=[1, 0, 0, 0],
                        ylim_prop=True,
                        rotate_rand=True,
                        rotate_lim=[0, 0, 0.75],
                    )
                    trials += 1
                
                if (
                    abs(block_pose.p[0]) < 0.05
                    or np.sum((block_pose.p[:2] - np.array([0, -0.1])) ** 2) < 0.0225
                    or not check_block_pose(block_pose)
                ):
                    raise RuntimeError(f"Failed to sample a valid block_pose for block {i} within 100 tries.")
                    
                block_pose_lst.append(deepcopy(block_pose))

            break

        def create_block(block_pose, half_size, color):
            return create_box(
                scene=self,
                pose=block_pose,
                half_size=(half_size, half_size, half_size),
                color=color,
                name="box",
            )

        self.blocks = []
        for i in range(self.num_blocks):
            block = create_block(block_pose_lst[i], self.halfsize_lst[i], color)
            self.blocks.append(block)
            setattr(self, f"block{i + 1}", block)

        for block in self.blocks:
            self.add_prohibit_area(block, padding=0.08)

        # 底座目标位置：小范围随机
        target_x = np.random.uniform(-0.03, 0.03)
        target_y = np.random.uniform(-0.15, -0.11)

        self.prohibited_area.append([
            target_x - 0.04,
            target_y - 0.02,
            target_x + 0.04,
            target_y + 0.02,
        ])

        self.base_target_pose = [
            target_x,
            target_y,
            0.74 + self.halfsize_lst[0] + self.table_z_bias,
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
        # for i in range(self.num_blocks):
        #     if i == 0:
        #         desc = "largest block"
        #     elif i == self.num_blocks - 1:
        #         desc = "smallest block"
        #     else:
        #         desc = f"block ranked {i + 1} by size"

        #     info_dict["{" + chr(ord("A") + i) + "}"] = desc
        #     info_dict["{" + chr(ord("a") + i) + "}"] = arm_tags[i]

        self.info["info"] = info_dict
        return self.info

    def pick_and_place_block(self, block: Actor):
        block_pose = block.get_pose().p
        arm_tag = ArmTag("left" if block_pose[0] < 0 else "right")

        if self.last_gripper is not None and self.last_gripper != arm_tag:
            self.move(
                self.grasp_actor(block, arm_tag=arm_tag, pre_grasp_dis=0.09),
                self.back_to_origin(arm_tag=arm_tag.opposite),
            )
        else:
            self.move(
                self.grasp_actor(block, arm_tag=arm_tag, pre_grasp_dis=0.09)
            )

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07))

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
                pre_dis=0.05,
                dis=0.0,
                pre_dis_axis="fp",
            )
        )

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07))

        self.last_gripper = arm_tag
        self.last_actor = block
        return str(arm_tag)

    def update_progress(self):
        eps_z = 0.02

        poses = [block.get_pose().p for block in self.blocks]

        base_z = poses[0][2]

        # 最底层默认直接成功
        self.task_success[0] = 1

        cur_z = base_z
        for i in range(1, self.num_blocks):
            cur_z += self.halfsize_lst[i - 1] + self.halfsize_lst[i]
            is_aligned = abs(poses[i][2] - cur_z) <= eps_z
            self.task_success[i] = 1 if is_aligned else 0

    def check_success(self):
        self.update_progress()
        return self.task_success == [1] * self.num_blocks