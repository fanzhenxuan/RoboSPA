from ._base_task import Base_Task
from .utils import *
import numpy as np
from copy import deepcopy


class Stack_Blocks_Color_Order_2(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.num_blocks = 2

        # 固定尺寸，避免颜色任务再叠加尺寸难度
        self.block_half_size = 0.018

        # 进度记录：最底下方块默认已满足
        self.task_success = [1, 0]

        # 颜色池：名字 + RGB
        color_pool = [
            ("red", (1.0, 0.0, 0.0)),
            ("green", (0.0, 1.0, 0.0)),
            ("blue", (0.0, 0.0, 1.0)),
            ("yellow", (1.0, 1.0, 0.0)),
            ("purple", (0.6, 0.2, 0.8)),
        ]

        # 固定选 num_blocks 个不同颜色
        chosen_colors = [color_pool[i] for i in range(self.num_blocks)]

        # 固定堆叠顺序：bottom -> top
        self.stack_order_names = [chosen_colors[i][0] for i in range(self.num_blocks)]
        self.stack_order_rgbs = [chosen_colors[i][1] for i in range(self.num_blocks)]

        while True:
            block_pose_lst = []

            for i in range(self.num_blocks):
                block_pose = rand_pose(
                    xlim=[-0.28, 0.28],
                    ylim=[-0.08, 0.05],
                    zlim=[0.741 + self.block_half_size],
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
                #     # abs(block_pose.p[0]) < 0.05
                #     # or np.sum((block_pose.p[:2] - np.array([0, -0.1])) ** 2) < 0.0225
                #     # or
                #     not check_block_pose(block_pose)
                # ):
                #     block_pose = rand_pose(
                #         xlim=[-0.28, 0.28],
                #         ylim=[-0.08, 0.05],
                #         zlim=[0.741 + self.block_half_size],
                #         qpos=[1, 0, 0, 0],
                #         ylim_prop=True,
                #         rotate_rand=True,
                #         rotate_lim=[0, 0, 0.75],
                #     )
                
                max_trials = 100
                trials = 0
                
                while (
                    not check_block_pose(block_pose)
                ) and trials < max_trials:
                    block_pose = rand_pose(
                        xlim=[-0.28, 0.28],
                        ylim=[-0.08, 0.05],
                        zlim=[0.741 + self.block_half_size],
                        qpos=[1, 0, 0, 0],
                        ylim_prop=True,
                        rotate_rand=True,
                        rotate_lim=[0, 0, 0.75],
                    )
                    trials += 1
                
                if not check_block_pose(block_pose):
                    raise RuntimeError("Failed to sample a valid block_pose within 100 tries.")

                block_pose_lst.append(deepcopy(block_pose))

            break

        def create_block(block_pose, color, name):
            return create_box(
                scene=self,
                pose=block_pose,
                half_size=(self.block_half_size, self.block_half_size, self.block_half_size),
                color=color,
                name=name,
            )

        self.blocks = []
        for i in range(self.num_blocks):
            block = create_block(
                block_pose_lst[i],
                self.stack_order_rgbs[i],
                self.stack_order_names[i] + "_block",
            )
            self.blocks.append(block)
            setattr(self, f"block{i+1}", block)

        for block in self.blocks:
            self.add_prohibit_area(block, padding=0.08)

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
            0.74 + self.block_half_size + self.table_z_bias,
            0, 1, 0, 0
        ]

    def play_once(self):
        self.last_gripper = None
        self.last_actor = None

        for block in self.blocks:
            self.pick_and_place_block(block)

        self.info["info"] = {
            # "{A}": "red",
            # "{B}": "green",
        }
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
            self.move(self.grasp_actor(block, arm_tag=arm_tag, pre_grasp_dis=0.09))

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

        self.task_success[0] = 1

        for i in range(1, self.num_blocks):
            expected_z = base_z + i * (2 * self.block_half_size)
            is_aligned = abs(poses[i][2] - expected_z) <= eps_z
            self.task_success[i] = 1 if is_aligned else 0

    def check_success(self):
        self.update_progress()
        return (self.task_success == [1, 1] and self.robot.is_left_gripper_open() and self.robot.is_right_gripper_open())