from ._base_task import Base_Task
from .utils import *
from copy import deepcopy
import numpy as np
import random

class color_memory_cover_base_zb_0409_v2(Base_Task):
    """
    任务逻辑：
    1. 桌面上生成 SHOW_NUM 个随机颜色 block
    2. 机器人 observe 2 秒
    3. 系统直接刷新 cover，把所有 block 盖住（不是机器人去盖）
    4. 机器人按随机颜色顺序揭开 cover
    5. 每次揭开后，把 cover 放到对应 block 前面的区域
    6. 在 info 中返回随机颜色顺序
    """

    SHOW_NUM = 1

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        if not (1 <= self.SHOW_NUM <= 5):
            raise ValueError(f"SHOW_NUM must be in [1, 5], got {self.SHOW_NUM}")

        self.get_obs_cnt = 0
        self.orig_left_endpose = self.get_arm_pose("left")
        self.orig_right_endpose = self.get_arm_pose("right")
        self.stage_sum = self.SHOW_NUM
        self.stage = 0
        self.progress = 0
        self.task_success = [0] * self.stage_sum
        self.fail_flag = False

        self.block_gripper = None
        self.covers_added = False

        self.blocks = []
        self.covers = []

        self.block_color_names = []
        self.block_color_values = []

        self.block_pose_lst = []
        self.close_cover_place_lst = []
        self.open_cover_place_lst = []

        # 参考你给的 cover_blocks
        self.quat_of_target_pose = [0.0, 1.0, 0.0, 0.0]

        self.COLOR_POOL = {
            "red": (1.0, 0.0, 0.0),
            "green": (0.0, 1.0, 0.0),
            "blue": (0.0, 0.0, 1.0),
            "yellow": (1.0, 1.0, 0.0),
            "cyan": (0.0, 1.0, 1.0),
            "magenta": (1.0, 0.0, 1.0),
            "orange": (1.0, 0.5, 0.0),
            "purple": (0.5, 0.0, 0.5),
            "white": (1.0, 1.0, 1.0),
            "black": (0.1, 0.1, 0.1),
        }

        if self.SHOW_NUM > len(self.COLOR_POOL):
            raise ValueError(
                f"Need {self.SHOW_NUM} unique colors, but COLOR_POOL only has {len(self.COLOR_POOL)}."
            )

        color_items = list(self.COLOR_POOL.items())
        np.random.shuffle(color_items)
        selected_items = color_items[:self.SHOW_NUM]

        self.block_color_names = [x[0] for x in selected_items]
        self.block_color_values = [x[1] for x in selected_items]

        self.open_order_idx = np.random.permutation(self.SHOW_NUM).tolist()
        self.open_order_color_names = [self.block_color_names[i] for i in self.open_order_idx]

        self.block_half_size = 0.02
        self.block_y = -0.18
        self.cover_open_offset_y = 0.13

        if self.SHOW_NUM == 1:
            # xs = [0.12] if random.random() < 0.5 else [-0.12]
            xs = [0]
        elif self.SHOW_NUM == 5:
            xs = np.linspace(-0.27, 0.23, self.SHOW_NUM).tolist()
        else:
            xs = np.linspace(-0.20, 0.20, self.SHOW_NUM).tolist()

        self.block_xs = xs
        self.cover_name = [f"cover_{i}" for i in range(self.SHOW_NUM)]

        def create_block(block_pose, color):
            return create_box(
                scene=self,
                pose=block_pose,
                half_size=(self.block_half_size, self.block_half_size, self.block_half_size),
                color=color,
                name="box",
                is_static=True,
            )

        for i in range(self.SHOW_NUM):
            block_pose = rand_pose(
                xlim=[xs[i], xs[i]],
                ylim=[self.block_y, self.block_y],
                zlim=[0.741 + self.block_half_size],
                qpos=[1, 0, 0, 0],
                rotate_rand=False,
            )
            self.block_pose_lst.append(deepcopy(block_pose))
            self.close_cover_place_lst.append([block_pose.p[0], block_pose.p[1]])
            self.open_cover_place_lst.append([block_pose.p[0], block_pose.p[1] + self.cover_open_offset_y])

        for i in range(self.SHOW_NUM):
            self.blocks.append(create_block(self.block_pose_lst[i], self.block_color_values[i]))

    def add_covers(self):
        if self.covers_added:
            return

        def create_cover(cover_pose):
            actor = create_actor(
                self,
                pose=cover_pose,
                modelname="003_cover",
                model_id=0,
                convex=True,
            )
            if actor is None:
                raise RuntimeError("Failed to load model '003_cover'.")
            return actor

        self.covers = []
        for i in range(self.SHOW_NUM):
            block_pose = self.blocks[i].get_pose().p
            cover_pose = rand_pose(
                xlim=[block_pose[0], block_pose[0]],
                ylim=[block_pose[1], block_pose[1]],
                zlim=[0.7418],
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=False,
            )
            self.covers.append(create_cover(cover_pose))

        self.covers_added = True

    def check_if_cover(self, idx):
        if not self.covers_added:
            return False

        cover_pose = self.covers[idx].get_pose().p
        block_pose = self.blocks[idx].get_pose().p

        return (
            np.abs(cover_pose[0] - block_pose[0]) < 0.03
            and np.abs(cover_pose[1] - block_pose[1]) < 0.03
            and cover_pose[2] < 0.745
        )

    def check_if_open(self, idx):
        if not self.covers_added:
            return False

        cover_pose = self.covers[idx].get_pose().p
        target_xy = self.open_cover_place_lst[idx]

        return (
            np.linalg.norm(cover_pose[:2] - target_xy) < 0.045
            and cover_pose[2] < 0.745
        )

    def cover_in_right_place(self):
        if not self.covers_added:
            return True

        for i in range(self.SHOW_NUM):
            cover_pose = self.covers[i].get_pose().p
            dist_xy_open = np.linalg.norm(cover_pose[:2] - self.open_cover_place_lst[i])
            dist_xy_close = np.linalg.norm(cover_pose[:2] - self.close_cover_place_lst[i])

            if min(dist_xy_open, dist_xy_close) > 0.035 or cover_pose[2] > 0.742:
                return False

        return True

    def update_progress(self):
        if self.fail_flag:
            return

        if not self.covers_added:
            self.stage = 0
            self.progress = 0
            for i in range(self.stage_sum):
                self.task_success[i] = 0
            return

        # 1. 已经完成的前缀必须保持 open
        for idx in self.open_order_idx[:self.stage]:
            if not self.check_if_open(idx):
                self.fail_flag = True
                return

        # 2. 如果已经全部完成，直接同步状态
        if self.stage >= self.stage_sum:
            self.progress = self.stage
            for i in range(self.stage_sum):
                self.task_success[i] = 1
            return

        # 3. 当前应该操作的目标
        cur_idx = self.open_order_idx[self.stage]

        # 4. 后面的目标不能提前打开
        for idx in self.open_order_idx[self.stage + 1:]:
            if self.check_if_open(idx):
                self.fail_flag = True
                return

        # 5. 当前目标如果已经稳定到 open 位置，则推进 stage
        if self.check_if_open(cur_idx):
            self.stage += 1

        self.progress = self.stage

        for i in range(self.stage_sum):
            self.task_success[i] = int(self.stage >= i + 1)


    def open_cover(self, block_idx):
        block_pose = self.blocks[block_idx].get_pose().p
        x, y = block_pose[0], block_pose[1]
        target_pose = [x, y + self.cover_open_offset_y, 0.741]

        arm_tag = ArmTag("left" if block_pose[0] < 0 else "right")
        # print("使用的胳膊是：", arm_tag)
        self.move(
            self.grasp_actor(
                self.covers[block_idx],
                arm_tag=arm_tag,
                pre_grasp_dis=0.05,
            )
        )
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.05))
        self.move(
            self.place_actor(
                self.covers[block_idx],
                target_pose=target_pose + self.quat_of_target_pose,
                arm_tag=arm_tag,
                functional_point_id=0,
                pre_dis=0.05,
                dis=0.005,
            )
        )
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.03))

        self.block_gripper = arm_tag

    def play_once(self):
        self.delay(delay_time=2, save_freq=-1)

        self.add_covers()
        self.update_progress()

        self.delay(delay_time=1, save_freq=-1)

        for i in self.open_order_idx:
            arm_tag = ArmTag("left" if self.blocks[i].get_pose().p[0] < 0 else "right")
            if self.block_gripper is not None and self.block_gripper != arm_tag:
                self.move(self.back_to_origin(arm_tag=self.block_gripper))
            self.open_cover(i)
            self.update_progress()
        self.move(self.open_gripper(arm_tag=self.block_gripper))
        self.get_obs_cnt = 10000
        colors = ", ".join(self.open_order_color_names)
        self.info["info"] = {
            "{A}": colors,
        }
        return self.info

    def check_success(self):
        if self.fail_flag:
            return False
        self.update_progress()
        self.get_obs_cnt += 1
        if self.get_obs_cnt == 500:
            self.add_covers()
        elif self.get_obs_cnt < 500:
            current_left_endpose = self.get_arm_pose("left")
            current_right_endpose = self.get_arm_pose("right")
            if np.linalg.norm(np.array(current_left_endpose[:3]) - np.array(self.orig_left_endpose[:3])) > 0.03 or \
               np.linalg.norm(np.array(current_right_endpose[:3]) - np.array(self.orig_right_endpose[:3])) > 0.03:
                print("Arm position deviation detected!")
                self.fail_flag = True
            return False
        # return (all(x == 1 for x in self.task_success) and self.robot.is_left_gripper_open() and self.robot.is_right_gripper_open())
        return all(x ==1 for x in self.task_success)