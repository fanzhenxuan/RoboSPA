from copy import deepcopy
from ._base_task import Base_Task
from .utils import *
import sapien
import numpy as np


class ObserveBlocksMoveMemoryBase(Base_Task):
    """
    Task:
    1. 参考平台上展示 SHOW_NUM 个颜色 block
    2. observe 2 秒
    3. 挡住参考平台
    4. 桌面上随机放 2 * SHOW_NUM 个颜色 block
       - 其中 SHOW_NUM 个颜色属于展示集合（target）
       - 另外 SHOW_NUM 个颜色属于干扰集合（distractor）
    5. 机器人把 target blocks 全部移动到靠近机器人区域
    6. 不要求颜色顺序；执行时按桌面从左到右扫描 target，再从左到右放入 goal

    成功条件：
    - 所有 target blocks 都在 near region
    - 所有 distractor blocks 都不在 near region
    """

    SHOW_NUM = 1

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        # =========================
        # 基础状态
        # =========================
        self.stage_sum = self.SHOW_NUM
        self.stage = 0
        self.progress = 0
        self.task_success = [0] * self.stage_sum
        self.get_obs_cnt = 0
        self.fail_flag = False
        self.wall = None
        self.pre_arm = None
        self.orig_left_endpose = self.get_arm_pose("left")
        self.orig_right_endpose = self.get_arm_pose("right")
        self.ref_blocks = []
        self.table_blocks = []
        self.target_blocks = []
        self.distractor_blocks = []

        self.ref_color_names = []
        self.ref_color_values = []

        self.table_color_names = []
        self.table_color_values = []

        self.target_color_names = []
        self.target_color_values = []

        self.distractor_color_names = []
        self.distractor_color_values = []

        # =========================
        # 颜色池
        # =========================
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
            "brown": (0.6, 0.3, 0.1),
            "pink": (1.0, 0.75, 0.8),
        }

        if 2 * self.SHOW_NUM > len(self.COLOR_POOL):
            raise ValueError(
                f"Need {2 * self.SHOW_NUM} unique colors, but COLOR_POOL only has {len(self.COLOR_POOL)}."
            )

        color_items = list(self.COLOR_POOL.items())
        np.random.shuffle(color_items)

        # 前 SHOW_NUM 个颜色作为 target 展示集合
        selected_target_items = color_items[:self.SHOW_NUM]
        # 后 SHOW_NUM 个颜色作为 distractor
        selected_distractor_items = color_items[self.SHOW_NUM: 2 * self.SHOW_NUM]

        self.target_color_names = [x[0] for x in selected_target_items]
        self.target_color_values = [x[1] for x in selected_target_items]

        self.distractor_color_names = [x[0] for x in selected_distractor_items]
        self.distractor_color_values = [x[1] for x in selected_distractor_items]

        # 展台显示 target 集合，但任务不要求按颜色顺序回放
        self.ref_color_names = deepcopy(self.target_color_names)
        self.ref_color_values = deepcopy(self.target_color_values)

        # =========================
        # 参考平台（整体上移）
        # =========================
        self.ref_platform_y = 0.20
        self.wall_y = 0.12

        self.ref_platform = create_box(
            self.scene,
            sapien.Pose(p=[0, self.ref_platform_y, 0.82]),
            half_size=[0.26, 0.055, 0.01],
            color=(0.85, 0.85, 0.85),
            name="ref_platform",
            is_static=True,
        )

        # =========================
        # 方块尺寸
        # =========================
        ref_half_size = (0.015, 0.015, 0.015)
        table_half_size = (0.015, 0.015, 0.015)

        # =========================
        # 参考平台上放 SHOW_NUM 个 block
        # =========================
        if self.SHOW_NUM == 1:
            ref_xs = [0.0]
        else:
            ref_xs = np.linspace(-0.18, 0.18, self.SHOW_NUM).tolist()

        for i in range(self.SHOW_NUM):
            ref_block = create_box(
                scene=self,
                pose=sapien.Pose(
                    p=[ref_xs[i], self.ref_platform_y, 0.82 + ref_half_size[2] + 0.01],
                    q=[1, 0, 0, 0],
                ),
                half_size=ref_half_size,
                color=self.ref_color_values[i],
                name=f"ref_block_{i}",
                is_static=True,
            )
            self.ref_blocks.append(ref_block)

        # =========================
        # near region 定义
        # =========================
        self.near_region_y_min = -100.0
        self.near_region_y_max = -0.12
        self.near_region_x_min = -0.25
        self.near_region_x_max = 0.25

        # =========================
        # goal 从左到右排列
        # 无顺序任务：只是按空间从左到右占坑
        # =========================
        if self.SHOW_NUM == 1:
            goal_xs = [0.0]
        else:
            goal_xs = np.linspace(-0.10, 0.10, self.SHOW_NUM).tolist()

        goal_y = -0.18

        self.goal_poses = []
        for x in goal_xs:
            goal_pose = [
                x,
                goal_y,
                0.741 + table_half_size[2] + self.table_z_bias,
                0,
                1,
                0,
                0,
            ]
            self.goal_poses.append(goal_pose)

        # =========================
        # 桌面 block 位置采样
        # =========================
        def sample_table_pose(existing_poses, max_trials=500):
            for _ in range(max_trials):
                pose = rand_pose(
                    xlim=[-0.25, 0.25],
                    ylim=[-0.10, 0.08],
                    zlim=[0.741],
                    qpos=[1, 0, 0, 0],
                    rotate_rand=False,
                )

                # 不放太靠中间，避免双臂冲突
                if abs(pose.p[0]) < 0.04:
                    continue

                # 不放进 near region
                if self._is_in_near_region_xy(pose.p[:2], margin=0.02):
                    continue

                # 不要太靠近墙
                if abs(pose.p[1] - self.wall_y) < 0.03:
                    continue

                too_close = False
                for old_pose in existing_poses:
                    if np.linalg.norm(pose.p[:2] - old_pose.p[:2]) < 0.06:
                        too_close = True
                        break

                if not too_close:
                    return pose

            raise RuntimeError("Failed to sample valid table pose for blocks.")

        # =========================
        # 桌面上先放 target blocks
        # =========================
        existing_poses = []

        for i in range(self.SHOW_NUM):
            pose = sample_table_pose(existing_poses)

            block_pose = sapien.Pose(
                p=[pose.p[0], pose.p[1], 0.741 + table_half_size[2]],
                q=[1, 0, 0, 0],
            )

            block = create_box(
                scene=self,
                pose=block_pose,
                half_size=table_half_size,
                color=self.target_color_values[i],
                name=f"target_block_{i}",
                is_static=False,
            )
            existing_poses.append(block_pose)

            self.table_blocks.append(block)
            self.target_blocks.append(block)

            self.table_color_names.append(self.target_color_names[i])
            self.table_color_values.append(self.target_color_values[i])

        # =========================
        # 桌面上再放 distractor blocks
        # =========================
        for i in range(self.SHOW_NUM):
            pose = sample_table_pose(existing_poses)

            block_pose = sapien.Pose(
                p=[pose.p[0], pose.p[1], 0.741 + table_half_size[2]],
                q=[1, 0, 0, 0],
            )

            block = create_box(
                scene=self,
                pose=block_pose,
                half_size=table_half_size,
                color=self.distractor_color_values[i],
                name=f"distractor_block_{i}",
                is_static=False,
            )
            existing_poses.append(block_pose)

            self.table_blocks.append(block)
            self.distractor_blocks.append(block)

            self.table_color_names.append(self.distractor_color_names[i])
            self.table_color_values.append(self.distractor_color_values[i])

        # =========================
        # prohibit area
        # =========================
        for block in self.table_blocks:
            self.add_prohibit_area(block, padding=0.03)

    def add_wall(self):
        if self.wall is not None:
            return
        self.wall = create_box(
            self.scene,
            sapien.Pose(p=[0, self.wall_y, 0.95]),
            half_size=[0.34, 0.005, 0.2],
            color=(1, 0.9, 0.9),
            name="wall",
            is_static=True,
        )

    # -------------------------
    # region checks
    # -------------------------
    def _is_in_near_region_xy(self, xy, margin=0.0):
        return (
            (self.near_region_x_min - margin) < xy[0] < (self.near_region_x_max + margin)
            and (self.near_region_y_min - margin) < xy[1] < (self.near_region_y_max + margin)
        )

    def _is_block_in_near_region(self, block, margin=0.01):
        pos = block.get_pose().p
        return self._is_in_near_region_xy(pos[:2], margin=margin)

    # -------------------------
    # progress
    # -------------------------
    def update_progress(self):
        if self.fail_flag:
            return

        for block in self.distractor_blocks:
            if self._is_block_in_near_region(block, margin=0.01):
                self.fail_flag = True
                return

        progress = 0
        for block in self.target_blocks:
            if self._is_block_in_near_region(block, margin=0.01):
                progress += 1

        self.progress = progress
        self.stage = progress

        for i in range(self.stage_sum):
            self.task_success[i] = int(self.progress >= i + 1)

    # -------------------------
    # pick and place
    # -------------------------
    def pick_and_place_block(self, block, target_pose=None):
        block_pose = block.get_pose().p
        arm_tag = ArmTag("left" if block_pose[0] < 0 else "right")

        if self.pre_arm is not None and self.pre_arm != arm_tag:
            self.move(
                self.grasp_actor(block, arm_tag=arm_tag, pre_grasp_dis=0.09),
                self.back_to_origin(arm_tag=self.pre_arm),
            )
        else:
            self.move(self.grasp_actor(block, arm_tag=arm_tag, pre_grasp_dis=0.09))

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07))

        self.move(
            self.place_actor(
                block,
                target_pose=target_pose,
                arm_tag=arm_tag,
                functional_point_id=0,
                pre_dis=0.09,
                dis=0.02,
                constrain="align",
            )
        )

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07, move_axis="arm"))
        self.pre_arm = arm_tag
        return str(arm_tag)

    # -------------------------
    # main episode
    # -------------------------
    def play_once(self):
        # observe 2 秒
        self.delay(delay_time=2, save_freq=-1)

        # 挡住参考平台
        self.add_wall()

        # 稍停一下
        self.delay(delay_time=1, save_freq=-1)

        self.stage = 0
        self.progress = 0

        for i, block in enumerate(self.target_blocks):
            self.pick_and_place_block(block, self.goal_poses[i])
            self.update_progress()
        self.move(self.open_gripper(arm_tag=self.pre_arm))
        self.get_obs_cnt = 10000
        if not hasattr(self, "info") or self.info is None:
            self.info = {}

        self.info["info"] = {}
        return self.info

    def check_success(self):
        if self.fail_flag:
            return False
        self.update_progress()
        print(f"task_success: {self.task_success}, stage: {self.stage}/{self.stage_sum}")
        self.get_obs_cnt += 1
        if self.get_obs_cnt == 500:
            self.add_wall()
        elif self.get_obs_cnt < 500:
            current_left_endpose = self.get_arm_pose("left")
            current_right_endpose = self.get_arm_pose("right")
            if np.linalg.norm(np.array(current_left_endpose[:3]) - np.array(self.orig_left_endpose[:3])) > 0.03 or \
               np.linalg.norm(np.array(current_right_endpose[:3]) - np.array(self.orig_right_endpose[:3])) > 0.03:
                print("Arm position deviation detected!")
                self.fail_flag = True
            return False
        return (all(x == 1 for x in self.task_success) and self.robot.is_left_gripper_open() and self.robot.is_right_gripper_open())