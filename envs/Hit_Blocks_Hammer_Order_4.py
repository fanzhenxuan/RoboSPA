from ._base_task import Base_Task
from .utils import *
import sapien
import numpy as np
from ._GLOBAL_CONFIGS import *


class Hit_Blocks_Hammer_Order_4(Base_Task):
    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.stage_sum = 4
        self.stage = 0
        self.task_success = [0, 0, 0, 0]
        self.hammer = create_actor(
            scene=self,
            pose=sapien.Pose([0, -0.06, 0.783], [0, 0, 0.995, 0.105]),
            modelname="020_hammer",
            convex=True,
            model_id=0,
        )

        # ====== 创建四个方块（红/绿/蓝/黄） ======
        def sample_block_pose(xlim=None):
            return rand_pose(
                xlim=xlim if xlim is not None else [-0.25, 0.25],
                ylim=[-0.25, 0.25],
                zlim=[0.76],
                qpos=[1, 0, 0, 0],
                rotate_rand=True,
                rotate_lim=[0, 0, 0.5],
            )

        def valid_pose(pose, existing_poses, min_dist=0.12):
            if abs(pose.p[0]) < 0.05 or np.sum(pow(pose.p[:2], 2)) < 0.001:
                return False
            for ep in existing_poses:
                if np.linalg.norm(pose.p[:2] - ep.p[:2]) < min_dist:
                    return False
            return True

        poses = []

        p0 = sample_block_pose()
        
        # while not valid_pose(p0, poses, min_dist=0.12):
        #     p0 = sample_block_pose()
        
        max_trials = 100
        trials = 0
        
        while not valid_pose(p0, poses, min_dist=0.12) and trials < max_trials:
            p0 = sample_block_pose()
            trials += 1
        
        if not valid_pose(p0, poses, min_dist=0.12):
            raise RuntimeError("Failed to sample a valid p0 within 100 tries.")
        
        poses.append(p0)
        
        # --- 根据第一个方块的x，确定后续方块的x约束 ---
        # 规则：
        # if x<0.1 => others x<0.1
        # if x>-0.1 => others x>-0.1
        x_low = -0.25
        x_high = 0.25
        if poses[0].p[0] < 0:
            x_high = min(x_high, 0.1)
            constrained_xlim = [-0.25, 0.1]
        else:
            x_low = max(x_low, -0.1)
            constrained_xlim = [-0.1, 0.25]
        # print("第一个创建成功~")
        
        # --- 生成剩下三个方块（带x约束，其它不变） ---
        while len(poses) < 4:
            # print(len(poses))
            # print(constrained_xlim)
            p = sample_block_pose(xlim=constrained_xlim)
        
            # while not valid_pose(p, poses, min_dist=0.12):
            #     p = sample_block_pose(xlim=constrained_xlim)
        
            max_trials = 100
            trials = 0
        
            while not valid_pose(p, poses, min_dist=0.12) and trials < max_trials:
                p = sample_block_pose(xlim=constrained_xlim)
                trials += 1
        
            if not valid_pose(p, poses, min_dist=0.12):
                raise RuntimeError("Failed to sample a valid block pose within 100 tries.")
        
            poses.append(p)

        self.blocks = []
        self.blocks.append(
            create_box(
                scene=self,
                pose=poses[0],
                half_size=(0.025, 0.025, 0.025),
                color=(1, 0, 0),
                name="box1",
                is_static=True,
            )
        )
        self.blocks.append(
            create_box(
                scene=self,
                pose=poses[1],
                half_size=(0.025, 0.025, 0.025),
                color=(0, 1, 0),
                name="box2",
                is_static=True,
            )
        )
        self.blocks.append(
            create_box(
                scene=self,
                pose=poses[2],
                half_size=(0.025, 0.025, 0.025),
                color=(0, 0, 1),
                name="box3",
                is_static=True,
            )
        )
        self.blocks.append(
            create_box(
                scene=self,
                pose=poses[3],
                half_size=(0.025, 0.025, 0.025),
                color=(1, 1, 0),
                name="box4",
                is_static=True,
            )
        )

        self.hammer.set_mass(0.001)

        self.add_prohibit_area(self.hammer, padding=0.10)
        for bp in poses:
            self.prohibited_area.append([
                bp.p[0] - 0.05,
                bp.p[1] - 0.05,
                bp.p[0] + 0.05,
                bp.p[1] + 0.05,
            ])

        self._hit_blocks = [False, False, False, False]

    # def update_progress(self):
    #     idx = min(self.stage, self.stage_sum - 1)
    #     target_block = self.blocks[idx]
    #     hit_now = self.check_actors_contact(self.hammer.get_name(), target_block.get_name())
    #     if hit_now:
    #         self._hit_blocks[idx] = True
    #         if self.stage < self.stage_sum:
    #             self.stage += 1

    def update_progress(self):
        idx = min(self.stage, self.stage_sum - 1)  # 0->block1, 1->block2
        target_block = self.blocks[idx]

        hammer_target_pose = self.hammer.get_functional_point(0, "pose").p
        block_pose = target_block.get_functional_point(1, "pose").p
        eps = np.array([0.02, 0.02])

        pos_match = np.all(np.abs(hammer_target_pose[:2] - block_pose[:2]) < eps)

        hit_now = pos_match and self.check_actors_contact(
            self.hammer.get_name(),
            target_block.get_name()
        )

        if hit_now:
            self._hit_blocks[idx] = True
            if self.stage < self.stage_sum:
                self.stage += 1

    def play_once(self):
        block1_pose = self.blocks[0].get_functional_point(0, "pose").p
        arm_tag = ArmTag("left" if block1_pose[0] < 0 else "right")

        self.move(self.grasp_actor(self.hammer, arm_tag=arm_tag, pre_grasp_dis=0.12, grasp_dis=0.01))
        self.move(self.move_by_displacement(arm_tag, z=0.07, move_axis="arm"))

        for i in range(4):
            self.move(
                self.place_actor(
                    self.hammer,
                    target_pose=self.blocks[i].get_functional_point(1, "pose"),
                    arm_tag=arm_tag,
                    functional_point_id=0,
                    pre_dis=0.06,
                    dis=0,
                    is_open=False,
                )
            )
            self.update_progress()
            if i != 3:
                self.move(self.move_by_displacement(arm_tag, z=0.05, move_axis="arm"))

        self.info["info"] = {"{A}": "020_hammer/base0", "{a}": str(arm_tag)}
        return self.info

    def check_success(self):
        for i in range(self.stage_sum):
            self.task_success[i] = int(self.stage >= i + 1)
        return self.stage >= self.stage_sum