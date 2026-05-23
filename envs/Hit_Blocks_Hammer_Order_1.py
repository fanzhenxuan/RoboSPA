from ._base_task import Base_Task
from .utils import *
import sapien
import numpy as np
from ._GLOBAL_CONFIGS import *


class Hit_Blocks_Hammer_Order_1(Base_Task):
    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.task_success = [0]
        self.stage_sum = 1
        self.stage = 0

        # 创建锤子
        self.hammer = create_actor(
            scene=self,
            pose=sapien.Pose([0, -0.06, 0.783], [0, 0, 0.995, 0.105]),
            modelname="020_hammer",
            convex=True,
            model_id=0,
        )

        # ===== 创建一个方块 =====
        def sample_block_pose():
            return rand_pose(
                xlim=[-0.25, 0.25],
                ylim=[-0.25, 0.25],
                zlim=[0.76],
                qpos=[1, 0, 0, 0],
                rotate_rand=True,
                rotate_lim=[0, 0, 0.5],
            )

        def valid_pose(pose):
            if abs(pose.p[0]) < 0.05:
                return False
            if np.sum(pow(pose.p[:2], 2)) < 0.001:
                return False
            return True

        p = sample_block_pose()
        
        # while not valid_pose(p):
        #     p = sample_block_pose()
        
        max_trials = 100
        trials = 0
        
        while not valid_pose(p) and trials < max_trials:
            p = sample_block_pose()
            trials += 1
        
        if not valid_pose(p):
            raise RuntimeError("Failed to sample a valid block pose within 100 tries.")

        
        self.block = create_box(
            scene=self,
            pose=p,
            half_size=(0.025, 0.025, 0.025),
            color=(1, 0, 0),
            name="box1",
            is_static=True,
        )

        # 锤子变轻
        self.hammer.set_mass(0.001)

        # 禁止区域
        self.add_prohibit_area(self.hammer, padding=0.10)
        self.prohibited_area.append([
            p.p[0] - 0.05,
            p.p[1] - 0.05,
            p.p[0] + 0.05,
            p.p[1] + 0.05,
        ])

        self._hit_block = False

    # def update_progress(self):
    #     hit_now = self.check_actors_contact(self.hammer.get_name(), self.block.get_name())
    #     if hit_now:
    #         self._hit_block = True
    #         if self.stage < self.stage_sum:
    #             self.stage = 1

    def update_progress(self):
        hammer_target_pose = self.hammer.get_functional_point(0, "pose").p
        block_pose = self.block.get_functional_point(1, "pose").p
        eps = np.array([0.02, 0.02])

        pos_match = np.all(np.abs(hammer_target_pose[:2] - block_pose[:2]) < eps)
        hit_now = pos_match and self.check_actors_contact(
            self.hammer.get_name(),
            self.block.get_name()
        )

        if hit_now:
            self._hit_block = True
            if self.stage < self.stage_sum:
                self.stage = 1

    def play_once(self):
        block_pose = self.block.get_functional_point(0, "pose").p
        arm_tag = ArmTag("left" if block_pose[0] < 0 else "right")

        self.move(self.grasp_actor(self.hammer, arm_tag=arm_tag, pre_grasp_dis=0.12, grasp_dis=0.01))
        self.move(self.move_by_displacement(arm_tag, z=0.07, move_axis="arm"))

        self.move(
            self.place_actor(
                self.hammer,
                target_pose=self.block.get_functional_point(1, "pose"),
                arm_tag=arm_tag,
                functional_point_id=0,
                pre_dis=0.06,
                dis=0,
                is_open=False,
            )
        )

        self.update_progress()

        self.info["info"] = {
            "{A}": "020_hammer/base0",
            # "{B}": str(self.stage_sum),
            "{a}": str(arm_tag),
        }

        return self.info

    def check_success(self):
        for i in range(self.stage_sum):
            self.task_success[i] = int(self.stage >= i + 1)
        return self.stage >= self.stage_sum