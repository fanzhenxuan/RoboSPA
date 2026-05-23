from ._base_task import Base_Task
from .utils import *
import sapien
import numpy as np


class Place_Bottle_Cup_1(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(table_xy_bias=[0.3, 0], **kwags)

    def load_actors(self):
        self.stage_sum = 1
        self.stage_num = 0
        self.task_success = [0]

        pose_lst = []

        def check_pose_valid(cur_pose, min_dist2=0.012):
            for pose in pose_lst:
                if np.sum((np.array(pose[:2]) - np.array(cur_pose.p[:2])) ** 2) < min_dist2:
                    return False
            return True

        def sample_pose(xlim, ylim, qpos, rotate_rand=False, rotate_lim=[0, 0, 0], min_dist2=0.012):
            pose = rand_pose(
                xlim=xlim,
                ylim=ylim,
                qpos=qpos,
                rotate_rand=rotate_rand,
                rotate_lim=rotate_lim,
            )
            # while not check_pose_valid(pose, min_dist2=min_dist2):
            #     pose = rand_pose(
            #         xlim=xlim,
            #         ylim=ylim,
            #         qpos=qpos,
            #         rotate_rand=rotate_rand,
            #         rotate_lim=rotate_lim,
            #     )
        
            max_trials = 100
            trials = 0
        
            while not check_pose_valid(pose, min_dist2=min_dist2) and trials < max_trials:
                pose = rand_pose(
                    xlim=xlim,
                    ylim=ylim,
                    qpos=qpos,
                    rotate_rand=rotate_rand,
                    rotate_lim=rotate_lim,
                )
                trials += 1
        
            if not check_pose_valid(pose, min_dist2=min_dist2):
                raise RuntimeError("Failed to sample a valid pose within 100 tries.")
            pose_lst.append(pose.p[:2])
            return pose

        # -------------------------
        # 1) 一个瓶子
        # -------------------------
        bottle_pose = sample_pose(
            xlim=[-0.28, -0.12],
            ylim=[0.05, 0.22],
            qpos=[0.707, 0.707, 0, 0],
            rotate_rand=False,
            rotate_lim=[0, 1, 0],
            min_dist2=0.016,
        )

        self.bottle_id = int(np.random.choice([1, 2, 3]))
        self.bottle = create_actor(
            self,
            bottle_pose,
            modelname="114_bottle",
            convex=True,
            model_id=self.bottle_id,
        )

        # -------------------------
        # 2) 垃圾桶
        # -------------------------
        self.dustbin = create_actor(
            self.scene,
            pose=sapien.Pose([-0.45, 0, 0], [0.5, 0.5, 0.5, 0.5]),
            modelname="011_dustbin",
            convex=True,
            is_static=True,
        )

        self.add_prohibit_area(self.bottle, padding=0.08)

        self.delay(2)

    def play_once(self):
        self.stage_num = 0
        self.task_success = [0]

        arm_info_1 = self.throw_bottle(self.bottle)
        self.update_progress()

        self.info["info"] = {
            "{A}": "011_dustbin/base0",
            "{B}": f"114_bottle/base{self.bottle_id}",
            "{a}": arm_info_1,
        }
        return self.info

    def throw_bottle(self, bottle):
        arm_tag = ArmTag("left")
        end_action = Action(
            "left", "move",
            [-0.35, -0.1, 0.93, 0.65, -0.25, 0.25, 0.65]
        )

        self.move(self.grasp_actor(bottle, arm_tag=arm_tag, pre_grasp_dis=0.1))
        self.move(self.move_by_displacement(arm_tag, z=0.1))
        self.move((arm_tag, [end_action]))
        self.move(self.open_gripper("left"))

        return "left"

    def bottle_in_dustbin(self, bottle):
        target_pose = np.array([-0.45, 0])
        eps = np.array([0.221, 0.325])
        pose = bottle.get_pose().p
        return (
            np.all(np.abs(pose[:2] - target_pose) < eps)
            and pose[2] > 0.2
            and pose[2] < 0.7
        )

    def update_progress(self):
        if self.bottle_in_dustbin(self.bottle):
            self.task_success[0] = 1

        self.stage_num = sum(self.task_success)

    def check_success(self):
        self.update_progress()
        return (
            self.task_success == [1]
            and self.is_left_gripper_open()
            and self.is_right_gripper_open()
        )