from ._base_task import Base_Task
from .utils import *
import sapien
import numpy as np


class Place_Bottle_Cup_3(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(table_xy_bias=[0.3, 0], **kwags)

    def load_actors(self):
        self.stage_sum = 3
        self.stage_num = 0
        self.task_success = [0, 0, 0]

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
        # 1) 三个瓶子：左1 右2
        # -------------------------
        left_bottle_pose = sample_pose(
            xlim=[-0.28, -0.12],
            ylim=[0.05, 0.22],
            qpos=[0.707, 0.707, 0, 0],
            rotate_rand=False,
            rotate_lim=[0, 1, 0],
            min_dist2=0.016,
        )
        right_bottle_pose_1 = sample_pose(
            xlim=[0.08, 0.18],
            ylim=[0.05, 0.22],
            qpos=[0.707, 0.707, 0, 0],
            rotate_rand=False,
            rotate_lim=[0, 1, 0],
            min_dist2=0.016,
        )
        right_bottle_pose_2 = sample_pose(
            xlim=[0.18, 0.30],
            ylim=[0.05, 0.22],
            qpos=[0.707, 0.707, 0, 0],
            rotate_rand=False,
            rotate_lim=[0, 1, 0],
            min_dist2=0.016,
        )

        bottle_ids = np.random.choice([1, 2, 3], size=3, replace=True)
        self.bottle_ids = [int(x) for x in bottle_ids]

        self.left_bottle = create_actor(
            self,
            left_bottle_pose,
            modelname="114_bottle",
            convex=True,
            model_id=self.bottle_ids[0],
        )
        self.right_bottle_1 = create_actor(
            self,
            right_bottle_pose_1,
            modelname="114_bottle",
            convex=True,
            model_id=self.bottle_ids[1],
        )
        self.right_bottle_2 = create_actor(
            self,
            right_bottle_pose_2,
            modelname="114_bottle",
            convex=True,
            model_id=self.bottle_ids[2],
        )
        self.bottles = [self.left_bottle, self.right_bottle_1, self.right_bottle_2]

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
        self.right_middle_pose = [0, 0.0, 0.88, 0, 1, 0, 0]

        for bottle in self.bottles:
            self.add_prohibit_area(bottle, padding=0.08)

        self.delay(2)

    def play_once(self):
        self.stage_num = 0
        self.task_success = [0, 0, 0]

        arm_info_1 = self.throw_left_bottle(self.left_bottle)
        self.update_progress()
        self.move(self.back_to_origin("left"), self.back_to_origin("right"))

        arm_info_2 = self.throw_right_bottle_bimanual(self.right_bottle_1)
        self.update_progress()
        self.move(self.back_to_origin("left"), self.back_to_origin("right"))

        arm_info_3 = self.throw_right_bottle_bimanual(self.right_bottle_2)
        self.update_progress()

        self.info["info"] = {
            # "{A}": f"114_bottle/base{self.bottle_ids[0]}",
            # "{B}": f"114_bottle/base{self.bottle_ids[1]}",
            # "{C}": f"114_bottle/base{self.bottle_ids[2]}",
            "{A}": "011_dustbin/base0",
            "{a}": arm_info_1,
            "{b}": arm_info_2,
            "{c}": arm_info_3,
        }
        return self.info

    def throw_left_bottle(self, bottle):
        arm_tag = ArmTag("left")
        left_end_action = Action(
            "left", "move",
            [-0.35, -0.1, 0.93, 0.65, -0.25, 0.25, 0.65]
        )

        self.move(self.grasp_actor(bottle, arm_tag=arm_tag, pre_grasp_dis=0.1))
        self.move(self.move_by_displacement(arm_tag, z=0.1))
        self.move((ArmTag("left"), [left_end_action]))
        self.move(self.open_gripper("left"))

        return "left"

    def throw_right_bottle_bimanual(self, bottle):
        delta_dis = 0.06
        left_end_action = Action(
            "left", "move",
            [-0.35, -0.1, 0.93, 0.65, -0.25, 0.25, 0.65]
        )

        right_action = self.grasp_actor(bottle, arm_tag="right", pre_grasp_dis=0.1)

        if isinstance(right_action, (list, tuple)) and len(right_action) > 1:
            for act in right_action[1]:
                if hasattr(act, "target_pose") and act.target_pose is not None:
                    act.target_pose[2] += delta_dis

        self.move(right_action, self.back_to_origin("left"))
        self.move(self.move_by_displacement("right", z=0.1))

        self.move(
            self.place_actor(
                bottle,
                target_pose=self.right_middle_pose,
                arm_tag="right",
                functional_point_id=0,
                pre_dis=0.0,
                dis=0.0,
                is_open=False,
                constrain="align",
            )
        )

        left_action = self.grasp_actor(bottle, arm_tag="left", pre_grasp_dis=0.1)

        if isinstance(left_action, (list, tuple)) and len(left_action) > 1:
            for act in left_action[1]:
                if hasattr(act, "target_pose") and act.target_pose is not None:
                    act.target_pose[2] -= delta_dis

        self.move(left_action)

        self.move(self.open_gripper("right"))
        self.move((ArmTag("left"), [left_end_action]), self.back_to_origin("right"))
        self.move(self.open_gripper("left"))

        return "left and right"

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
        if self.bottle_in_dustbin(self.left_bottle):
            self.task_success[0] = 1

        if self.bottle_in_dustbin(self.right_bottle_1):
            self.task_success[1] = 1

        if self.bottle_in_dustbin(self.right_bottle_2):
            self.task_success[2] = 1

        self.stage_num = sum(self.task_success)

    def check_success(self):
        self.update_progress()
        return (
            self.task_success == [1, 1, 1]
            and self.is_left_gripper_open()
            and self.is_right_gripper_open()
        )