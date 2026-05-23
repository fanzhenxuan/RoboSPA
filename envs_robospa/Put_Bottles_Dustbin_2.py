from ._base_task import Base_Task
from .utils import *
import sapien
import numpy as np
from copy import deepcopy


class Put_Bottles_Dustbin_2(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(table_xy_bias=[0.3, 0], **kwags)

    def load_actors(self):
        pose_lst = []
        self.bottle_num = 2

        self.stage = 0
        self.task_success = [0] * self.bottle_num

        def create_bottle(model_id):
            bottle_pose = rand_pose(
                xlim=[-0.25, 0.3],
                ylim=[0.03, 0.23],
                rotate_rand=False,
                rotate_lim=[0, 1, 0],
                qpos=[0.707, 0.707, 0, 0],
            )
            tag = True
            gen_lim = 100
            i = 1
            # while tag and i < gen_lim:
            #     tag = False
            #     if np.abs(bottle_pose.p[0]) < 0.05:
            #         tag = True
            #     for pose in pose_lst:
            #         if np.sum(np.power(np.array(pose[:2]) - np.array(bottle_pose.p[:2]), 2)) < 0.0169:
            #             tag = True
            #             break
            #     if tag:
            #         i += 1
            #         bottle_pose = rand_pose(
            #             xlim=[-0.25, 0.3],
            #             ylim=[0.03, 0.23],
            #             rotate_rand=False,
            #             rotate_lim=[0, 1, 0],
            #             qpos=[0.707, 0.707, 0, 0],
            #         )
            
            max_trials = 100
            trials = 0
            
            while tag and trials < max_trials:
                tag = False
                if np.abs(bottle_pose.p[0]) < 0.05:
                    tag = True
                for pose in pose_lst:
                    if np.sum(np.power(np.array(pose[:2]) - np.array(bottle_pose.p[:2]), 2)) < 0.0169:
                        tag = True
                        break
                if tag:
                    trials += 1
                    bottle_pose = rand_pose(
                        xlim=[-0.25, 0.3],
                        ylim=[0.03, 0.23],
                        rotate_rand=False,
                        rotate_lim=[0, 1, 0],
                        qpos=[0.707, 0.707, 0, 0],
                    )
            
            if tag:
                raise RuntimeError("Failed to sample a valid bottle_pose within 100 tries.")
            pose_lst.append(bottle_pose.p[:2])
            bottle = create_actor(
                self,
                bottle_pose,
                modelname="114_bottle",
                convex=True,
                model_id=model_id,
            )
            return bottle

        self.bottles = []
        self.bottle_model_map = {}

        available_ids = [1, 2, 3, 4]
        self.stage_sum = self.bottle_num
        self.bottle_id = list(np.random.choice(available_ids, self.bottle_num, replace=False))

        for i in range(self.bottle_num):
            bottle = create_bottle(self.bottle_id[i])
            self.bottles.append(bottle)
            self.bottle_model_map[bottle] = self.bottle_id[i]
            self.add_prohibit_area(bottle, padding=0.1)

        self.dustbin = create_actor(
            self.scene,
            pose=sapien.Pose([-0.45, 0, 0], [0.5, 0.5, 0.5, 0.5]),
            modelname="011_dustbin",
            convex=True,
            is_static=True,
        )
        self.delay(2)
        self.right_middle_pose = [0, 0.0, 0.88, 0, 1, 0, 0]

    def is_bottle_in_dustbin(self, bottle):
        target_pose = np.array([-0.45, 0])
        eps = np.array([0.221, 0.325])

        bottle_pose = bottle.get_pose().p
        return (
            np.all(np.abs(bottle_pose[:2] - target_pose) < eps)
            and 0.2 < bottle_pose[2] < 0.7
        )

    def update_progress(self):
        self.task_success = [
            1 if self.is_bottle_in_dustbin(bottle) else 0
            for bottle in self.bottles
        ]
        self.stage = sum(self.task_success)
        return self.stage

    def play_once(self):
        bottle_lst = sorted(
            self.bottles,
            key=lambda x: [x.get_pose().p[0] > 0, x.get_pose().p[1]]
        )

        for i in range(self.bottle_num):
            bottle = bottle_lst[i]

            arm_tag = "left" if bottle.get_pose().p[0] < 0 else "right"
            delta_dis = 0.06

            left_end_action = Action(
                "left",
                "move",
                [-0.35, -0.1, 0.93, 0.65, -0.25, 0.25, 0.65]
            )

            if arm_tag == "left":
                self.move(self.grasp_actor(bottle, arm_tag=arm_tag, pre_grasp_dis=0.1))
                self.update_progress()

                self.move(self.move_by_displacement(arm_tag, z=0.1))
                self.update_progress()

                self.move((ArmTag("left"), [left_end_action]))
                self.update_progress()

            else:
                right_action = self.grasp_actor(bottle, arm_tag=arm_tag, pre_grasp_dis=0.1)
                if len(right_action[1]) >= 2:
                    right_action[1][0].target_pose[2] += delta_dis
                    right_action[1][1].target_pose[2] += delta_dis
                self.move(right_action, self.back_to_origin("left"))
                self.update_progress()

                self.move(self.move_by_displacement(arm_tag, z=0.1))
                self.update_progress()

                self.move(
                    self.place_actor(
                        bottle,
                        target_pose=self.right_middle_pose,
                        arm_tag=arm_tag,
                        functional_point_id=0,
                        pre_dis=0.0,
                        dis=0.0,
                        is_open=False,
                        constrain="align",
                    )
                )
                self.update_progress()

                left_action = self.grasp_actor(bottle, arm_tag="left", pre_grasp_dis=0.1)
                if len(left_action[1]) >= 2:
                    left_action[1][0].target_pose[2] -= delta_dis
                    left_action[1][1].target_pose[2] -= delta_dis
                self.move(left_action)
                self.update_progress()

                self.move(self.open_gripper(ArmTag("right")))
                self.update_progress()

                self.move((ArmTag("left"), [left_end_action]), self.back_to_origin("right"))
                self.update_progress()

            self.move(self.open_gripper("left"))
            self.update_progress()

        self.info["info"] = {
            # "{A}": f"114_bottle/base{self.bottle_model_map[bottle_lst[0]]}",
            # "{B}": f"114_bottle/base{self.bottle_model_map[bottle_lst[1]]}",
            "{A}": f"011_dustbin/base0",
        }
        return self.info

    def stage_reward(self):
        self.update_progress()
        return self.stage / self.stage_sum

    def check_success(self):
        self.update_progress()
        return all(self.task_success)