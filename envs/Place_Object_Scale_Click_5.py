from copy import deepcopy
from ._base_task import Base_Task
from .utils import *
import sapien
import math
import glob
import numpy as np
import os


class Place_Object_Scale_Click_5(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        def get_available_model_ids(modelname):
            asset_path = os.path.join("assets/objects", modelname)
            json_files = glob.glob(os.path.join(asset_path, "model_data*.json"))

            available_ids = []
            for file in json_files:
                base = os.path.basename(file)
                try:
                    idx = int(base.replace("model_data", "").replace(".json", ""))
                    available_ids.append(idx)
                except ValueError:
                    continue

            return available_ids

        # 前两阶段 + playingcards 一次点击
        self.click_stage_sum = 1
        self.stage_sum = 5
        self.stage = 0
        self.has_left_press_area = True
        self.task_success = [0, 0, 0, 0, 0]

        # ==================================================
        # 1. 左侧主任务物体
        # ==================================================
        rand_pos = rand_pose(
            xlim=[-0.25, 0],
            ylim=[-0.2, 0.05],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=True,
            rotate_lim=[0, 3.14, 0],
        )
        # while abs(rand_pos.p[0]) < 0.02:
        #     rand_pos = rand_pose(
        #         xlim=[-0.25, 0],
        #         ylim=[-0.2, 0.05],
        #         qpos=[0.5, 0.5, 0.5, 0.5],
        #         rotate_rand=True,
        #         rotate_lim=[0, 3.14, 0],
        #     )
        
        max_trials = 100
        trials = 0
        
        while abs(rand_pos.p[0]) < 0.02 and trials < max_trials:
            rand_pos = rand_pose(
                xlim=[-0.25, 0],
                ylim=[-0.2, 0.05],
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=True,
                rotate_lim=[0, 3.14, 0],
            )
            trials += 1
        
        if abs(rand_pos.p[0]) < 0.02:
            raise RuntimeError("Failed to sample a valid rand_pos within 100 tries.")

        object_list = ["050_bell"]
        self.selected_modelname = np.random.choice(object_list)

        available_model_ids = get_available_model_ids(self.selected_modelname)
        if not available_model_ids:
            raise ValueError(f"No available model_data.json files found for {self.selected_modelname}")

        self.selected_model_id = np.random.choice(available_model_ids)

        self.object = create_actor(
            scene=self,
            pose=rand_pos,
            modelname=self.selected_modelname,
            convex=True,
            model_id=self.selected_model_id,
        )
        self.object.set_mass(0.05)
        
        self.object_init_pose = sapien.Pose(
            self.object.get_pose().p.copy(),
            self.object.get_pose().q.copy(),
        )
        
        # ==================================================
        # 2. 左侧电子秤（目标）
        # ==================================================
        if rand_pos.p[0] > 0:
            xlim = [0.02, 0.25]
        else:
            xlim = [-0.25, -0.02]

        target_rand_pose = rand_pose(
            xlim=xlim,
            ylim=[-0.2, -0.05],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=True,
            rotate_lim=[0, 3.14, 0],
        )
        # while np.sqrt(
        #     (target_rand_pose.p[0] - rand_pos.p[0]) ** 2
        #     + (target_rand_pose.p[1] - rand_pos.p[1]) ** 2
        # ) < 0.15:
        #     target_rand_pose = rand_pose(
        #         xlim=xlim,
        #         ylim=[-0.2, -0.05],
        #         qpos=[0.5, 0.5, 0.5, 0.5],
        #         rotate_rand=True,
        #         rotate_lim=[0, 3.14, 0],
        #     )
        
        max_trials = 100
        trials = 0
        
        while np.sqrt(
            (target_rand_pose.p[0] - rand_pos.p[0]) ** 2
            + (target_rand_pose.p[1] - rand_pos.p[1]) ** 2
        ) < 0.15 and trials < max_trials:
            target_rand_pose = rand_pose(
                xlim=xlim,
                ylim=[-0.2, -0.05],
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=True,
                rotate_lim=[0, 3.14, 0],
            )
            trials += 1
        
        if np.sqrt(
            (target_rand_pose.p[0] - rand_pos.p[0]) ** 2
            + (target_rand_pose.p[1] - rand_pos.p[1]) ** 2
        ) < 0.15:
            raise RuntimeError("Failed to sample a valid target_rand_pose within 100 tries.")

        self.scale_id = np.random.choice([0, 1, 5, 6], 1)[0]

        self.scale = create_actor(
            scene=self,
            pose=target_rand_pose,
            modelname="072_electronicscale",
            model_id=self.scale_id,
            convex=True,
            is_static=True,
        )
        self.scale.set_mass(0.05)

        # ==================================================
        # 3. 右侧额外添加一个 displaystand 和一个物体（干扰项）
        # ==================================================
        distractor_object_list = [
            "047_mouse",
            "048_stapler",
            "073_rubikscube",
            "057_toycar",
            "079_remotecontrol",
        ]

        right_stand_pose = rand_pose(
            xlim=[0.10, 0.20],
            ylim=[-0.18, -0.08],
            qpos=[0.707, 0.707, 0.0, 0.0],
            rotate_rand=True,
            rotate_lim=[0, np.pi / 6, 0],
        )

        self.displaystand_id = np.random.choice([0, 1, 2, 3, 4], 1)[0]
        self.displaystand = create_actor(
            scene=self,
            pose=right_stand_pose,
            modelname="074_displaystand",
            convex=True,
            model_id=self.displaystand_id,
            is_static=True,
        )
        self.displaystand.set_mass(0.01)

        right_object_pose = rand_pose(
            xlim=[0.12, 0.28],
            ylim=[-0.02, 0.10],
            qpos=[0.707, 0.707, 0.0, 0.0],
            rotate_rand=True,
            rotate_lim=[0, np.pi / 3, 0],
        )

        # while (
        #     (right_object_pose.p[0] - right_stand_pose.p[0]) ** 2
        #     + (right_object_pose.p[1] - right_stand_pose.p[1]) ** 2
        # ) < 0.01:
        #     right_object_pose = rand_pose(
        #         xlim=[0.12, 0.28],
        #         ylim=[-0.02, 0.10],
        #         qpos=[0.707, 0.707, 0.0, 0.0],
        #         rotate_rand=True,
        #         rotate_lim=[0, np.pi / 3, 0],
        #     )
        
        max_trials = 100
        trials = 0
        
        while (
            (right_object_pose.p[0] - right_stand_pose.p[0]) ** 2
            + (right_object_pose.p[1] - right_stand_pose.p[1]) ** 2
        ) < 0.01 and trials < max_trials:
            right_object_pose = rand_pose(
                xlim=[0.12, 0.28],
                ylim=[-0.02, 0.10],
                qpos=[0.707, 0.707, 0.0, 0.0],
                rotate_rand=True,
                rotate_lim=[0, np.pi / 3, 0],
            )
            trials += 1
        
        if (
            (right_object_pose.p[0] - right_stand_pose.p[0]) ** 2
            + (right_object_pose.p[1] - right_stand_pose.p[1]) ** 2
        ) < 0.01:
            raise RuntimeError("Failed to sample a valid right_object_pose within 100 tries.")

        self.right_object_modelname = np.random.choice(distractor_object_list)
        right_available_ids = get_available_model_ids(self.right_object_modelname)
        if not right_available_ids:
            raise ValueError(
                f"No available model_data.json files found for {self.right_object_modelname}"
            )
        self.right_object_model_id = np.random.choice(right_available_ids)

        self.right_object = create_actor(
            scene=self,
            pose=right_object_pose,
            modelname=self.right_object_modelname,
            convex=True,
            model_id=self.right_object_model_id,
        )
        self.right_object.set_mass(0.01)


        self.right_object_init_pose = sapien.Pose(
            self.right_object.get_pose().p.copy(),
            self.right_object.get_pose().q.copy(),
        )

        # ==================================================
        # 4. playingcards（点击物体）
        # ==================================================
        playingcard_pose = rand_pose(
            xlim=[-0.25, 0.25],
            ylim=[-0.2, 0.0],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=True,
            rotate_lim=[0, 3.14, 0],
        )

        def far_enough_xy(p0, p1, min_dist=0.14):
            return np.linalg.norm(np.array(p0[:2]) - np.array(p1[:2])) >= min_dist

        # while (
        #     abs(playingcard_pose.p[0]) < 0.05
        #     or (not far_enough_xy(playingcard_pose.p, rand_pos.p, min_dist=0.14))
        #     or (not far_enough_xy(playingcard_pose.p, target_rand_pose.p, min_dist=0.14))
        #     or (not far_enough_xy(playingcard_pose.p, right_stand_pose.p, min_dist=0.14))
        #     or (not far_enough_xy(playingcard_pose.p, right_object_pose.p, min_dist=0.14))
        # ):
        #     playingcard_pose = rand_pose(
        #         xlim=[-0.25, 0.25],
        #         ylim=[-0.2, 0.0],
        #         qpos=[0.5, 0.5, 0.5, 0.5],
        #         rotate_rand=True,
        #         rotate_lim=[0, 3.14, 0],
        #     )
        
        max_trials = 100
        trials = 0
        
        while (
            abs(playingcard_pose.p[0]) < 0.05
            or (not far_enough_xy(playingcard_pose.p, rand_pos.p, min_dist=0.14))
            or (not far_enough_xy(playingcard_pose.p, target_rand_pose.p, min_dist=0.14))
            or (not far_enough_xy(playingcard_pose.p, right_stand_pose.p, min_dist=0.14))
            or (not far_enough_xy(playingcard_pose.p, right_object_pose.p, min_dist=0.14))
        ) and trials < max_trials:
            playingcard_pose = rand_pose(
                xlim=[-0.25, 0.25],
                ylim=[-0.2, 0.0],
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=True,
                rotate_lim=[0, 3.14, 0],
            )
            trials += 1
        
        if (
            abs(playingcard_pose.p[0]) < 0.05
            or (not far_enough_xy(playingcard_pose.p, rand_pos.p, min_dist=0.14))
            or (not far_enough_xy(playingcard_pose.p, target_rand_pose.p, min_dist=0.14))
            or (not far_enough_xy(playingcard_pose.p, right_stand_pose.p, min_dist=0.14))
            or (not far_enough_xy(playingcard_pose.p, right_object_pose.p, min_dist=0.14))
        ):
            raise RuntimeError("Failed to sample a valid playingcard_pose within 100 tries.")

        self.playingcards_id = np.random.choice([0, 1, 2], 1)[0]
        self.playingcards = create_actor(
            scene=self,
            pose=playingcard_pose,
            modelname="081_playingcards",
            convex=True,
            model_id=self.playingcards_id,
            is_static=True,
        )

        self.check_arm_function = (
            self.is_left_gripper_close
            if self.playingcards.get_pose().p[0] < 0
            else self.is_right_gripper_close
        )

        self.add_prohibit_area(self.object, padding=0.05)
        self.add_prohibit_area(self.scale, padding=0.05)
        self.add_prohibit_area(self.displaystand, padding=0.05)
        self.add_prohibit_area(self.right_object, padding=0.05)

    def _is_click_success(self):
        if not self.check_arm_function():
            return False

        playingcard_pose = self.playingcards.get_contact_point(0)[:3]
        positions = self.get_gripper_actor_contact_position("081_playingcards")
        eps = [0.03, 0.03]

        for position in positions:
            if (
                np.all(np.abs(position[:2] - playingcard_pose[:2]) < eps)
                and abs(position[2] - playingcard_pose[2]) < 0.03
            ):
                return True
        return False

    def _has_left_click_area(self):
        playingcard_pose = self.playingcards.get_contact_point(0)[:3]
        positions = self.get_gripper_actor_contact_position("081_playingcards")

        if len(positions) == 0:
            return True

        for position in positions:
            if (
                np.all(np.abs(position[:2] - playingcard_pose[:2]) < np.array([0.03, 0.03]))
                and abs(position[2] - playingcard_pose[2]) < 0.03
            ):
                return False

        return True

    def update_progress(self):
        object_pose = self.object.get_pose().p
        scale_pose = self.scale.get_functional_point(0)
        scale_distance = np.linalg.norm(np.array(scale_pose[:2]) - np.array(object_pose[:2]))
        scale_done = int(
            scale_distance < 0.035
            and object_pose[2] > (scale_pose[2] - 0.01)
        )

        right_object_pose = self.right_object.get_pose().p
        displaystand_pose = self.displaystand.get_functional_point(0)
        stand_distance = np.linalg.norm(
            np.array(displaystand_pose[:2]) - np.array(right_object_pose[:2])
        )
        stand_done = int(
            stand_distance < 0.035
            and right_object_pose[2] > (displaystand_pose[2] - 0.01)
        )

        bell_init_z = self.object_init_pose.p[2]
        bell_back_done = int(abs(object_pose[2] - bell_init_z) < 0.005)

        right_init_z = self.right_object_init_pose.p[2]
        right_back_done = int(abs(right_object_pose[2] - right_init_z) < 0.005)

        if self.stage < self.click_stage_sum:
            if self._has_left_click_area():
                self.has_left_press_area = True

            if self.has_left_press_area and self._is_click_success():
                self.stage += 1
                self.has_left_press_area = False

        if self.task_success[0] == 0:
            self.task_success[0] = scale_done

        if self.task_success[1] == 0:
            self.task_success[1] = stand_done

        for i in range(self.click_stage_sum):
            self.task_success[i + 2] = int(self.stage >= i + 1)

        if self.task_success[0] == 1:
            self.task_success[3] = bell_back_done

        if self.task_success[1] == 1:
            self.task_success[4] = right_back_done

    def _do_one_click(self, arm_tag):
        self.move(
            self.grasp_actor(
                self.playingcards,
                arm_tag=arm_tag,
                pre_grasp_dis=0.1,
                grasp_dis=0.02,
                contact_point_id=0,
            )
        )

        self.move(self.move_by_displacement(arm_tag, z=-0.05))
        self.update_progress()

        self.move(self.move_by_displacement(arm_tag, z=0.05))
        self.update_progress()

    def play_once(self):
        self.last_gripper = None

        # ==================================================
        # 1. 把 bell 放到 scale 上
        # ==================================================
        self.arm_tag = ArmTag("right" if self.object.get_pose().p[0] > 0 else "left")

        self.move(self.grasp_actor(self.object, arm_tag=self.arm_tag))
        self.move(self.move_by_displacement(arm_tag=self.arm_tag, z=0.15))
        self.move(
            self.place_actor(
                self.object,
                arm_tag=self.arm_tag,
                target_pose=self.scale.get_functional_point(0),
                constrain="free",
                pre_dis=0.05,
                dis=0.005,
            )
        )
        self.move(self.move_by_displacement(arm_tag=self.arm_tag, z=0.05))
        self.update_progress()
        self.last_gripper = self.arm_tag

        # ==================================================
        # 2. 把右侧物体放到 displaystand 上
        # ==================================================
        self.right_arm_tag = ArmTag("right" if self.right_object.get_pose().p[0] > 0 else "left")

        if self.last_gripper is not None and self.last_gripper != self.right_arm_tag:
            self.move(
                self.grasp_actor(
                    self.right_object,
                    arm_tag=self.right_arm_tag,
                    pre_grasp_dis=0.1,
                ),
                self.back_to_origin(arm_tag=self.right_arm_tag.opposite),
            )
        else:
            self.move(
                self.grasp_actor(
                    self.right_object,
                    arm_tag=self.right_arm_tag,
                    pre_grasp_dis=0.1,
                )
            )

        self.move(self.move_by_displacement(arm_tag=self.right_arm_tag, z=0.08))
        self.move(
            self.place_actor(
                self.right_object,
                arm_tag=self.right_arm_tag,
                target_pose=self.displaystand.get_functional_point(0),
                constrain="free",
                pre_dis=0.07,
                dis=0.005,
            )
        )
        self.update_progress()
        self.last_gripper = self.right_arm_tag

        # ==================================================
        # 3. click playingcards
        # ==================================================
        self.card_arm_tag = ArmTag("right" if self.playingcards.get_pose().p[0] > 0 else "left")

        if self.last_gripper is not None and self.last_gripper != self.card_arm_tag:
            self.move(
                self.close_gripper(arm_tag=self.card_arm_tag, pos=0),
                self.back_to_origin(arm_tag=self.card_arm_tag.opposite),
            )
        else:
            self.move(
                self.close_gripper(arm_tag=self.card_arm_tag, pos=0)
            )

        for _ in range(self.click_stage_sum):
            self._do_one_click(self.card_arm_tag)

        self.move(self.open_gripper(self.card_arm_tag))

        self.last_gripper = self.card_arm_tag

        # ==================================================
        # 4. 把 bell 从 scale 上拿下来，放回原来位置
        # ==================================================
        self.bell_back_arm_tag = ArmTag("right" if self.object.get_pose().p[0] > 0 else "left")

        if self.last_gripper is not None and self.last_gripper != self.bell_back_arm_tag:
            self.move(
                self.grasp_actor(
                    self.object,
                    arm_tag=self.bell_back_arm_tag,
                    pre_grasp_dis=0.1,
                ),
                self.back_to_origin(arm_tag=self.bell_back_arm_tag.opposite),
            )
        else:
            self.move(
                self.grasp_actor(
                    self.object,
                    arm_tag=self.bell_back_arm_tag,
                    pre_grasp_dis=0.1,
                )
            )

        # 先抬起来
        self.move(self.move_by_displacement(arm_tag=self.bell_back_arm_tag, z=0.10))

        # 平移到原始 xy
        target_xy = self.object_init_pose.p[:2]
        cur_xy = self.object.get_pose().p[:2]
        delta_xy = np.array(target_xy) - np.array(cur_xy)

        self.move(
            self.move_by_displacement(
                self.bell_back_arm_tag,
                x=float(delta_xy[0]),
                y=float(delta_xy[1]),
            )
        )

        # 放下
        place_down_z = -0.12
        self.move(
            self.move_by_displacement(
                self.bell_back_arm_tag,
                z=place_down_z,
            )
        )

        # 松手
        self.move(self.open_gripper(self.bell_back_arm_tag))

        # 稍微抬一点，避免碰撞
        self.move(self.move_by_displacement(arm_tag=self.bell_back_arm_tag, z=0.07))

        self.update_progress()
        self.last_gripper = self.bell_back_arm_tag


        # ==================================================
        # 5. 把 stand 上的物体拿下来，放回原来位置
        # ==================================================
        self.right_back_arm_tag = ArmTag("right" if self.right_object.get_pose().p[0] > 0 else "left")

        if self.last_gripper is not None and self.last_gripper != self.right_back_arm_tag:
            self.move(
                self.grasp_actor(
                    self.right_object,
                    arm_tag=self.right_back_arm_tag,
                    pre_grasp_dis=0.1,
                ),
                self.back_to_origin(arm_tag=self.right_back_arm_tag.opposite),
            )
        else:
            self.move(
                self.grasp_actor(
                    self.right_object,
                    arm_tag=self.right_back_arm_tag,
                    pre_grasp_dis=0.1,
                )
            )

        # 先抬起来
        self.move(self.move_by_displacement(arm_tag=self.right_back_arm_tag, z=0.10))

        # 平移到原始 xy
        target_xy = self.right_object_init_pose.p[:2]
        cur_xy = self.right_object.get_pose().p[:2]
        delta_xy = np.array(target_xy) - np.array(cur_xy)

        self.move(
            self.move_by_displacement(
                self.right_back_arm_tag,
                x=float(delta_xy[0]),
                y=float(delta_xy[1]),
            )
        )

        # 放下
        place_down_z = -0.15
        self.move(
            self.move_by_displacement(
                self.right_back_arm_tag,
                z=place_down_z,
            )
        )

        # 松手
        self.move(self.open_gripper(self.right_back_arm_tag))

        # 稍微抬一点，避免碰撞
        self.move(self.move_by_displacement(arm_tag=self.right_back_arm_tag, z=0.07))

        self.update_progress()
        self.last_gripper = self.right_back_arm_tag

        self.info["info"] = {
            "{A}": f"{self.selected_modelname}/base{self.selected_model_id}",
            "{B}": f"072_electronicscale/base{self.scale_id}",
            "{C}": f"{self.right_object_modelname}/base{self.right_object_model_id}",
            "{D}": f"074_displaystand/base{self.displaystand_id}",
            "{E}": f"081_playingcards/base{self.playingcards_id}",
            "{a}": str(self.arm_tag),
            "{b}": str(self.right_arm_tag),
            "{c}": str(self.card_arm_tag),
            "{d}": str(self.bell_back_arm_tag),
            "{e}": str(self.right_back_arm_tag),
        }
        return self.info

    def check_success(self):
        self.update_progress()
        # print(self.task_success)
        # return True
        return self.task_success == [1, 1, 1, 1, 1]