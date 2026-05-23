from copy import deepcopy
from ._base_task import Base_Task
from .utils import *
import sapien
import math
import glob
import numpy as np
import os


class Place_Object_Scale_Click_1(Base_Task):

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

        # 单阶段：
        # 1. 主物体放到电子秤上
        self.stage_sum = 1
        self.task_success = [0]

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
        # 3. prohibit areas
        # ==================================================
        self.add_prohibit_area(self.object, padding=0.05)
        self.add_prohibit_area(self.scale, padding=0.05)

    def update_progress(self):
        # ==================================================
        # 主物体在 scale 上
        # ==================================================
        object_pose = self.object.get_pose().p
        scale_pose = self.scale.get_functional_point(0)
        scale_distance = np.linalg.norm(np.array(scale_pose[:2]) - np.array(object_pose[:2]))
        scale_done = int(
            scale_distance < 0.035
            and object_pose[2] > (scale_pose[2] - 0.01)
        )

        self.task_success[0] = scale_done

    def play_once(self):
        self.last_gripper = None

        # ==================================================
        # Stage 1: 放主物体到 scale
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

        self.info["info"] = {
            "{A}": f"{self.selected_modelname}/base{self.selected_model_id}",
            "{B}": f"072_electronicscale/base{self.scale_id}",
            "{a}": str(self.arm_tag),
        }
        return self.info

    def check_success(self):
        self.update_progress()
        return self.task_success == [1]