from ._base_task import Base_Task
from .utils import *

import os
import glob
import json
import numpy as np


class Pick_Mixed_Objects_Canonical_B_1(Base_Task):

    OBJECT_SPECS = [{'modelname': '021_cup', 'skip_model_ids': [10,11,12]}, 
                    {'modelname': '078_phonestand', 'skip_model_ids': [1,5]}, 
                    {'modelname': '071_can', 'skip_model_ids': []}, 
                    {'modelname': '080_pillbottle', 'skip_model_ids': []}, 
                    {'modelname': '005_french-fries', 'skip_model_ids': []}, 
                    {'modelname': '035_apple', 'skip_model_ids': []}, 
                    {'modelname': '107_soap', 'skip_model_ids': []}, 
                    {'modelname': '115_perfume', 'skip_model_ids': [2]}, 
                    {'modelname': '112_tea-box', 'skip_model_ids': []}, 
                    {'modelname': '113_coffee-box', 'skip_model_ids': []}, 
                    {'modelname': '105_sauce-can', 'skip_model_ids': []}, 
                    {'modelname': '075_bread', 'skip_model_ids': [1,2]}]

    def setup_demo(self, **kwargs):
        super()._init_task_env_(**kwargs)

    def _sample_model_id(self, spec):
        asset_path = os.path.join("assets/objects", spec["modelname"])
        json_files = glob.glob(os.path.join(asset_path, "model_data*.json"))

        model_ids = []
        for file in json_files:
            base = os.path.basename(file)
            try:
                idx = int(base.replace("model_data", "").replace(".json", ""))
            except ValueError:
                continue

            if idx in spec.get("skip_model_ids", []):
                continue
            model_ids.append(idx)

        if len(model_ids) == 0:
            raise ValueError(f"No available model_data*.json found for {spec['modelname']} after skipping ids")

        model_ids = sorted(model_ids)
        return int(np.random.choice(model_ids))

    def _sampled_specs(self):
        specs = []
        for spec in self.OBJECT_SPECS[:3]:
            spec = dict(spec)
            spec["model_id"] = self._sample_model_id(spec)
            specs.append(spec)
        return specs

    def load_actors(self):
        self.object_specs = self._sampled_specs()
        # 位置随机。
        np.random.shuffle(self.object_specs)
        self.padding = 0.02
        self.prohibit_area_range = [-0.26, -0.24, 0.26, -0.12]

        self.pre_grasp_dis = 0.10
        self.lift_z = 0.10
        self.target_lift_thresh = 0.04
        self.other_stable_thresh = 0.03
        self.use_contact_point = False

        center_x = np.random.uniform(-0.12, 0.08)
        center_y = np.random.uniform(-0.125, -0.01)
        col_gap = 0.156
        x1 = center_x - col_gap
        x2 = center_x
        x3 = center_x + col_gap
        xs = [x1, x2, x3]
        y_values = [center_y]

        x_jitter = 0.015
        y_jitter = 0.02
        def fixed_item_pose(x, y):
            return rand_pose(
                xlim=[x - x_jitter, x + x_jitter],
                ylim=[y - y_jitter, y + y_jitter],
                rotate_rand=False,
                qpos=[0.5, 0.5, 0.5, 0.5],
            )

        poses = []
        for y in y_values:
            for x in xs:
                poses.append(fixed_item_pose(x, y))

        def create_item(item_pose, spec):
            return create_actor(
                scene=self,
                pose=item_pose,
                modelname=spec["modelname"],
                convex=True,
                model_id=spec["model_id"],
            )

        self.items = [create_item(poses[i], self.object_specs[i]) for i in range(3)]
        self.item_grid = [
            self.items[0:3]
        ]

        for item in self.items:
            self.add_prohibit_area(item, padding=self.padding)
        self.prohibited_area.append(self.prohibit_area_range)

        self.delay(5)

        self.target_row = 0
        self.target_col = np.random.randint(3)
        self.target_item = self.item_grid[self.target_row][self.target_col]

        self.init_item_z = [item.get_pose().p[2] for item in self.items]
        self.target_index = self.target_col

        self.col_from_left_to_right = bool(np.random.randint(2))

    def play_once(self):
        self.last_gripper = None
        arm_tag = self.pick_target_item(self.target_item)

        if self.col_from_left_to_right:
            col_text = ['1st', '2nd', '3rd']
            col_dir_text = "from left to right"
        else:
            col_text = ['3rd', '2nd', '1st']
            col_dir_text = "from right to left"

        self.info["info"] = {
            "{C}": col_text[self.target_col],
            "{col_dir}": col_dir_text,
            "{a}": str(arm_tag),
        }

        return self.info

    def pick_target_item(self, item):
        item_pose = item.get_pose().p
        arm_tag = ArmTag("left" if item_pose[0] < 0 else "right")

        grasp_kwargs = {
            "arm_tag": arm_tag,
            "pre_grasp_dis": self.pre_grasp_dis,
        }

        if self.use_contact_point:
            grasp_kwargs["contact_point_id"] = 0 if str(arm_tag) == "right" else 2

        if self.last_gripper is not None and self.last_gripper != arm_tag:
            self.move(
                self.grasp_actor(item, **grasp_kwargs),
                self.back_to_origin(arm_tag=self.last_gripper),
            )
        else:
            self.move(
                self.grasp_actor(item, **grasp_kwargs)
            )

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=self.lift_z))
        self.last_gripper = arm_tag
        return str(arm_tag)

    def check_success(self):
        cur_z = [item.get_pose().p[2] for item in self.items]

        target_lifted = cur_z[self.target_index] > self.init_item_z[self.target_index] + self.target_lift_thresh

        other_stable = True
        for i in range(len(self.items)):
            if i == self.target_index:
                continue
            if abs(cur_z[i] - self.init_item_z[i]) > self.other_stable_thresh:
                other_stable = False
                break

        return target_lifted and other_stable
