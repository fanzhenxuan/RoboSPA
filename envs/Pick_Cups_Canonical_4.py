from ._base_task import Base_Task
from .utils import *

import os
import glob
import json
import numpy as np


class Pick_Cups_Canonical_4(Base_Task):

    def setup_demo(self, **kwargs):
        super()._init_task_env_(**kwargs)

    def _sample_model_id(self, config):
        asset_path = os.path.join("assets/objects", config["modelname"])
        json_files = glob.glob(os.path.join(asset_path, "model_data*.json"))

        model_ids = []
        stable_model_ids = []
        for file in json_files:
            base = os.path.basename(file)
            try:
                idx = int(base.replace("model_data", "").replace(".json", ""))
            except ValueError:
                continue

            if idx in config.get("skip_model_ids", []):
                continue

            model_ids.append(idx)

            try:
                with open(file, "r") as f:
                    model_meta = json.load(f)
                if bool(model_meta.get("stable", False)):
                    stable_model_ids.append(idx)
            except Exception:
                pass

        if len(model_ids) == 0:
            raise ValueError(f"No available model_data*.json found for {config['modelname']}")

        model_ids = sorted(model_ids)
        stable_model_ids = sorted(stable_model_ids)

        if config.get("use_stable_only_if_available", False) and len(stable_model_ids) > 0:
            candidate_model_ids = stable_model_ids
        else:
            candidate_model_ids = model_ids

        preferred_model_id = config.get("preferred_model_id", None)
        if preferred_model_id is not None and preferred_model_id in candidate_model_ids:
            return int(preferred_model_id)

        return int(np.random.choice(candidate_model_ids))

    def load_actors(self):
        self.object_config = {
            "modelname": "021_cup",
            "padding": 0.02,
            "prohibit_area_range": [-0.26, -0.24, 0.26, -0.12],
            "pre_grasp_dis": 0.10,
            "lift_z": 0.10,
            "target_lift_thresh": 0.04,
            "other_stable_thresh": 0.03,
            "use_contact_point": True,
            "preferred_model_id": None,
            "skip_model_ids": [],
            "use_stable_only_if_available": False,
        }

        self.modelname = self.object_config["modelname"]
        self.padding = self.object_config["padding"]
        self.prohibit_area_range = self.object_config["prohibit_area_range"]

        self.pre_grasp_dis = self.object_config["pre_grasp_dis"]
        self.lift_z = self.object_config["lift_z"]
        self.target_lift_thresh = self.object_config["target_lift_thresh"]
        self.other_stable_thresh = self.object_config["other_stable_thresh"]
        self.use_contact_point = self.object_config["use_contact_point"]

        self.model_id = self._sample_model_id(self.object_config)
        asset_path = os.path.join("assets/objects", self.modelname)

        center_x = np.random.uniform(-0.06, 0.00)
        center_y = np.random.uniform(-0.13, -0.1)
        col_gap = 0.15
        row_gap = 0.11
        x1 = center_x - col_gap
        x2 = center_x
        x3 = center_x + col_gap
        xs = [x1, x2, x3]
        y1 = center_y - row_gap
        y2 = center_y
        y3 = center_y + row_gap
        y_values = [y1, y2, y3]

        model_meta_path = os.path.join(asset_path, f"model_data{self.model_id}.json")
        with open(model_meta_path, "r") as f:
            model_meta = json.load(f)

        extents = np.array(model_meta.get("extents", [1.0, 1.0, 1.0]), dtype=float)
        scale = np.array(model_meta.get("scale", [1.0, 1.0, 1.0]), dtype=float)
        obj_size = np.abs(extents * scale)
        obj_width = float(max(obj_size[0], 1e-3))
        obj_depth = float(max(obj_size[1], 1e-3))

        min_col_gap = float(col_gap)

        x_jitter = min(0.012, max(0.004, 0.18 * obj_width))
        x_jitter = min(x_jitter, 0.18 * min_col_gap)

        y_jitter = min(0.008, max(0.002, 0.10 * obj_depth))
        y_jitter = min(y_jitter, 0.12 * float(row_gap))
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

        def create_item(item_pose):
            return create_actor(
                scene=self,
                pose=item_pose,
                modelname=self.modelname,
                convex=True,
                model_id=self.model_id,
            )

        self.items = [create_item(pose) for pose in poses]
        self.item_grid = [
            self.items[0:3],
            self.items[3:6],
            self.items[6:9]
        ]

        for item in self.items:
            self.add_prohibit_area(item, padding=self.padding)
        self.prohibited_area.append(self.prohibit_area_range)

        self.delay(5)

        self.target_row = np.random.randint(3)
        self.target_col = np.random.randint(3)
        self.target_item = self.item_grid[self.target_row][self.target_col]

        self.init_item_z = [item.get_pose().p[2] for item in self.items]
        self.target_index = self.target_row * 3 + self.target_col

        self.row_from_far_to_near = bool(np.random.randint(2))
        self.col_from_left_to_right = bool(np.random.randint(2))

    def play_once(self):
        self.last_gripper = None
        arm_tag = self.pick_target_item(self.target_item)

        if self.row_from_far_to_near:
            row_text = ["1st row", "2nd row", "3rd row"]
            row_dir_text = "from near to far"
        else:
            row_text = ["3rd row", "2nd row", "1st row"]
            row_dir_text = "from far to near"

        if self.col_from_left_to_right:
            col_text = ['1st', '2nd', '3rd']
            col_dir_text = "from left to right"
        else:
            col_text = ['3rd', '2nd', '1st']
            col_dir_text = "from right to left"

        self.info["info"] = {
            "{R}": row_text[self.target_row],
            "{C}": col_text[self.target_col],
            "{row_dir}": row_dir_text,
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
