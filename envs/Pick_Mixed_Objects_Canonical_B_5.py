from ._base_task import Base_Task
from .utils import *

import os
import glob
import json
import numpy as np


class Pick_Mixed_Objects_Canonical_B_5(Base_Task):

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
        for spec in self.OBJECT_SPECS[:12]:
            spec = dict(spec)
            spec["model_id"] = self._sample_model_id(spec)
            specs.append(spec)
        return specs

    def load_actors(self):
            self.object_specs = self._sampled_specs()
            # Shuffle positions at random
            np.random.shuffle(self.object_specs)
            
            self.padding = 0.02
            self.prohibit_area_range = [-0.26, -0.24, 0.26, -0.12]

            self.pre_grasp_dis = 0.10
            self.lift_z = 0.10
            self.target_lift_thresh = 0.04
            self.other_stable_thresh = 0.03
            self.use_contact_point = False

            # Compute grid-layout parameters
            center_x = np.random.uniform(-0.04, -0.03)
            center_y = np.random.uniform(-0.105, -0.10)
            col_gap = 0.156
            row_gap = 0.122

            x_list = [
                center_x - 1.5 * col_gap, 
                center_x - 0.5 * col_gap, 
                center_x + 0.5 * col_gap, 
                center_x + 1.5 * col_gap
            ]
            y_list = [
                center_y - row_gap, 
                center_y, 
                center_y + row_gap
            ]

            x_jitter = 0.003
            y_jitter = 0.003

            # Updated pose sampler: supports a custom qpos
            def fixed_item_pose(x, y, qpos):
                return rand_pose(
                    xlim=[x - x_jitter, x + x_jitter],
                    ylim=[y - y_jitter, y + y_jitter],
                    rotate_rand=False,
                    qpos=qpos,
                )

            # Pre-generate 12 grid points
            poses_coords = []
            for y in y_list:
                for x in x_list:
                    poses_coords.append((x, y))

            self.items = []
            for i in range(12):
                spec = self.object_specs[i]
                x, y = poses_coords[i]
                
                # --- Key change: lying-flat pose for fries ---
                if spec['modelname'] == '005_french-fries':
                    # [1, 0, 0, 0] is the default YCB orientation, usually lying flat on the table
                    target_qpos = [1, 0, 0, 0]
                else:
                    # Other objects (cans, cups, etc.) keep the original standing pose
                    target_qpos = [0.5, 0.5, 0.5, 0.5]
                
                item_pose = fixed_item_pose(x, y, qpos=target_qpos)
                
                item = create_actor(
                    scene=self,
                    pose=item_pose,
                    modelname=spec["modelname"],
                    convex=True,
                    model_id=spec["model_id"],
                )
                self.items.append(item)

            # Organize into a 3x4 logical grid
            self.item_grid = [
                [self.items[0], self.items[1], self.items[2], self.items[3]],
                [self.items[4], self.items[5], self.items[6], self.items[7]],
                [self.items[8], self.items[9], self.items[10], self.items[11]],
            ]

            # Set forbidden regions (collision avoidance)
            for item in self.items:
                self.add_prohibit_area(item, padding=self.padding)
            self.prohibited_area.append(self.prohibit_area_range)

            # Wait for the physics sim to settle
            self.delay(5)

            # Randomly choose the target object
            self.target_row = np.random.randint(3)
            self.target_col = np.random.randint(4)
            self.target_item = self.item_grid[self.target_row][self.target_col]

            # Record initial height for later check_success
            self.init_item_z = [item.get_pose().p[2] for item in self.items]
            self.target_index = self.target_row * 4 + self.target_col

            # Randomize the described direction (near/far, left/right)
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
            col_text = ["1st", "2nd", "3rd", "4th"]
            col_dir_text = "from left to right"
        else:
            col_text = ["4th", "3rd", "2nd", "1st"]
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
