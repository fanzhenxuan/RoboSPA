from ._base_task import Base_Task
from .utils import *

import os
import glob
import json
import numpy as np


class Pick_Mixed_Objects_Canonical_B_5(Base_Task):

    # Candidate object categories used to build the 3x4 mixed-object grid.
    # skip_model_ids excludes unstable or unsuitable variants for this task.
    OBJECT_SPECS = [
        {'modelname': '021_cup', 'skip_model_ids': [10, 11, 12]},
        {'modelname': '078_phonestand', 'skip_model_ids': [1, 5]},
        {'modelname': '071_can', 'skip_model_ids': []},
        {'modelname': '080_pillbottle', 'skip_model_ids': []},
        {'modelname': '005_french-fries', 'skip_model_ids': []},
        {'modelname': '035_apple', 'skip_model_ids': []},
        {'modelname': '107_soap', 'skip_model_ids': []},
        {'modelname': '115_perfume', 'skip_model_ids': [2]},
        {'modelname': '112_tea-box', 'skip_model_ids': []},
        {'modelname': '113_coffee-box', 'skip_model_ids': []},
        {'modelname': '105_sauce-can', 'skip_model_ids': []},
        {'modelname': '075_bread', 'skip_model_ids': [1, 2]},
    ]

    def setup_demo(self, **kwargs):
        # Initialize the task environment with the provided keyword arguments.
        super()._init_task_env_(**kwargs)

    def _sample_model_id(self, spec):
        # Find all model_data*.json files for this asset category.
        asset_path = os.path.join("assets/objects", spec["modelname"])
        json_files = glob.glob(os.path.join(asset_path, "model_data*.json"))

        model_ids = []
        for file in json_files:
            base = os.path.basename(file)

            # Extract the numeric model id from file names like model_data3.json.
            try:
                idx = int(base.replace("model_data", "").replace(".json", ""))
            except ValueError:
                continue

            # Skip variants that are known to be unsuitable for this task.
            if idx in spec.get("skip_model_ids", []):
                continue
            model_ids.append(idx)

        # Fail clearly if no usable variant exists for this category.
        if len(model_ids) == 0:
            raise ValueError(f"No available model_data*.json found for {spec['modelname']} after skipping ids")

        # Randomly select one available model variant.
        model_ids = sorted(model_ids)
        return int(np.random.choice(model_ids))

    def _sampled_specs(self):
        # Materialize the object specs by assigning one sampled model_id to each category.
        specs = []
        for spec in self.OBJECT_SPECS[:12]:
            spec = dict(spec)
            spec["model_id"] = self._sample_model_id(spec)
            specs.append(spec)
        return specs

    def load_actors(self):
        # Sample one concrete model variant for each object category, then shuffle
        # the categories so the grid arrangement changes across episodes.
        self.object_specs = self._sampled_specs()
        np.random.shuffle(self.object_specs)

        # Basic placement and task parameters.
        self.padding = 0.02
        self.prohibit_area_range = [-0.26, -0.24, 0.26, -0.12]

        self.pre_grasp_dis = 0.10
        self.lift_z = 0.10
        self.target_lift_thresh = 0.04
        self.other_stable_thresh = 0.03
        self.use_contact_point = False

        # Define the center and spacing for a 3-row by 4-column object grid.
        center_x = np.random.uniform(-0.04, -0.03)
        center_y = np.random.uniform(-0.105, -0.10)
        col_gap = 0.156
        row_gap = 0.122

        x_list = [
            center_x - 1.5 * col_gap,
            center_x - 0.5 * col_gap,
            center_x + 0.5 * col_gap,
            center_x + 1.5 * col_gap,
        ]
        y_list = [
            center_y - row_gap,
            center_y,
            center_y + row_gap,
        ]

        # Small jitter keeps the grid mostly canonical while avoiding identical placements.
        x_jitter = 0.003
        y_jitter = 0.003

        def fixed_item_pose(x, y, qpos):
            # Sample a nearly fixed pose around the given grid coordinate.
            return rand_pose(
                xlim=[x - x_jitter, x + x_jitter],
                ylim=[y - y_jitter, y + y_jitter],
                rotate_rand=False,
                qpos=qpos,
            )

        # Generate grid coordinates in row-major order.
        poses_coords = []
        for y in y_list:
            for x in x_list:
                poses_coords.append((x, y))

        # Create all 12 object actors.
        self.items = []
        for i in range(12):
            spec = self.object_specs[i]
            x, y = poses_coords[i]

            # French fries use an upright quaternion; other assets use the shared canonical pose.
            if spec['modelname'] == '005_french-fries':
                target_qpos = [1, 0, 0, 0]
            else:
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

        # Store the flat item list as a 3x4 grid for row/column target selection.
        self.item_grid = [
            [self.items[0], self.items[1], self.items[2], self.items[3]],
            [self.items[4], self.items[5], self.items[6], self.items[7]],
            [self.items[8], self.items[9], self.items[10], self.items[11]],
        ]

        # Register prohibited areas around each object and an extra region near the robot/base.
        for item in self.items:
            self.add_prohibit_area(item, padding=self.padding)
        self.prohibited_area.append(self.prohibit_area_range)

        # Let objects settle before recording their initial heights.
        self.delay(5)

        # Randomly choose a target grid cell.
        self.target_row = np.random.randint(3)
        self.target_col = np.random.randint(4)
        self.target_item = self.item_grid[self.target_row][self.target_col]

        # Record initial heights for later success and disturbance checks.
        self.init_item_z = [item.get_pose().p[2] for item in self.items]
        self.target_index = self.target_row * 4 + self.target_col

        # Randomize how row and column indices are described in the instruction.
        self.row_from_far_to_near = bool(np.random.randint(2))
        self.col_from_left_to_right = bool(np.random.randint(2))

    def play_once(self):
        # Reset the last used gripper before executing the demonstration.
        self.last_gripper = None

        # Pick the target item and record which arm was used.
        arm_tag = self.pick_target_item(self.target_item)

        # Build row text and row-direction text according to the randomized convention.
        if self.row_from_far_to_near:
            row_text = ["1st row", "2nd row", "3rd row"]
            row_dir_text = "from near to far"
        else:
            row_text = ["3rd row", "2nd row", "1st row"]
            row_dir_text = "from far to near"

        # Build column text and column-direction text according to the randomized convention.
        if self.col_from_left_to_right:
            col_text = ["1st", "2nd", "3rd", "4th"]
            col_dir_text = "from left to right"
        else:
            col_text = ["4th", "3rd", "2nd", "1st"]
            col_dir_text = "from right to left"

        # Store placeholders for natural-language instruction generation.
        self.info["info"] = {
            "{R}": row_text[self.target_row],
            "{C}": col_text[self.target_col],
            "{row_dir}": row_dir_text,
            "{col_dir}": col_dir_text,
            "{a}": str(arm_tag),
        }

        return self.info

    def pick_target_item(self, item):
        # Choose the arm based on the target item's x position.
        item_pose = item.get_pose().p
        arm_tag = ArmTag("left" if item_pose[0] < 0 else "right")

        # Basic grasp parameters.
        grasp_kwargs = {
            "arm_tag": arm_tag,
            "pre_grasp_dis": self.pre_grasp_dis,
        }

        # Optionally use category-specific contact points if enabled.
        if self.use_contact_point:
            grasp_kwargs["contact_point_id"] = 0 if str(arm_tag) == "right" else 2

        # If switching arms, return the previously used arm to origin while grasping.
        if self.last_gripper is not None and self.last_gripper != arm_tag:
            self.move(
                self.grasp_actor(item, **grasp_kwargs),
                self.back_to_origin(arm_tag=self.last_gripper),
            )
        else:
            # Otherwise, directly grasp the target item.
            self.move(
                self.grasp_actor(item, **grasp_kwargs)
            )

        # Lift the grasped item upward.
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=self.lift_z))

        # Remember the gripper used for this pick.
        self.last_gripper = arm_tag
        return str(arm_tag)

    def check_success(self):
        # Read current heights of all objects.
        cur_z = [item.get_pose().p[2] for item in self.items]

        # The target must be lifted above its initial height by the target threshold.
        target_lifted = cur_z[self.target_index] > self.init_item_z[self.target_index] + self.target_lift_thresh

        # All non-target objects should remain approximately stable.
        other_stable = True
        for i in range(len(self.items)):
            if i == self.target_index:
                continue
            if abs(cur_z[i] - self.init_item_z[i]) > self.other_stable_thresh:
                other_stable = False
                break

        # Success requires lifting the target without significantly disturbing other objects.
        return target_lifted and other_stable