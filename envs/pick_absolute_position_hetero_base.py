import os
from copy import deepcopy

import numpy as np

from .pick_absolute_position_base import (
    ABSOLUTE_POSITION_RENDER_CAMERA_VIEW,
    ABSOLUTE_POSITION_VIEW_TEXT,
    ASSET_QPOS,
    HEAD_CAMERA_MARGIN_X_RATIO,
    HEAD_CAMERA_MARGIN_Y_RATIO,
    HEAD_CAMERA_MIN_CENTROID_DIST,
    HEAD_CAMERA_MAX_BBOX_OVERLAP_RATIO,
    HEAD_CAMERA_SETTLE_STEPS,
    TABLE_Z,
    PickAbsolutePositionMultiviewBase,
)
from .utils import ArmTag, create_actor, create_box, rand_pose, save_pkl


def _asset_spec(modelname, model_id, alias):
    # Build a concrete asset spec for a normal mesh-based object variant.
    return {
        "modelname": modelname,
        "model_id": model_id,
        "asset_alias": alias,
        "asset_key": f"{modelname}/base{model_id}",
    }


def _asset_family_spec(modelname, alias, variant_ids):
    # Build a family spec that can later be materialized into one concrete variant.
    return {
        "modelname": modelname,
        "asset_alias": alias,
        "variant_ids": list(variant_ids),
    }


def _primitive_block_spec(alias="block", variant_id=0):
    # Build a concrete spec for a primitive box object.
    return {
        "modelname": "primitive_block",
        "model_id": variant_id,
        "asset_alias": alias,
        "asset_key": f"primitive_block/base{variant_id}",
        "primitive_kind": "box",
        "half_size": (0.015, 0.015, 0.015),
        "color": (0.84, 0.22, 0.18),
    }


def _primitive_block_family_spec(alias="block"):
    # Build a primitive block family spec with multiple color variants.
    return {
        "modelname": "primitive_block",
        "asset_alias": alias,
        "primitive_kind": "box",
        "variant_ids": [0, 1, 2, 3, 4, 5],
    }


# Natural-language nouns used for each asset category.
ABSOLUTE_HETERO_CATEGORY_NOUNS = {
    "block": "block",
    "bread": "bread loaf",
    "seal": "seal",
    "toycar": "toy car",
    "phone": "phone",
    "rubikscube": "rubik's cube",
    "remotecontrol": "remote control",
    "stapler": "stapler",
    "can": "can",
    "soap": "soap bar",
    "tea_box": "tea box",
    "mouse": "computer mouse",
    "playingcards": "playing cards",
    "bell": "bell",
}

# Approximate object radii used for collision-free layout sampling.
ABSOLUTE_HETERO_RADIUS = {
    "block": 0.022,
    "bread": 0.048,
    "seal": 0.045,
    "toycar": 0.042,
    "phone": 0.052,
    "rubikscube": 0.044,
    "remotecontrol": 0.042,
    "stapler": 0.044,
    "can": 0.038,
    "soap": 0.036,
    "tea_box": 0.045,
    "mouse": 0.040,
    "playingcards": 0.032,
    "bell": 0.034,
}

# Per-category spacing requirements used to avoid visually or physically crowded layouts.
ABSOLUTE_HETERO_SPACING = {
    "block": {"x": 0.095, "y": 0.095, "pair": 0.095},
    "bread": {"x": 0.112, "y": 0.092, "pair": 0.100},
    "seal": {"x": 0.094, "y": 0.086, "pair": 0.088},
    "toycar": {"x": 0.098, "y": 0.086, "pair": 0.090},
    "phone": {"x": 0.126, "y": 0.094, "pair": 0.112},
    "rubikscube": {"x": 0.098, "y": 0.098, "pair": 0.094},
    "remotecontrol": {"x": 0.124, "y": 0.078, "pair": 0.104},
    "stapler": {"x": 0.118, "y": 0.092, "pair": 0.104},
    "can": {"x": 0.102, "y": 0.088, "pair": 0.095},
    "soap": {"x": 0.096, "y": 0.082, "pair": 0.088},
    "tea_box": {"x": 0.108, "y": 0.090, "pair": 0.098},
    "mouse": {"x": 0.100, "y": 0.082, "pair": 0.090},
    "playingcards": {"x": 0.090, "y": 0.078, "pair": 0.084},
    "bell": {"x": 0.090, "y": 0.090, "pair": 0.086},
}

# Extra clearance added between objects, depending on scene density.
ABSOLUTE_HETERO_CLEARANCE = {
    3: 0.014,
    4: 0.012,
    5: 0.010,
    6: 0.008,
    7: 0.006,
}

# Stable mesh variants for each object category.
ABSOLUTE_HETERO_STABLE_VARIANT_IDS = {
    "047_mouse": [0, 1, 2],
    "048_stapler": [0, 1, 2, 3, 4, 5, 6],
    "050_bell": [0, 1],
    "057_toycar": [0, 1, 2, 3, 4, 5],
    "071_can": [0, 1, 2, 3, 5, 6],
    "073_rubikscube": [0, 1, 2],
    "075_bread": [0, 1, 2, 3, 4, 5, 6],
    "077_phone": [0, 1, 2, 3],
    "079_remotecontrol": [0, 1, 2, 3, 4, 5, 6],
    "081_playingcards": [0, 1, 2],
    "100_seal": [1, 2, 3, 4, 6],
    "107_soap": [0, 1, 2, 3],
    "112_tea-box": [0, 3, 4, 5],
}

# Primitive block variants with different colors.
ABSOLUTE_HETERO_PRIMITIVE_BLOCK_VARIANTS = [
    {"variant_id": 0, "half_size": (0.015, 0.015, 0.015), "color": (0.84, 0.22, 0.18)},
    {"variant_id": 1, "half_size": (0.015, 0.015, 0.015), "color": (0.16, 0.58, 0.82)},
    {"variant_id": 2, "half_size": (0.015, 0.015, 0.015), "color": (0.22, 0.67, 0.28)},
    {"variant_id": 3, "half_size": (0.015, 0.015, 0.015), "color": (0.90, 0.74, 0.18)},
    {"variant_id": 4, "half_size": (0.015, 0.015, 0.015), "color": (0.58, 0.34, 0.80)},
    {"variant_id": 5, "half_size": (0.015, 0.015, 0.015), "color": (0.28, 0.28, 0.28)},
]

# Two object pools used by subclasses through asset_group_key.
ABSOLUTE_HETERO_GROUPS = {
    "a": [
        _primitive_block_family_spec("block"),
        _asset_family_spec("075_bread", "bread", ABSOLUTE_HETERO_STABLE_VARIANT_IDS["075_bread"]),
        _asset_family_spec("100_seal", "seal", ABSOLUTE_HETERO_STABLE_VARIANT_IDS["100_seal"]),
        _asset_family_spec("057_toycar", "toycar", ABSOLUTE_HETERO_STABLE_VARIANT_IDS["057_toycar"]),
        _asset_family_spec("077_phone", "phone", ABSOLUTE_HETERO_STABLE_VARIANT_IDS["077_phone"]),
        _asset_family_spec("073_rubikscube", "rubikscube", ABSOLUTE_HETERO_STABLE_VARIANT_IDS["073_rubikscube"]),
        _asset_family_spec("079_remotecontrol", "remotecontrol", ABSOLUTE_HETERO_STABLE_VARIANT_IDS["079_remotecontrol"]),
    ],
    "b": [
        _asset_family_spec("048_stapler", "stapler", ABSOLUTE_HETERO_STABLE_VARIANT_IDS["048_stapler"]),
        _asset_family_spec("071_can", "can", ABSOLUTE_HETERO_STABLE_VARIANT_IDS["071_can"]),
        _asset_family_spec("107_soap", "soap", ABSOLUTE_HETERO_STABLE_VARIANT_IDS["107_soap"]),
        _asset_family_spec("112_tea-box", "tea_box", ABSOLUTE_HETERO_STABLE_VARIANT_IDS["112_tea-box"]),
        _asset_family_spec("047_mouse", "mouse", ABSOLUTE_HETERO_STABLE_VARIANT_IDS["047_mouse"]),
        _asset_family_spec("081_playingcards", "playingcards", ABSOLUTE_HETERO_STABLE_VARIANT_IDS["081_playingcards"]),
        _asset_family_spec("050_bell", "bell", ABSOLUTE_HETERO_STABLE_VARIANT_IDS["050_bell"]),
    ],
}


class PickAbsolutePositionHeteroMultiviewBaseLyt0409V2(PickAbsolutePositionMultiviewBaseLyt331V2):
    # Subclasses should specify object_count and which heterogeneous asset group to use.
    object_count = None
    asset_group_key = None

    def _asset_group_specs(self):
        # Return the object family pool selected by asset_group_key.
        if self.asset_group_key not in ABSOLUTE_HETERO_GROUPS:
            raise RuntimeError(f"Unknown asset_group_key: {self.asset_group_key}")
        return [dict(spec) for spec in ABSOLUTE_HETERO_GROUPS[self.asset_group_key]]

    def _materialize_asset_spec(self, family_spec, rng):
        # Convert a family spec into one concrete asset variant.
        if family_spec.get("primitive_kind") == "box":
            # Primitive boxes use locally defined size/color variants.
            variant_lookup = {
                int(variant["variant_id"]): variant
                for variant in ABSOLUTE_HETERO_PRIMITIVE_BLOCK_VARIANTS
            }
            variant_id = int(rng.choice(family_spec["variant_ids"]))
            variant = variant_lookup[variant_id]
            return _primitive_block_spec(family_spec["asset_alias"], variant_id=variant_id) | {
                "half_size": variant["half_size"],
                "color": variant["color"],
            }

        # Mesh assets use the selected model_id variant.
        variant_id = int(rng.choice(family_spec["variant_ids"]))
        return _asset_spec(
            family_spec["modelname"],
            variant_id,
            family_spec["asset_alias"],
        )

    def _serialize_asset_spec(self, asset_spec):
        # Convert tuple values to lists so the asset spec can be safely pickled/serialized.
        serialized = {}
        for key, value in asset_spec.items():
            if isinstance(value, tuple):
                serialized[key] = list(value)
            else:
                serialized[key] = value
        return serialized

    def _deserialize_asset_spec(self, asset_spec):
        # Restore list values back to tuples for primitive object parameters.
        spec = dict(asset_spec)
        if "half_size" in spec and isinstance(spec["half_size"], list):
            spec["half_size"] = tuple(spec["half_size"])
        if "color" in spec and isinstance(spec["color"], list):
            spec["color"] = tuple(spec["color"])
        return spec

    def _serialize_scene_config(self):
        # Extend the parent scene config with the exact heterogeneous assets used.
        base = super()._serialize_scene_config() or {}
        if hasattr(self, "scene_assets"):
            base["scene_assets"] = [self._serialize_asset_spec(spec) for spec in self.scene_assets]
        return base

    def _asset_selection_seed(self):
        # Build a deterministic seed so asset selection is reproducible per task/episode.
        task_bias = sum((idx + 1) * ord(ch) for idx, ch in enumerate(str(self.task_name or "absolute_hetero")))
        return int(task_bias + 1009 * int(getattr(self, "seed_value", 0)) + 9173 * int(self.ep_num))

    def _sample_scene_assets(self):
        # Reuse saved scene assets when replaying a stored trajectory.
        if isinstance(self.scene_config, dict) and self.scene_config.get("scene_assets"):
            return [
                self._deserialize_asset_spec(spec)
                for spec in self.scene_config["scene_assets"]
            ]

        # Otherwise, sample distinct asset families from the selected group.
        pool = self._asset_group_specs()
        rng = np.random.RandomState(self._asset_selection_seed() % (2**31 - 1))
        chosen_idx = rng.choice(len(pool), size=self.object_count, replace=False)
        return [
            self._materialize_asset_spec(dict(pool[int(idx)]), rng)
            for idx in chosen_idx
        ]

    def save_traj_data(self, idx):
        # Save joint paths and serialized scene configuration for replay.
        file_path = os.path.join(self.save_dir, "_traj_data", f"episode{idx}.pkl")
        traj_data = {
            "left_joint_path": deepcopy(self.left_joint_path),
            "right_joint_path": deepcopy(self.right_joint_path),
            "scene_config": self._serialize_scene_config(),
        }
        save_pkl(file_path, traj_data)

    def _category_noun(self, asset_spec):
        # Return the human-readable category name for an asset.
        alias = asset_spec["asset_alias"]
        return ABSOLUTE_HETERO_CATEGORY_NOUNS.get(alias, alias.replace("_", " "))

    def _asset_radius(self, asset_spec):
        # Estimate the footprint radius used for spacing checks.
        if asset_spec.get("primitive_kind") == "box":
            return float(np.linalg.norm(np.asarray(asset_spec["half_size"][:2], dtype=float)))
        return ABSOLUTE_HETERO_RADIUS[asset_spec["asset_alias"]]

    def _pair_spacing(self, spec_a, spec_b):
        # Compute pairwise spacing thresholds by taking the stricter value of the two categories.
        spacing_a = ABSOLUTE_HETERO_SPACING[spec_a["asset_alias"]]
        spacing_b = ABSOLUTE_HETERO_SPACING[spec_b["asset_alias"]]
        return (
            max(float(spacing_a["x"]), float(spacing_b["x"])),
            max(float(spacing_a["y"]), float(spacing_b["y"])),
            max(float(spacing_a["pair"]), float(spacing_b["pair"])),
        )

    def _candidate_respects_safe_spacing(self, candidate, candidate_spec, positions, specs):
        # Check whether a newly sampled object position is safely separated from existing objects.
        for existing, existing_spec in zip(positions, specs):
            min_gap_x, min_gap_y, min_pair_gap = self._pair_spacing(candidate_spec, existing_spec)
            dx = abs(float(candidate[0] - existing[0]))
            dy = abs(float(candidate[1] - existing[1]))

            # Reject if the two objects are too close in both x and y.
            if dx < min_gap_x and dy < min_gap_y:
                return False

            # Also enforce a radial distance based on approximate object radii.
            radius_gap = (
                self._asset_radius(candidate_spec)
                + self._asset_radius(existing_spec)
                + ABSOLUTE_HETERO_CLEARANCE[self.object_count]
            )
            required_pair_gap = max(min_pair_gap, radius_gap)
            if float(np.linalg.norm(candidate[:2] - existing[:2])) < required_pair_gap:
                return False
        return True

    def _sample_positions(self, position_key):
        # Sample object positions for a heterogeneous absolute-position picking scene.
        if self.object_count not in {3, 4, 5, 6, 7}:
            raise ValueError("object_count must be one of {3, 4, 5, 6, 7}")

        # Workspace bounds for object placement.
        x_min_bound, x_max_bound = -0.240, 0.240
        y_min_bound, y_max_bound = -0.225, 0.095
        valid_candidates = []

        max_trials = 240
        for _ in range(max_trials):
            positions = []
            attempts = 0
            chosen_specs = list(self.scene_assets)

            # Sequentially sample positions, checking spacing against already placed objects.
            while len(positions) < self.object_count and attempts < 90:
                attempts += 1
                asset_spec = chosen_specs[len(positions)]
                radius = self._asset_radius(asset_spec)

                # Keep the object footprint inside the workspace bounds.
                pos_x = float(np.random.uniform(x_min_bound + radius, x_max_bound - radius))
                pos_y = float(np.random.uniform(y_min_bound + radius, y_max_bound - radius))

                # Primitive boxes sit at TABLE_Z plus their half-height; mesh assets use TABLE_Z directly.
                pos_z = TABLE_Z + float(asset_spec["half_size"][2]) if asset_spec.get("primitive_kind") == "box" else TABLE_Z

                candidate = np.array([pos_x, pos_y, pos_z], dtype=float)

                # Reject candidates that are too close to existing objects.
                if not self._candidate_respects_safe_spacing(candidate, asset_spec, positions, chosen_specs[: len(positions)]):
                    continue

                positions.append(candidate)

            # Reject incomplete samples.
            if len(positions) != self.object_count:
                continue

            # Require the requested extreme object to be uniquely identifiable.
            if not self._has_unique_extreme(positions, position_key):
                continue

            # Require all object centers to be visible in the head camera.
            projected = self._project_positions_to_head_camera(positions)
            if not self._positions_visible_in_head_camera(projected):
                continue

            poses = []
            for pos, asset_spec in zip(positions, chosen_specs):
                if asset_spec.get("primitive_kind") == "box":
                    # Primitive boxes use upright orientation with a small random yaw.
                    poses.append(
                        rand_pose(
                            xlim=[float(pos[0]), float(pos[0])],
                            ylim=[float(pos[1]), float(pos[1])],
                            zlim=[float(pos[2]), float(pos[2])],
                            qpos=[1, 0, 0, 0],
                            rotate_rand=True,
                            rotate_lim=[0, 0, 0.75],
                        )
                    )
                else:
                    # Mesh assets use the shared object quaternion and randomized yaw/pitch range.
                    poses.append(
                        rand_pose(
                            xlim=[float(pos[0]), float(pos[0])],
                            ylim=[float(pos[1]), float(pos[1])],
                            zlim=[float(pos[2]), float(pos[2])],
                            qpos=ASSET_QPOS,
                            rotate_rand=True,
                            rotate_lim=[0, float(np.pi), 0],
                        )
                    )

            valid_candidates.append((positions, poses))
            if len(valid_candidates) >= 8:
                break

        # Fail explicitly if no valid candidate layout is found.
        if not valid_candidates:
            raise RuntimeError(
                f"Failed to sample a clear {self.object_count}-object hetero absolute layout"
            )

        # Randomly pick one of the valid layouts for diversity.
        chosen_positions, chosen_poses = valid_candidates[int(np.random.randint(len(valid_candidates)))]
        return {"positions": chosen_positions, "poses": chosen_poses, "score": None}

    def _create_asset_object(self, pose, asset_spec):
        # Create either a primitive block or a normal mesh actor from the asset spec.
        if asset_spec.get("primitive_kind") == "box":
            obj = create_box(
                scene=self.scene,
                pose=pose,
                half_size=asset_spec["half_size"],
                color=asset_spec["color"],
                name=asset_spec["asset_alias"],
            )
            obj.set_mass(0.05)
            self.add_prohibit_area(obj, padding=0.05)
            return obj

        obj = create_actor(
            scene=self,
            pose=pose,
            modelname=asset_spec["modelname"],
            convex=True,
            model_id=asset_spec["model_id"],
        )
        if obj is None:
            raise RuntimeError(f"Failed to create asset: {asset_spec['asset_key']}")

        obj.set_mass(0.05)
        self.add_prohibit_area(obj, padding=0.05)
        return obj

    def _target_phrase(self, position_key, asset_spec):
        # Currently only describe the target by its relative position.
        # asset_spec is kept here so subclasses can include object-category wording later.
        _ = asset_spec
        return self._relation_phrase(position_key)

    def load_actors(self):
        # Select the instruction viewpoint and apply the fixed render camera.
        self._apply_observation_view(self._select_observation_view())

        # Select the target relation and sample heterogeneous scene assets.
        self.position_key = self._desired_position_key()
        self.scene_assets = self._sample_scene_assets()

        # Try multiple layouts until the rendered scene is visually clear.
        for _ in range(40):
            sampled = self._sample_positions(self.position_key)
            prohibit_area_length = len(self.prohibited_area)
            candidate_objects = []

            # Create candidate objects for this sampled layout.
            for pose, asset_spec in zip(sampled["poses"], self.scene_assets):
                candidate_objects.append(self._create_asset_object(pose, asset_spec))

            # Validate rendered segmentation clarity.
            rendered = self._rendered_object_layout(candidate_objects)
            if not self._is_visually_clear_rendered_layout(rendered):
                self._remove_candidate_objects(candidate_objects, prohibit_area_length)
                continue

            # Accept this scene layout.
            self.objects = candidate_objects
            self.object_positions = [obj.get_pose().p.copy() for obj in self.objects]

            # Select the target object by absolute position.
            self.target_index = self._target_index(self.object_positions, self.position_key)
            self.target_object = self.objects[self.target_index]
            self.target_asset = self.scene_assets[self.target_index]

            # Choose the robot arm based on the target object's x coordinate.
            target_x = float(self.target_object.get_pose().p[0])
            self.arm_tag = ArmTag("right" if target_x > 0 else "left")

            # Store the initial z position for later lift-success checking.
            self.start_z = float(self.target_object.get_pose().p[2])

            # Convert the absolute target relation into the selected viewpoint's wording.
            self.observed_position_key = self._view_relative_position_key(self.position_key)
            self.position_phrase = self._target_phrase(self.observed_position_key, self.target_asset)
            return

        raise RuntimeError(
            f"Failed to sample a visually clear {self.object_count}-object hetero absolute layout"
        )

    def play_once(self):
        # Grasp the target object with the selected arm.
        arm_tag = self.arm_tag
        self.move(
            self.grasp_actor(
                self.target_object,
                arm_tag=arm_tag,
                pre_grasp_dis=self._play_once_pre_grasp_dis(),
                grasp_dis=self._play_once_grasp_dis(),
            )
        )

        # Lift the grasped target object upward.
        self.move(
            self.move_by_displacement(
                arm_tag=arm_tag,
                z=self._play_once_lift_z(),
                move_axis="arm",
            )
        )

        # Store layout metadata for evaluation, replay, and instruction generation.
        self.info["layout"] = {
            "object_count": self.object_count,
            "target_index": self.target_index,
            "position_key": self.position_key,
            "observed_position_key": self.observed_position_key,
            "scene_assets": [spec["asset_key"] for spec in self.scene_assets],
            "scene_asset_aliases": [spec["asset_alias"] for spec in self.scene_assets],
            "target_asset_alias": self.target_asset["asset_alias"],
            "view_name": self.view_name,
            "camera_view_name": getattr(self, "camera_view_name", ABSOLUTE_POSITION_RENDER_CAMERA_VIEW),
            "object_positions": [
                [float(pos[0]), float(pos[1]), float(pos[2])]
                for pos in self.object_positions
            ],
        }

        # Store natural-language placeholders used by the task instruction.
        self.info["info"] = {
            "{P}": self.position_phrase,
            "{V}": ABSOLUTE_POSITION_VIEW_TEXT[self.view_name],
            "{a}": str(arm_tag),
        }
        return self.info