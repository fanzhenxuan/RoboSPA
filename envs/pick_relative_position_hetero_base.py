import os
from copy import deepcopy

import numpy as np

from .pick_relative_position_base import (
    ASSET_QPOS,
    DIRECTION_TO_DELTA,
    HEAD_CAMERA_MARGIN_X_RATIO,
    HEAD_CAMERA_MARGIN_Y_RATIO,
    TABLE_Z,
    PickRelativePositionBase,
)
from .utils import ArmTag, create_actor, create_box, rand_pose, save_pkl


def _asset_spec(modelname, model_id, alias):
    return {
        "modelname": modelname,
        "model_id": model_id,
        "asset_alias": alias,
        "asset_key": f"{modelname}/base{model_id}",
    }


def _asset_family_spec(modelname, alias, variant_ids):
    return {
        "modelname": modelname,
        "asset_alias": alias,
        "variant_ids": list(variant_ids),
    }


def _primitive_block_spec(alias="block", variant_id=0):
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
    return {
        "modelname": "primitive_block",
        "asset_alias": alias,
        "primitive_kind": "box",
        "variant_ids": [0, 1, 2, 3, 4, 5],
    }


RELATIVE_HETERO_CATEGORY_NOUNS = {
    "block": "block",
    "can": "can",
    "soap": "soap bar",
    "stapler": "stapler",
    "tea_box": "tea box",
    "toycar": "toy car",
    "seal": "seal",
    "rubikscube": "rubik's cube",
    "bell": "bell",
    "mouse": "computer mouse",
    "playingcards": "playing cards",
    "remotecontrol": "remote control",
}

RELATIVE_HETERO_SPACING = {
    "block": {"x": 0.095, "y": 0.095, "pair": 0.095},
    "can": {"x": 0.102, "y": 0.088, "pair": 0.095},
    "soap": {"x": 0.096, "y": 0.082, "pair": 0.088},
    "stapler": {"x": 0.118, "y": 0.092, "pair": 0.104},
    "tea_box": {"x": 0.108, "y": 0.090, "pair": 0.098},
    "toycar": {"x": 0.098, "y": 0.086, "pair": 0.090},
    "seal": {"x": 0.094, "y": 0.086, "pair": 0.088},
    "rubikscube": {"x": 0.094, "y": 0.094, "pair": 0.092},
    "bell": {"x": 0.090, "y": 0.090, "pair": 0.086},
    "mouse": {"x": 0.100, "y": 0.082, "pair": 0.090},
    "playingcards": {"x": 0.090, "y": 0.078, "pair": 0.084},
    "remotecontrol": {"x": 0.124, "y": 0.078, "pair": 0.104},
}

RELATIVE_HETERO_STABLE_VARIANT_IDS = {
    "047_mouse": [0, 1, 2],
    "048_stapler": [0, 1, 2, 3, 4, 5, 6],
    "050_bell": [0, 1],
    "057_toycar": [0, 1, 2, 3, 4, 5],
    "071_can": [0, 1, 2, 3, 5, 6],
    "073_rubikscube": [0, 1, 2],
    "079_remotecontrol": [0, 1, 2, 3, 4, 5, 6],
    "081_playingcards": [0, 1, 2],
    "100_seal": [1, 2, 3, 4, 6],
    "107_soap": [0, 1, 2, 3],
    "112_tea-box": [0, 3, 4, 5],
}

RELATIVE_HETERO_PRIMITIVE_BLOCK_VARIANTS = [
    {"variant_id": 0, "half_size": (0.015, 0.015, 0.015), "color": (0.84, 0.22, 0.18)},
    {"variant_id": 1, "half_size": (0.015, 0.015, 0.015), "color": (0.16, 0.58, 0.82)},
    {"variant_id": 2, "half_size": (0.015, 0.015, 0.015), "color": (0.22, 0.67, 0.28)},
    {"variant_id": 3, "half_size": (0.015, 0.015, 0.015), "color": (0.90, 0.74, 0.18)},
    {"variant_id": 4, "half_size": (0.015, 0.015, 0.015), "color": (0.58, 0.34, 0.80)},
    {"variant_id": 5, "half_size": (0.015, 0.015, 0.015), "color": (0.28, 0.28, 0.28)},
]

RELATIVE_HETERO_GROUPS = {
    "a": [
        _primitive_block_family_spec("block"),
        _asset_family_spec("071_can", "can", RELATIVE_HETERO_STABLE_VARIANT_IDS["071_can"]),
        _asset_family_spec("107_soap", "soap", RELATIVE_HETERO_STABLE_VARIANT_IDS["107_soap"]),
        _asset_family_spec("048_stapler", "stapler", RELATIVE_HETERO_STABLE_VARIANT_IDS["048_stapler"]),
        _asset_family_spec("112_tea-box", "tea_box", RELATIVE_HETERO_STABLE_VARIANT_IDS["112_tea-box"]),
        _asset_family_spec("057_toycar", "toycar", RELATIVE_HETERO_STABLE_VARIANT_IDS["057_toycar"]),
        _asset_family_spec("100_seal", "seal", RELATIVE_HETERO_STABLE_VARIANT_IDS["100_seal"]),
        _asset_family_spec("073_rubikscube", "rubikscube", RELATIVE_HETERO_STABLE_VARIANT_IDS["073_rubikscube"]),
        _asset_family_spec("050_bell", "bell", RELATIVE_HETERO_STABLE_VARIANT_IDS["050_bell"]),
    ],
    "b": [
        _asset_family_spec("047_mouse", "mouse", RELATIVE_HETERO_STABLE_VARIANT_IDS["047_mouse"]),
        _asset_family_spec("081_playingcards", "playingcards", RELATIVE_HETERO_STABLE_VARIANT_IDS["081_playingcards"]),
        _asset_family_spec("079_remotecontrol", "remotecontrol", RELATIVE_HETERO_STABLE_VARIANT_IDS["079_remotecontrol"]),
        _asset_family_spec("071_can", "can", RELATIVE_HETERO_STABLE_VARIANT_IDS["071_can"]),
        _asset_family_spec("107_soap", "soap", RELATIVE_HETERO_STABLE_VARIANT_IDS["107_soap"]),
        _asset_family_spec("048_stapler", "stapler", RELATIVE_HETERO_STABLE_VARIANT_IDS["048_stapler"]),
        _asset_family_spec("057_toycar", "toycar", RELATIVE_HETERO_STABLE_VARIANT_IDS["057_toycar"]),
        _asset_family_spec("100_seal", "seal", RELATIVE_HETERO_STABLE_VARIANT_IDS["100_seal"]),
        _asset_family_spec("050_bell", "bell", RELATIVE_HETERO_STABLE_VARIANT_IDS["050_bell"]),
    ],
}


class PickRelativePositionHeteroBase(PickRelativePositionBase):
    object_count = None
    asset_group_key = None

    def _asset_group_specs(self):
        if self.asset_group_key not in RELATIVE_HETERO_GROUPS:
            raise RuntimeError(f"Unknown asset_group_key: {self.asset_group_key}")
        return [dict(spec) for spec in RELATIVE_HETERO_GROUPS[self.asset_group_key]]

    def _materialize_asset_spec(self, family_spec, rng):
        if family_spec.get("primitive_kind") == "box":
            variant_lookup = {
                int(variant["variant_id"]): variant
                for variant in RELATIVE_HETERO_PRIMITIVE_BLOCK_VARIANTS
            }
            variant_id = int(rng.choice(family_spec["variant_ids"]))
            variant = variant_lookup[variant_id]
            return _primitive_block_spec(family_spec["asset_alias"], variant_id=variant_id) | {
                "half_size": variant["half_size"],
                "color": variant["color"],
            }

        variant_id = int(rng.choice(family_spec["variant_ids"]))
        return _asset_spec(
            family_spec["modelname"],
            variant_id,
            family_spec["asset_alias"],
        )

    def _serialize_asset_spec(self, asset_spec):
        serialized = {}
        for key, value in asset_spec.items():
            if isinstance(value, tuple):
                serialized[key] = list(value)
            else:
                serialized[key] = value
        return serialized

    def _deserialize_asset_spec(self, asset_spec):
        spec = dict(asset_spec)
        if "half_size" in spec and isinstance(spec["half_size"], list):
            spec["half_size"] = tuple(spec["half_size"])
        if "color" in spec and isinstance(spec["color"], list):
            spec["color"] = tuple(spec["color"])
        return spec

    def _asset_selection_seed(self, occupied_slots):
        task_bias = sum((idx + 1) * ord(ch) for idx, ch in enumerate(str(self.task_name or "relative_hetero")))
        slot_bias = sum((idx + 1) * sum(ord(ch) for ch in slot_name) for idx, slot_name in enumerate(occupied_slots))
        return int(task_bias + 1009 * int(getattr(self, "seed_value", 0)) + 9173 * int(self.ep_num) + slot_bias)

    def _slot_asset_specs_from_scene_config(self, occupied_slots):
        if not isinstance(self.scene_config, dict):
            return None
        slot_specs = self.scene_config.get("slot_asset_specs")
        if not isinstance(slot_specs, dict):
            return None
        if set(slot_specs.keys()) != set(occupied_slots):
            return None
        return {
            slot_name: self._deserialize_asset_spec(spec)
            for slot_name, spec in slot_specs.items()
        }

    def _select_slot_asset_specs(self, occupied_slots):
        slot_specs = self._slot_asset_specs_from_scene_config(occupied_slots)
        if slot_specs is not None:
            return slot_specs

        pool = self._asset_group_specs()
        rng = np.random.RandomState(self._asset_selection_seed(occupied_slots) % (2**31 - 1))
        chosen_idx = rng.choice(len(pool), size=len(occupied_slots), replace=False)
        chosen_specs = [self._materialize_asset_spec(dict(pool[int(idx)]), rng) for idx in chosen_idx]
        return {
            slot_name: asset_spec
            for slot_name, asset_spec in zip(sorted(occupied_slots, key=self._slot_sort_key), chosen_specs)
        }

    def _serialize_scene_config(self):
        if not hasattr(self, "slot_asset_specs"):
            return None
        return {
            "slot_asset_specs": {
                slot_name: self._serialize_asset_spec(spec)
                for slot_name, spec in self.slot_asset_specs.items()
            }
        }

    def save_traj_data(self, idx):
        file_path = os.path.join(self.save_dir, "_traj_data", f"episode{idx}.pkl")
        traj_data = {
            "left_joint_path": deepcopy(self.left_joint_path),
            "right_joint_path": deepcopy(self.right_joint_path),
            "scene_config": self._serialize_scene_config(),
        }
        save_pkl(file_path, traj_data)

    def _category_noun(self, asset_spec):
        alias = asset_spec["asset_alias"]
        return RELATIVE_HETERO_CATEGORY_NOUNS.get(alias, alias.replace("_", " "))

    def _slot_phrase_with_asset(self, slot_name, occupied_slots, slot_asset_specs):
        _ = slot_asset_specs
        return super()._describe_slot(slot_name, occupied_slots)

    def _relative_min_safe_gaps(self, slot_asset_specs):
        max_x = max(float(RELATIVE_HETERO_SPACING[spec["asset_alias"]]["x"]) for spec in slot_asset_specs.values())
        max_y = max(float(RELATIVE_HETERO_SPACING[spec["asset_alias"]]["y"]) for spec in slot_asset_specs.values())
        max_pair = max(float(RELATIVE_HETERO_SPACING[spec["asset_alias"]]["pair"]) for spec in slot_asset_specs.values())
        mode = self._grid_mode()
        if mode == "2x2":
            scale = 0.78
        elif mode == "2x3":
            scale = 0.92
        else:
            scale = 0.86
        return max_x * scale, max_y * scale, max_pair * scale

    def _layout_respects_safe_spacing(self, occupied_slots, positions, slot_asset_specs):
        slot_to_coord = self._slot_to_grid_map()
        min_gap_x, min_gap_y, _ = self._relative_min_safe_gaps(slot_asset_specs)

        used_columns = sorted({slot_to_coord[slot_name][0] for slot_name in occupied_slots})
        used_rows = sorted({slot_to_coord[slot_name][1] for slot_name in occupied_slots})

        column_centers = []
        for grid_x in used_columns:
            values = [
                float(positions[slot_name][0])
                for slot_name in occupied_slots
                if slot_to_coord[slot_name][0] == grid_x
            ]
            column_centers.append(float(np.mean(values)))
        row_centers = []
        for grid_y in used_rows:
            values = [
                float(positions[slot_name][1])
                for slot_name in occupied_slots
                if slot_to_coord[slot_name][1] == grid_y
            ]
            row_centers.append(float(np.mean(values)))

        if any((right - left) < min_gap_x for left, right in zip(column_centers, column_centers[1:])):
            return False
        if any((upper - lower) < min_gap_y for lower, upper in zip(row_centers, row_centers[1:])):
            return False

        for idx, slot_i in enumerate(occupied_slots):
            point_i = np.asarray(positions[slot_i][:2], dtype=float)
            spec_i = slot_asset_specs[slot_i]
            for slot_j in occupied_slots[idx + 1:]:
                point_j = np.asarray(positions[slot_j][:2], dtype=float)
                spec_j = slot_asset_specs[slot_j]
                pair_gap = max(
                    float(RELATIVE_HETERO_SPACING[spec_i["asset_alias"]]["pair"]),
                    float(RELATIVE_HETERO_SPACING[spec_j["asset_alias"]]["pair"]),
                )
                if float(np.linalg.norm(point_i - point_j)) < pair_gap:
                    return False
        return True

    def _fixed_layout_bounds(self):
        mode = self._grid_mode()
        if mode == "2x2":
            return (-0.205, 0.205), (-0.195, 0.035)
        if mode == "2x3":
            return (-0.225, 0.225), (-0.205, 0.045)
        return (-0.228, 0.228), (-0.215, 0.075)

    def _fixed_gap_ranges(self):
        min_gap_x, min_gap_y, _ = self._relative_min_safe_gaps(self._default_slot_asset_specs())
        mode = self._grid_mode()
        if mode == "2x2":
            return (max(0.090, min_gap_x + 0.006), 0.185), (max(0.085, min_gap_y + 0.006), 0.170)
        if mode == "2x3":
            return (max(0.088, min_gap_x + 0.006), 0.160), (max(0.082, min_gap_y + 0.006), 0.160)
        return (max(0.080, min_gap_x + 0.004), 0.132), (max(0.076, min_gap_y + 0.004), 0.122)

    def _fixed_shift_ranges(self):
        mode = self._grid_mode()
        if mode == "2x2":
            return 0.014, 0.012
        if mode == "2x3":
            return 0.018, 0.012
        return 0.016, 0.014

    def _default_slot_asset_specs(self):
        pool = self._asset_group_specs()
        slots = self._slot_names()[: min(len(pool), len(self._slot_names()))]
        return {slot: pool[idx] for idx, slot in enumerate(slots)}

    def _build_layout_from_xy(self, direction_key, reference_slot, target_slot, occupied_slots, xy_positions, slot_asset_specs):
        poses = {}
        positions = {}
        for slot_name in occupied_slots:
            asset_spec = slot_asset_specs[slot_name]
            pos_x, pos_y = xy_positions[slot_name]
            if asset_spec.get("primitive_kind") == "box":
                pos_z = TABLE_Z + float(asset_spec["half_size"][2])
                poses[slot_name] = rand_pose(
                    xlim=[pos_x, pos_x],
                    ylim=[pos_y, pos_y],
                    zlim=[pos_z, pos_z],
                    qpos=[1, 0, 0, 0],
                    rotate_rand=True,
                    rotate_lim=[0, 0, 0.75],
                )
            else:
                pos_z = TABLE_Z
                poses[slot_name] = rand_pose(
                    xlim=[pos_x, pos_x],
                    ylim=[pos_y, pos_y],
                    zlim=[pos_z, pos_z],
                    qpos=ASSET_QPOS,
                    rotate_rand=True,
                    rotate_lim=[0, float(np.pi), 0],
                )
            positions[slot_name] = np.array([pos_x, pos_y, pos_z], dtype=float)

        return {
            "direction_key": direction_key,
            "seed_reference_slot": reference_slot,
            "seed_target_slot": target_slot,
            "occupied_slots": occupied_slots,
            "poses": poses,
            "positions": positions,
            "slot_asset_specs": slot_asset_specs,
        }

    def _sample_layout(self, direction_key):
        all_slots = self._slot_names()
        valid_reference_slots = self._valid_reference_slots(direction_key)
        if not valid_reference_slots:
            raise RuntimeError(f"No valid reference slots for direction {direction_key}")

        slot_to_coord = self._slot_to_grid_map()
        grid_to_slot = self._grid_to_slot_map()
        (x_min_bound, x_max_bound), (y_min_bound, y_max_bound) = self._fixed_layout_bounds()
        gap_x_range, gap_y_range = self._fixed_gap_ranges()
        column_shift_mag, row_shift_mag = self._fixed_shift_ranges()

        if self.object_count == 2:
            max_tries = 40
        elif self.object_count in {3, 4}:
            max_tries = 70
        elif self.object_count == 6:
            max_tries = 100
        else:
            max_tries = 140

        for _ in range(max_tries):
            reference_slot = str(np.random.choice(valid_reference_slots))
            ref_x, ref_y = slot_to_coord[reference_slot]
            delta_x, delta_y = DIRECTION_TO_DELTA[direction_key]
            target_slot = grid_to_slot[(ref_x + delta_x, ref_y + delta_y)]

            remaining_slots = [slot_name for slot_name in all_slots if slot_name not in {reference_slot, target_slot}]
            distractor_slots = []
            if self.object_count > 2:
                distractor_slots = [
                    str(slot_name)
                    for slot_name in np.random.choice(remaining_slots, size=self.object_count - 2, replace=False)
                ]

            occupied_slots = [reference_slot, target_slot] + distractor_slots
            occupied_slots.sort(key=self._slot_sort_key)
            slot_asset_specs = self._select_slot_asset_specs(occupied_slots)

            grid_gap_x = float(np.random.uniform(*gap_x_range))
            grid_gap_y = float(np.random.uniform(*gap_y_range))

            used_columns = sorted({slot_to_coord[slot_name][0] for slot_name in occupied_slots})
            used_rows = sorted({slot_to_coord[slot_name][1] for slot_name in occupied_slots})
            column_shift = {
                grid_x: float(np.random.uniform(-column_shift_mag, column_shift_mag))
                for grid_x in used_columns
            }
            row_shift = {
                grid_y: float(np.random.uniform(-row_shift_mag, row_shift_mag))
                for grid_y in used_rows
            }

            xy_positions = {}
            for slot_name in occupied_slots:
                offset_x, offset_y = self._slot_center_offset(slot_name)
                grid_x, grid_y = slot_to_coord[slot_name]
                xy_positions[slot_name] = np.array(
                    [
                        offset_x * grid_gap_x + column_shift.get(grid_x, 0.0),
                        offset_y * grid_gap_y + row_shift.get(grid_y, 0.0),
                    ],
                    dtype=float,
                )

            offset_xs = [position[0] for position in xy_positions.values()]
            offset_ys = [position[1] for position in xy_positions.values()]
            anchor_x_min = x_min_bound - max(offset_xs)
            anchor_x_max = x_max_bound - min(offset_xs)
            anchor_y_min = y_min_bound - max(offset_ys)
            anchor_y_max = y_max_bound - min(offset_ys)
            if anchor_x_min >= anchor_x_max or anchor_y_min >= anchor_y_max:
                continue

            anchor_x = float(np.random.uniform(anchor_x_min, anchor_x_max))
            anchor_y = float(np.random.uniform(anchor_y_min, anchor_y_max))
            shifted_positions = {
                slot_name: np.array([position[0] + anchor_x, position[1] + anchor_y], dtype=float)
                for slot_name, position in xy_positions.items()
            }

            layout = self._build_layout_from_xy(
                direction_key,
                reference_slot,
                target_slot,
                occupied_slots,
                shifted_positions,
                slot_asset_specs,
            )
            if not self._layout_respects_safe_spacing(occupied_slots, layout["positions"], slot_asset_specs):
                continue
            if not self._is_visually_consistent_layout(layout):
                continue
            return layout

        raise RuntimeError(f"Failed to sample a valid {self.object_count}-object hetero relative layout")

    def _build_layout_metadata(self, layout):
        occupied_slots = list(layout["occupied_slots"])
        slot_asset_specs = layout["slot_asset_specs"]
        slot_descriptions = {
            slot_name: self._slot_phrase_with_asset(slot_name, occupied_slots, slot_asset_specs)
            for slot_name in occupied_slots
        }
        if len(set(slot_descriptions.values())) != len(slot_descriptions):
            return None

        reference_slot = layout["seed_reference_slot"]
        target_slot = layout["seed_target_slot"]
        if reference_slot not in slot_descriptions or target_slot not in occupied_slots:
            return None

        relation = self._relation_geometry(reference_slot, target_slot)
        if relation is None:
            return None

        reference_phrase = slot_descriptions[reference_slot]
        return {
            "reference_slot": reference_slot,
            "target_slot": target_slot,
            "reference_phrase": reference_phrase,
            "relation_phrase": self._relation_phrase(
                relation["relation_kind"],
                relation["dx_sign"],
                relation["dy_sign"],
                reference_phrase,
            ),
            "relation_kind": relation["relation_kind"],
            "direction_key": layout["direction_key"],
            "slot_descriptions": slot_descriptions,
            "clarification_phrase": reference_phrase,
            "slot_asset_aliases": {
                slot_name: slot_asset_specs[slot_name]["asset_alias"]
                for slot_name in occupied_slots
            },
        }

    def _create_asset_object(self, pose, slot_name, asset_spec):
        if asset_spec.get("primitive_kind") == "box":
            obj = create_box(
                scene=self.scene,
                pose=pose,
                half_size=asset_spec["half_size"],
                color=asset_spec["color"],
                name=f"{asset_spec['asset_alias']}_{slot_name}",
            )
            obj.set_mass(0.05)
            self.add_prohibit_area(obj, padding=0.05)
            return obj
        return PickRelativePositionBase._create_asset_object(self, pose, slot_name, asset_spec)

    def load_actors(self):
        layout = None
        metadata = None
        objects_by_slot = None
        desired_direction_key = self._desired_direction_key()

        max_candidates = 32 if self.object_count <= 4 else 48
        for _ in range(max_candidates):
            candidate_layout = self._sample_layout(desired_direction_key)

            prohibit_area_length = len(self.prohibited_area)
            candidate_objects = {}
            for slot_name in candidate_layout["occupied_slots"]:
                candidate_objects[slot_name] = self._create_asset_object(
                    candidate_layout["poses"][slot_name],
                    slot_name,
                    candidate_layout["slot_asset_specs"][slot_name],
                )

            if not self._is_visually_consistent_rendered_layout(candidate_layout, candidate_objects):
                self._remove_candidate_objects(candidate_objects, prohibit_area_length)
                continue

            candidate_metadata = self._build_layout_metadata(candidate_layout)
            if candidate_metadata is not None:
                layout = candidate_layout
                metadata = candidate_metadata
                objects_by_slot = candidate_objects
                break

            self._remove_candidate_objects(candidate_objects, prohibit_area_length)

        if layout is None or metadata is None or objects_by_slot is None:
            raise RuntimeError("Failed to sample an unambiguous hetero relative layout")

        self.slot_asset_specs = {
            slot_name: dict(layout["slot_asset_specs"][slot_name])
            for slot_name in layout["occupied_slots"]
        }
        self.occupied_slots = list(layout["occupied_slots"])
        self.reference_slot = metadata["reference_slot"]
        self.target_slot = metadata["target_slot"]
        self.distractor_slots = [
            slot_name for slot_name in self.occupied_slots if slot_name not in {self.reference_slot, self.target_slot}
        ]
        self.slot_descriptions = metadata["slot_descriptions"]
        self.reference_phrase = metadata["reference_phrase"]
        self.clarification_phrase = metadata["clarification_phrase"]
        self.relation_phrase = metadata["relation_phrase"]
        self.relation_kind = metadata["relation_kind"]
        self.direction_key = metadata["direction_key"]

        self.objects_by_slot = objects_by_slot
        self.reference_object = self.objects_by_slot[self.reference_slot]
        self.target_object = self.objects_by_slot[self.target_slot]
        self.target_asset = self.slot_asset_specs[self.target_slot]
        # Keep the base class info layout contract intact for downstream logging.
        self.scene_asset = dict(self.target_asset)
        self.distractor_objects = [self.objects_by_slot[slot_name] for slot_name in self.distractor_slots]

        target_x = float(self.target_object.get_pose().p[0])
        reference_x = float(self.reference_object.get_pose().p[0])
        if abs(target_x) >= 0.02:
            arm_name = "right" if target_x > 0 else "left"
        else:
            arm_name = "right" if reference_x >= 0 else "left"

        self.arm_tag = ArmTag(arm_name)
        self.start_z = float(self.target_object.get_pose().p[2])

    def play_once(self):
        info = super().play_once()
        info["layout"]["slot_asset_aliases"] = {
            slot_name: self.slot_asset_specs[slot_name]["asset_alias"]
            for slot_name in self.occupied_slots
        }
        info["layout"]["slot_assets"] = {
            slot_name: self.slot_asset_specs[slot_name]["asset_key"]
            for slot_name in self.occupied_slots
        }
        info["layout"]["target_asset_alias"] = self.target_asset["asset_alias"]
        return info
