import glob
import os

import numpy as np
import sapien

from ._base_task import Base_Task
from .stable_simple_pick_assets import (
    build_interleaved_stable_asset_catalog,
    stable_asset_catalog_index,
)
from .utils import *


# Default orientation used when spawning sampled assets.
ASSET_QPOS = [0.5, 0.5, 0.5, 0.5]

# Table height used for object placement.
TABLE_Z = 0.741

# Grid slot definitions for different numbers of objects.
# Each slot is mapped to a discrete grid coordinate.
GRID_MODE_SLOT_TO_COORD = {
    "2x2": {
        "front_left": (0, 1),
        "front_right": (1, 1),
        "back_left": (0, 0),
        "back_right": (1, 0),
    },
    "2x3": {
        "front_left": (0, 1),
        "front_middle": (1, 1),
        "front_right": (2, 1),
        "back_left": (0, 0),
        "back_middle": (1, 0),
        "back_right": (2, 0),
    },
    "3x3": {
        "front_left": (0, 2),
        "front_middle": (1, 2),
        "front_right": (2, 2),
        "middle_left": (0, 1),
        "center": (1, 1),
        "middle_right": (2, 1),
        "back_left": (0, 0),
        "back_middle": (1, 0),
        "back_right": (2, 0),
    },
}

# Slot order is sorted from front to back, then left to right.
GRID_MODE_SLOT_ORDER = {
    mode: [
        slot_name
        for slot_name, _ in sorted(
            slot_to_coord.items(),
            key=lambda item: (-item[1][1], item[1][0]),
        )
    ]
    for mode, slot_to_coord in GRID_MODE_SLOT_TO_COORD.items()
}

# Relative direction candidates used to define the target object
# with respect to a reference object.
DIRECTION_CYCLE = [
    "left",
    "right",
    "front",
    "back",
    "front_left",
    "front_right",
    "back_left",
    "back_right",
]

# Convert direction names into grid-coordinate deltas.
DIRECTION_TO_DELTA = {
    "left": (-1, 0),
    "right": (1, 0),
    "front": (0, 1),
    "back": (0, -1),
    "front_left": (-1, 1),
    "front_right": (1, 1),
    "back_left": (-1, -1),
    "back_right": (1, -1),
}

# Head-camera layout validation thresholds.
HEAD_CAMERA_MARGIN_X_RATIO = 0.06
HEAD_CAMERA_MARGIN_Y_RATIO = 0.08
HEAD_CAMERA_MAX_COLUMN_SPREAD_RATIO = 0.12
HEAD_CAMERA_MAX_ROW_SPREAD_RATIO = 0.11
HEAD_CAMERA_MIN_COLUMN_GAP_RATIO = 0.10
HEAD_CAMERA_MIN_ROW_GAP_RATIO = 0.08
HEAD_CAMERA_MIN_OBJECT_PIXELS = 40
HEAD_CAMERA_SETTLE_STEPS = 180


class PickRelativePositionBase(Base_Task):
    # Subclasses should set object_count to one of {2, 3, 4, 6, 9}.
    object_count = None

    # Shared class-level cache for stable asset sampling.
    asset_catalog_cache = None

    def setup_demo(self, **kwags):
        # Initialize the task environment.
        super()._init_task_env_(**kwags)

    def _valid_object_counts(self):
        # Supported object counts and corresponding grid modes.
        return {2, 3, 4, 6, 9}

    def _grid_mode(self):
        # Select a grid layout based on the number of objects.
        if self.object_count in {2, 3, 4}:
            return "2x2"
        if self.object_count == 6:
            return "2x3"
        if self.object_count == 9:
            return "3x3"
        raise ValueError(f"object_count must be one of {sorted(self._valid_object_counts())}")

    def _slot_to_grid_map(self):
        # Return slot-to-coordinate mapping for the current grid mode.
        return GRID_MODE_SLOT_TO_COORD[self._grid_mode()]

    def _grid_to_slot_map(self):
        # Build the inverse coordinate-to-slot mapping.
        return {value: key for key, value in self._slot_to_grid_map().items()}

    def _slot_names(self):
        # Return deterministic slot names for the current grid mode.
        return list(GRID_MODE_SLOT_ORDER[self._grid_mode()])

    def _slot_sort_key(self, slot_name):
        # Sort slots from front to back and left to right.
        grid_x, grid_y = self._slot_to_grid_map()[slot_name]
        return (-grid_y, grid_x)

    def _grid_axes(self):
        # Return sorted column indices and row indices.
        slot_to_coord = self._slot_to_grid_map()
        columns = sorted({grid_x for grid_x, _ in slot_to_coord.values()})
        rows = sorted({grid_y for _, grid_y in slot_to_coord.values()}, reverse=True)
        return columns, rows

    def _axis_index_maps(self):
        # Build maps from grid coordinates to compact axis indices.
        columns, rows = self._grid_axes()
        return (
            {grid_x: idx for idx, grid_x in enumerate(columns)},
            {grid_y: idx for idx, grid_y in enumerate(rows)},
        )

    def _slot_center_offset(self, slot_name):
        # Compute the slot's offset relative to the center of the grid.
        slot_to_coord = self._slot_to_grid_map()
        columns, rows = self._grid_axes()
        mean_x = float(np.mean(columns))
        mean_y = float(np.mean(rows))
        grid_x, grid_y = slot_to_coord[slot_name]
        return float(grid_x - mean_x), float(grid_y - mean_y)

    def _slot_label(self, slot_name):
        # Convert internal slot names to instruction-friendly labels.
        return slot_name.replace("_", "-")

    def _get_available_model_ids(self, modelname):
        # Scan asset metadata files and collect available model IDs.
        asset_path = os.path.join("assets/objects", modelname)
        json_files = sorted(glob.glob(os.path.join(asset_path, "model_data*.json")))

        model_ids = []
        for file_path in json_files:
            base_name = os.path.basename(file_path)

            # The default model_data.json file corresponds to model_id=None.
            if base_name == "model_data.json":
                model_ids.append(None)
                continue

            # Parse model_data{id}.json into an integer model id.
            try:
                model_ids.append(int(base_name.replace("model_data", "").replace(".json", "")))
            except ValueError:
                continue
        return model_ids

    def _build_asset_catalog(self):
        # Reuse a cached asset catalog if available.
        cache = self.__class__.asset_catalog_cache
        if cache is not None:
            return cache

        # Build a stable interleaved catalog of available assets.
        catalog = build_interleaved_stable_asset_catalog(self._get_available_model_ids)
        self.__class__.asset_catalog_cache = catalog
        return catalog

    def _sample_scene_asset(self):
        # Deterministically select one asset for this task episode.
        catalog = self._build_asset_catalog()
        asset_idx = stable_asset_catalog_index(self.task_name, self.ep_num, len(catalog))
        return dict(catalog[asset_idx])

    def _create_asset_object(self, pose, slot_name, asset_spec):
        # Create a physical object actor for a sampled slot.
        obj = create_actor(
            scene=self,
            pose=pose,
            modelname=asset_spec["modelname"],
            convex=True,
            model_id=asset_spec["model_id"],
        )
        if obj is None:
            raise RuntimeError(f"Failed to create asset for {slot_name}: {asset_spec['asset_key']}")

        # Use a light mass and reserve nearby space.
        obj.set_mass(0.05)
        self.add_prohibit_area(obj, padding=0.045)
        return obj

    def _apply_category_noun(self, phrase):
        # Hook for subclasses to insert category-specific nouns.
        return str(phrase)

    def _desired_direction_key(self):
        # Choose a deterministic direction based on task name, seed, and episode index.
        task_bias = sum(ord(ch) for ch in str(getattr(self, 'task_name', 'relative')))
        direction_seed = task_bias * 10007 + int(getattr(self, 'seed_value', 0)) * 1009 + int(self.ep_num) * 9173
        direction_order = list(np.random.default_rng(direction_seed).permutation(DIRECTION_CYCLE))
        return direction_order[self.ep_num % len(direction_order)]

    def _layout_sample_budget(self):
        # More objects require more layout sampling attempts.
        if self.object_count == 2:
            return 24
        if self.object_count == 3:
            return 60
        if self.object_count == 4:
            return 90
        if self.object_count == 6:
            return 140
        return 200

    def _scene_candidate_budget(self):
        # Number of full candidate scenes to test before giving up.
        if self.object_count == 2:
            return 12
        if self.object_count == 3:
            return 36
        if self.object_count == 4:
            return 54
        if self.object_count == 6:
            return 84
        return 120

    def _layout_world_bounds(self):
        # Workspace bounds are adjusted by grid size.
        mode = self._grid_mode()
        if mode == "2x2":
            return (-0.195, 0.195), (-0.190, 0.030)
        if mode == "2x3":
            return (-0.225, 0.225), (-0.205, 0.050)
        return (-0.230, 0.230), (-0.220, 0.080)

    def _layout_gap_ranges(self):
        # Randomized grid spacing ranges for each grid mode.
        mode = self._grid_mode()
        if mode == "2x2":
            return (0.145, 0.215), (0.120, 0.175)
        if mode == "2x3":
            return (0.105, 0.165), (0.115, 0.165)
        return (0.090, 0.140), (0.085, 0.130)

    def _layout_shift_ranges(self):
        # Per-column and per-row jitter magnitudes.
        mode = self._grid_mode()
        if mode == "2x2":
            return 0.016, 0.014
        if mode == "2x3":
            return 0.018, 0.014
        return 0.020, 0.018

    def _valid_reference_slots(self, direction_key):
        # A reference slot is valid only if the target slot exists after applying the direction delta.
        dx, dy = DIRECTION_TO_DELTA[direction_key]
        slot_to_coord = self._slot_to_grid_map()
        grid_to_slot = self._grid_to_slot_map()
        valid_slots = []
        for slot_name, (grid_x, grid_y) in slot_to_coord.items():
            if (grid_x + dx, grid_y + dy) in grid_to_slot:
                valid_slots.append(slot_name)
        return valid_slots

    def _sample_layout(self, direction_key):
        # Sample a grid-based object layout with a valid reference-target relation.
        all_slots = self._slot_names()
        valid_reference_slots = self._valid_reference_slots(direction_key)
        if not valid_reference_slots:
            raise RuntimeError(f"No valid reference slots for direction {direction_key}")

        (x_min_bound, x_max_bound), (y_min_bound, y_max_bound) = self._layout_world_bounds()
        gap_x_range, gap_y_range = self._layout_gap_ranges()
        column_shift_mag, row_shift_mag = self._layout_shift_ranges()
        slot_to_coord = self._slot_to_grid_map()
        grid_to_slot = self._grid_to_slot_map()

        for _ in range(self._layout_sample_budget()):
            # Select a reference slot, then determine the target slot by direction.
            reference_slot = str(np.random.choice(valid_reference_slots))
            ref_x, ref_y = slot_to_coord[reference_slot]
            delta_x, delta_y = DIRECTION_TO_DELTA[direction_key]
            target_slot = grid_to_slot[(ref_x + delta_x, ref_y + delta_y)]

            # Choose distractor slots when the scene contains more than two objects.
            remaining_slots = [
                slot_name
                for slot_name in all_slots
                if slot_name not in {reference_slot, target_slot}
            ]
            distractor_slots = []
            if self.object_count > 2:
                distractor_slots = [
                    str(slot_name)
                    for slot_name in np.random.choice(
                        remaining_slots,
                        size=self.object_count - 2,
                        replace=False,
                    )
                ]

            # Sort occupied slots for deterministic metadata and placement order.
            occupied_slots = [reference_slot, target_slot] + distractor_slots
            occupied_slots.sort(key=self._slot_sort_key)

            # Randomize grid spacing.
            grid_gap_x = float(np.random.uniform(*gap_x_range))
            grid_gap_y = float(np.random.uniform(*gap_y_range))

            # Add small shifts to used rows and columns to avoid perfectly rigid grids.
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

            # Convert discrete grid slots into continuous local offsets.
            slot_offsets = {}
            for slot_name in occupied_slots:
                offset_x, offset_y = self._slot_center_offset(slot_name)
                grid_x, grid_y = slot_to_coord[slot_name]
                slot_offsets[slot_name] = (
                    offset_x * grid_gap_x + column_shift.get(grid_x, 0.0),
                    offset_y * grid_gap_y + row_shift.get(grid_y, 0.0),
                )

            # Pick a global anchor that keeps all objects inside workspace bounds.
            offset_xs = [offset[0] for offset in slot_offsets.values()]
            offset_ys = [offset[1] for offset in slot_offsets.values()]
            anchor_x_min = x_min_bound - max(offset_xs)
            anchor_x_max = x_max_bound - min(offset_xs)
            anchor_y_min = y_min_bound - max(offset_ys)
            anchor_y_max = y_max_bound - min(offset_ys)
            if anchor_x_min >= anchor_x_max or anchor_y_min >= anchor_y_max:
                continue

            anchor_x = float(np.random.uniform(anchor_x_min, anchor_x_max))
            anchor_y = float(np.random.uniform(anchor_y_min, anchor_y_max))

            # Build fixed poses and world positions for all occupied slots.
            poses = {}
            positions = {}
            for slot_name in occupied_slots:
                offset_x, offset_y = slot_offsets[slot_name]
                pos_x = anchor_x + offset_x
                pos_y = anchor_y + offset_y
                poses[slot_name] = rand_pose(
                    xlim=[pos_x, pos_x],
                    ylim=[pos_y, pos_y],
                    zlim=[TABLE_Z, TABLE_Z],
                    qpos=ASSET_QPOS,
                    rotate_rand=True,
                    rotate_lim=[0, float(np.pi), 0],
                )
                positions[slot_name] = np.array([pos_x, pos_y, TABLE_Z], dtype=float)

            return {
                "direction_key": direction_key,
                "seed_reference_slot": reference_slot,
                "seed_target_slot": target_slot,
                "occupied_slots": occupied_slots,
                "poses": poses,
                "positions": positions,
            }

        raise RuntimeError(f"Failed to sample a valid {self.object_count}-object relative layout")

    def _compressed_column_labels(self, used_columns):
        # Assign readable labels to only the columns that are currently occupied.
        if len(used_columns) == 1:
            labels = ["middle"]
        elif len(used_columns) == 2:
            labels = ["left", "right"]
        else:
            labels = ["left", "middle", "right"]
        return {grid_x: labels[idx] for idx, grid_x in enumerate(sorted(used_columns))}

    def _column_slots(self, grid_x, occupied_slots):
        # Return occupied slots in one column, sorted front to back.
        slot_to_coord = self._slot_to_grid_map()
        return sorted(
            [slot_name for slot_name in occupied_slots if slot_to_coord[slot_name][0] == grid_x],
            key=lambda slot_name: -slot_to_coord[slot_name][1],
        )

    def _row_slots(self, grid_y, occupied_slots):
        # Return occupied slots in one row, sorted left to right.
        slot_to_coord = self._slot_to_grid_map()
        return sorted(
            [slot_name for slot_name in occupied_slots if slot_to_coord[slot_name][1] == grid_y],
            key=lambda slot_name: slot_to_coord[slot_name][0],
        )

    def _describe_slot_2x2(self, slot_name, occupied_slots):
        # Generate a unique phrase for a slot in a 2x2 layout.
        slot_to_coord = self._slot_to_grid_map()
        used_columns = sorted({slot_to_coord[slot][0] for slot in occupied_slots})
        used_rows = sorted({slot_to_coord[slot][1] for slot in occupied_slots}, reverse=True)
        grid_x, grid_y = slot_to_coord[slot_name]

        # If all objects are in one column, distinguish them by front/back.
        if len(used_columns) == 1:
            column_slots = self._column_slots(grid_x, occupied_slots)
            if len(column_slots) == 1:
                return self._apply_category_noun("the object")
            prefix = "front" if column_slots.index(slot_name) == 0 else "back"
            return self._apply_category_noun(f"the {prefix} object")

        # If all objects are in one row, distinguish them by left/right.
        if len(used_rows) == 1:
            row_slots = self._row_slots(grid_y, occupied_slots)
            if len(row_slots) == 1:
                return self._apply_category_noun("the object")
            label = "left" if row_slots.index(slot_name) == 0 else "right"
            return self._apply_category_noun(f"the {label} object")

        # Otherwise, use a row-column phrase such as front-left object.
        column_labels = self._compressed_column_labels(used_columns)
        row_label = "front" if grid_y == used_rows[0] else "back"
        return self._apply_category_noun(f"the {row_label}-{column_labels[grid_x]} object")

    def _describe_slot_2x3(self, slot_name, occupied_slots):
        # Generate a unique phrase for a slot in a 2x3 layout.
        slot_to_coord = self._slot_to_grid_map()
        used_columns = sorted({slot_to_coord[slot][0] for slot in occupied_slots})
        grid_x, _ = slot_to_coord[slot_name]
        column_slots = self._column_slots(grid_x, occupied_slots)
        column_rank = column_slots.index(slot_name)
        column_labels = self._compressed_column_labels(used_columns)
        column_label = column_labels[grid_x]

        # If only one column is used, describe by front/back.
        if len(used_columns) == 1:
            if len(column_slots) == 1:
                return self._apply_category_noun("the object")
            prefix = "front" if column_rank == 0 else "back"
            return self._apply_category_noun(f"the {prefix} object")

        # If the column has only one object, the column label is enough.
        if len(column_slots) == 1:
            return self._apply_category_noun(f"the {column_label} object")

        # Otherwise, combine front/back with column label.
        prefix = "front" if column_rank == 0 else "back"
        return self._apply_category_noun(f"the {prefix}-{column_label} object")

    def _describe_slot_3x3(self, slot_name):
        # Generate a slot phrase for a full 3x3 grid.
        if slot_name == "center":
            return self._apply_category_noun("the center object")
        return self._apply_category_noun(f"the {self._slot_label(slot_name)} object")

    def _describe_slot(self, slot_name, occupied_slots):
        # Dispatch slot description based on grid mode.
        mode = self._grid_mode()
        if mode == "2x2":
            return self._describe_slot_2x2(slot_name, occupied_slots)
        if mode == "2x3":
            return self._describe_slot_2x3(slot_name, occupied_slots)
        return self._describe_slot_3x3(slot_name)

    def _describe_all_slots(self, occupied_slots):
        # Build descriptions for every occupied slot.
        slot_descriptions = {
            slot_name: self._describe_slot(slot_name, occupied_slots)
            for slot_name in occupied_slots
        }

        # If descriptions are not unique, force the caller to use a fallback scheme.
        if len(set(slot_descriptions.values())) != len(slot_descriptions):
            return None
        return slot_descriptions

    def _fallback_describe_slots(self, occupied_slots):
        # Fallback descriptions always use the explicit slot label.
        return {
            slot_name: self._apply_category_noun(f"the {self._slot_label(slot_name)} object")
            for slot_name in occupied_slots
        }

    def _relation_geometry(self, reference_slot, target_slot):
        # Compute the geometric relation from the reference slot to the target slot.
        slot_to_coord = self._slot_to_grid_map()
        column_index, row_index = self._axis_index_maps()
        ref_x, ref_y = slot_to_coord[reference_slot]
        target_x, target_y = slot_to_coord[target_slot]
        dx = target_x - ref_x
        dy = target_y - ref_y

        # A slot cannot be related to itself.
        if dx == 0 and dy == 0:
            return None

        dx_sign = 0 if dx == 0 else int(np.sign(dx))
        dy_sign = 0 if dy == 0 else int(np.sign(dy))

        # Detect adjacent horizontal or vertical relations.
        direct_x = (
            target_y == ref_y
            and abs(column_index[target_x] - column_index[ref_x]) == 1
        )
        direct_y = (
            target_x == ref_x
            and abs(row_index[target_y] - row_index[ref_y]) == 1
        )

        # Choose relation wording type.
        if direct_x:
            relation_kind = "x" if self.object_count == 2 else "direct_x"
        elif direct_y:
            relation_kind = "y" if self.object_count == 2 else "direct_y"
        elif dx != 0 and dy != 0:
            relation_kind = "xy"
        elif dx != 0:
            relation_kind = "x"
        else:
            relation_kind = "y"

        return {
            "dx_sign": dx_sign,
            "dy_sign": dy_sign,
            "relation_kind": relation_kind,
        }

    def _relation_phrase(self, kind, dx_sign, dy_sign, reference_phrase):
        # Convert relation geometry into natural-language instruction text.
        if kind == "direct_x":
            direction = "left" if dx_sign < 0 else "right"
            return f"directly to the {direction} of {reference_phrase}"
        if kind == "direct_y":
            return f"directly in front of {reference_phrase}" if dy_sign > 0 else f"directly at the back of {reference_phrase}"
        if kind == "x":
            direction = "left" if dx_sign < 0 else "right"
            return f"to the {direction} of {reference_phrase}"
        if kind == "y":
            return f"in front of {reference_phrase}" if dy_sign > 0 else f"at the back of {reference_phrase}"

        # Diagonal relation.
        front_back = "front" if dy_sign > 0 else "back"
        left_right = "left" if dx_sign < 0 else "right"
        return f"to the {front_back}-{left_right} of {reference_phrase}"

    def _build_layout_metadata(self, layout):
        # Build all metadata needed for instruction generation and target selection.
        occupied_slots = list(layout["occupied_slots"])
        slot_descriptions = self._describe_all_slots(occupied_slots)
        if slot_descriptions is None:
            slot_descriptions = self._fallback_describe_slots(occupied_slots)

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
        }

    def _head_camera_projection_config(self):
        # Collect camera matrices and image size for projection checks.
        head_camera_id = getattr(self.cameras, "head_camera_id", None)
        if head_camera_id is None:
            return None

        head_camera = self.cameras.static_camera_list[head_camera_id]
        camera_config = self.cameras.static_camera_config[head_camera_id]
        return {
            "camera": head_camera,
            "intrinsic": np.asarray(head_camera.get_intrinsic_matrix(), dtype=float),
            "extrinsic": np.asarray(head_camera.get_extrinsic_matrix(), dtype=float),
            "width": int(camera_config["w"]),
            "height": int(camera_config["h"]),
        }

    def _project_positions_to_head_camera(self, positions_by_slot):
        # Project world positions into head-camera pixel coordinates.
        camera_config = self._head_camera_projection_config()
        if camera_config is None:
            return None

        projected = {}
        intrinsic = camera_config["intrinsic"]
        extrinsic = camera_config["extrinsic"]

        for slot_name, position in positions_by_slot.items():
            # Convert world point to homogeneous coordinates.
            world_point = np.concatenate([np.asarray(position, dtype=float), [1.0]])

            # Transform into camera coordinates.
            camera_point = extrinsic @ world_point
            depth = float(camera_point[2])
            if depth <= 1e-6:
                return None

            # Apply intrinsics and normalize by depth.
            pixel = intrinsic @ camera_point[:3]
            projected[slot_name] = np.array(
                [float(pixel[0] / depth), float(pixel[1] / depth), depth],
                dtype=float,
            )

        return {
            "points": projected,
            "width": camera_config["width"],
            "height": camera_config["height"],
        }

    def _axis_groups_with_projection(self, occupied_slots, projected_points, axis):
        # Group projected pixel coordinates by grid column or row.
        slot_to_coord = self._slot_to_grid_map()
        index = 0 if axis == "x" else 1
        groups = {}
        for slot_name in occupied_slots:
            grid_value = slot_to_coord[slot_name][index]
            groups.setdefault(grid_value, []).append(float(projected_points[slot_name][index]))
        return groups

    def _visual_group_params(self):
        # Use looser visual grouping thresholds for smaller grids.
        mode = self._grid_mode()
        if mode == "2x2":
            return {
                "max_column_spread_ratio": 0.20,
                "max_row_spread_ratio": 0.18,
                "min_column_gap_ratio": 0.06,
                "min_row_gap_ratio": 0.06,
            }
        if mode == "2x3":
            return {
                "max_column_spread_ratio": 0.16,
                "max_row_spread_ratio": 0.14,
                "min_column_gap_ratio": 0.05,
                "min_row_gap_ratio": 0.05,
            }
        return {
            "max_column_spread_ratio": HEAD_CAMERA_MAX_COLUMN_SPREAD_RATIO,
            "max_row_spread_ratio": HEAD_CAMERA_MAX_ROW_SPREAD_RATIO,
            "min_column_gap_ratio": HEAD_CAMERA_MIN_COLUMN_GAP_RATIO,
            "min_row_gap_ratio": HEAD_CAMERA_MIN_ROW_GAP_RATIO,
        }

    def _is_visually_consistent_layout(self, layout):
        # Check whether the projected slot layout is visually consistent in the head camera.
        projection = self._project_positions_to_head_camera(layout["positions"])
        if projection is None:
            return False

        projected_points = projection["points"]
        width = float(projection["width"])
        height = float(projection["height"])
        margin_x = width * HEAD_CAMERA_MARGIN_X_RATIO
        margin_y = height * HEAD_CAMERA_MARGIN_Y_RATIO
        min_x = margin_x
        max_x = width - margin_x
        min_y = margin_y
        max_y = height - margin_y

        # All projected object centers must lie inside the image margins.
        for slot_name in layout["occupied_slots"]:
            pixel_x, pixel_y = projected_points[slot_name][:2]
            if not (min_x <= pixel_x <= max_x and min_y <= pixel_y <= max_y):
                return False

        # For small layouts, margin visibility is enough.
        if self.object_count <= 4:
            return True

        # For larger layouts, also validate row/column visual consistency.
        visual_params = self._visual_group_params()
        column_groups = self._axis_groups_with_projection(layout["occupied_slots"], projected_points, "x")
        row_groups = self._axis_groups_with_projection(layout["occupied_slots"], projected_points, "y")
        max_column_spread = width * visual_params["max_column_spread_ratio"]
        max_row_spread = height * visual_params["max_row_spread_ratio"]
        min_column_gap = width * visual_params["min_column_gap_ratio"]
        min_row_gap = height * visual_params["min_row_gap_ratio"]

        # Objects in the same grid column should appear close in image x.
        for values in column_groups.values():
            if len(values) > 1 and (max(values) - min(values)) > max_column_spread:
                return False

        # Objects in the same grid row should appear close in image y.
        for values in row_groups.values():
            if len(values) > 1 and (max(values) - min(values)) > max_row_spread:
                return False

        # Different columns and rows should be separated clearly.
        column_centers = sorted(float(np.mean(values)) for values in column_groups.values())
        row_centers = sorted(float(np.mean(values)) for values in row_groups.values())
        if any((right - left) < min_column_gap for left, right in zip(column_centers, column_centers[1:])):
            return False
        if any((lower - upper) < min_row_gap for upper, lower in zip(row_centers, row_centers[1:])):
            return False
        return True

    def _rendered_object_centroids(self, objects_by_slot):
        # Render segmentation and compute each object's visible centroid.
        head_camera_info = self._head_camera_projection_config()
        if head_camera_info is None:
            return None

        # Let physics and rendering settle before reading segmentation.
        for _ in range(HEAD_CAMERA_SETTLE_STEPS):
            self.scene.step()
        self._update_render()
        self.cameras.update_picture()
        head_camera = head_camera_info["camera"]
        segmentation = np.asarray(head_camera.get_picture("Segmentation")[..., 1], dtype=np.int32)

        centroids = {}
        for slot_name, obj in objects_by_slot.items():
            actor_id = int(obj.actor.get_per_scene_id())
            mask = segmentation == actor_id

            # Reject if the object is too small or invisible in segmentation.
            if int(mask.sum()) < HEAD_CAMERA_MIN_OBJECT_PIXELS:
                return None

            pixel_y, pixel_x = np.nonzero(mask)
            centroids[slot_name] = np.array(
                [float(np.mean(pixel_x)), float(np.mean(pixel_y))],
                dtype=float,
            )

        return {
            "points": centroids,
            "width": int(head_camera_info["width"]),
            "height": int(head_camera_info["height"]),
        }

    def _is_visually_consistent_rendered_layout(self, layout, objects_by_slot):
        # Validate the rendered object centroids, not just projected object centers.
        rendered = self._rendered_object_centroids(objects_by_slot)
        if rendered is None:
            return False

        rendered_points = rendered["points"]
        width = float(rendered["width"])
        height = float(rendered["height"])
        margin_x = width * HEAD_CAMERA_MARGIN_X_RATIO
        margin_y = height * HEAD_CAMERA_MARGIN_Y_RATIO
        min_x = margin_x
        max_x = width - margin_x
        min_y = margin_y
        max_y = height - margin_y

        # Every rendered centroid must be inside the camera margins.
        for slot_name in layout["occupied_slots"]:
            pixel_x, pixel_y = rendered_points[slot_name]
            if not (min_x <= pixel_x <= max_x and min_y <= pixel_y <= max_y):
                return False
        return True

    def _remove_candidate_objects(self, objects_by_slot, prohibit_area_length):
        # Remove rejected candidate objects and restore prohibited-area state.
        for obj in objects_by_slot.values():
            obj.actor.remove_from_scene()
        del self.prohibited_area[prohibit_area_length:]
        self.scene.step()
        self.scene.update_render()

    def load_actors(self):
        # Search for a layout that is geometrically valid, visually clear, and unambiguous.
        layout = None
        metadata = None
        objects_by_slot = None
        desired_direction_key = self._desired_direction_key()
        self.scene_asset = self._sample_scene_asset()

        for _ in range(self._scene_candidate_budget()):
            candidate_layout = self._sample_layout(desired_direction_key)

            # First validate the ideal projected layout.
            if not self._is_visually_consistent_layout(candidate_layout):
                continue

            # Temporarily create candidate objects for rendered validation.
            prohibit_area_length = len(self.prohibited_area)
            candidate_objects = {}
            for slot_name in candidate_layout["occupied_slots"]:
                candidate_objects[slot_name] = self._create_asset_object(
                    candidate_layout["poses"][slot_name],
                    slot_name,
                    self.scene_asset,
                )

            # Validate the actual rendered centroids after object creation.
            if not self._is_visually_consistent_rendered_layout(candidate_layout, candidate_objects):
                self._remove_candidate_objects(candidate_objects, prohibit_area_length)
                continue

            # Build language metadata and accept the scene if it is unambiguous.
            candidate_metadata = self._build_layout_metadata(candidate_layout)
            if candidate_metadata is not None:
                layout = candidate_layout
                metadata = candidate_metadata
                objects_by_slot = candidate_objects
                break

            # Remove candidate objects if the metadata is ambiguous.
            self._remove_candidate_objects(candidate_objects, prohibit_area_length)

        if layout is None or metadata is None or objects_by_slot is None:
            raise RuntimeError("Failed to sample an unambiguous relative layout")

        # Store accepted layout metadata.
        self.occupied_slots = list(layout["occupied_slots"])
        self.reference_slot = metadata["reference_slot"]
        self.target_slot = metadata["target_slot"]
        self.distractor_slots = [
            slot_name
            for slot_name in self.occupied_slots
            if slot_name not in {self.reference_slot, self.target_slot}
        ]
        self.slot_descriptions = metadata["slot_descriptions"]
        self.reference_phrase = metadata["reference_phrase"]
        self.clarification_phrase = metadata["clarification_phrase"]
        self.relation_phrase = metadata["relation_phrase"]
        self.relation_kind = metadata["relation_kind"]
        self.direction_key = metadata["direction_key"]

        # Store object references by role.
        self.objects_by_slot = objects_by_slot
        self.reference_object = self.objects_by_slot[self.reference_slot]
        self.target_object = self.objects_by_slot[self.target_slot]
        self.distractor_objects = [
            self.objects_by_slot[slot_name] for slot_name in self.distractor_slots
        ]

        # Choose the arm based on target x position, falling back to reference position near center.
        target_x = float(self.target_object.get_pose().p[0])
        reference_x = float(self.reference_object.get_pose().p[0])
        if abs(target_x) >= 0.02:
            arm_name = "right" if target_x > 0 else "left"
        else:
            arm_name = "right" if reference_x >= 0 else "left"

        self.arm_tag = ArmTag(arm_name)

        # Record initial target height for success checking.
        self.start_z = float(self.target_object.get_pose().p[2])

    def _play_once_pre_grasp_dis(self):
        # Distance used before the final grasp.
        return 0.09

    def _play_once_grasp_dis(self):
        # Final grasp distance.
        return 0.01

    def _play_once_lift_z(self):
        # Lift distance for the demonstration.
        return 0.12

    def play_once(self):
        # Grasp the target object using the selected arm.
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

        # Store layout metadata for replay, debugging, or instruction generation.
        self.info["layout"] = {
            "object_count": self.object_count,
            "grid_mode": self._grid_mode(),
            "reference_slot": self.reference_slot,
            "target_slot": self.target_slot,
            "occupied_slots": self.occupied_slots,
            "slot_descriptions": self.slot_descriptions,
            "reference_phrase": self.reference_phrase,
            "clarification_phrase": self.clarification_phrase,
            "relation_kind": self.relation_kind,
            "direction_key": self.direction_key,
            "scene_asset": self.scene_asset["asset_key"],
        }

        # Store natural-language placeholders for the task instruction.
        self.info["info"] = {
            "{R}": self.relation_phrase,
            # "{B}": self.reference_phrase,
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        # Success requires the target object to be lifted while the selected gripper is closed.
        target_pose = self.target_object.get_pose().p
        lifted = (float(target_pose[2]) - self.start_z) > 0.06
        grasped = self.is_left_gripper_close() if self.arm_tag == "left" else self.is_right_gripper_close()
        return bool(lifted and grasped)