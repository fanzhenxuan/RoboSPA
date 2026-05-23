import numpy as np

from .pick_relative_position_base import (
    ASSET_QPOS,
    DIRECTION_TO_DELTA,
    HEAD_CAMERA_MARGIN_X_RATIO,
    HEAD_CAMERA_MARGIN_Y_RATIO,
    TABLE_Z,
    PickRelativePositionBase,
)
from .stable_simple_pick_assets import STABLE_SIMPLE_PICK_ASSET_SPECS
from .utils import ArmTag, create_box, rand_pose


# Human-readable category nouns used in generated instructions.
CATEGORY_NOUNS = {
    "block": "block",
    "stapler": "stapler",
    "can": "can",
    "rubikscube": "rubik's cube",
    "soap": "soap bar",
    "tea_box": "tea box",
    "bread": "bread loaf",
    "phone": "phone",
    "remotecontrol": "remote control",
    "pillbottle": "pill bottle",
    "coffee_box": "coffee box",
}

# Primitive block variants with different colors.
PRIMITIVE_BLOCK_VARIANTS = [
    {"variant_id": 0, "half_size": (0.015, 0.015, 0.015), "color": (0.84, 0.22, 0.18)},
    {"variant_id": 1, "half_size": (0.015, 0.015, 0.015), "color": (0.16, 0.58, 0.82)},
    {"variant_id": 2, "half_size": (0.015, 0.015, 0.015), "color": (0.22, 0.67, 0.28)},
    {"variant_id": 3, "half_size": (0.015, 0.015, 0.015), "color": (0.90, 0.74, 0.18)},
    {"variant_id": 4, "half_size": (0.015, 0.015, 0.015), "color": (0.58, 0.34, 0.80)},
    {"variant_id": 5, "half_size": (0.015, 0.015, 0.015), "color": (0.28, 0.28, 0.28)},
]

# Minimum safe spacing thresholds for each fixed object category.
# x/y gaps help preserve grid readability, while pair gap avoids object overlap.
RELATIVE_SAFE_SPACING = {
    "block": {"x": 0.095, "y": 0.095, "pair": 0.095},
    "stapler": {"x": 0.118, "y": 0.092, "pair": 0.104},
    "can": {"x": 0.102, "y": 0.088, "pair": 0.095},
    "soap": {"x": 0.096, "y": 0.082, "pair": 0.088},
    "tea_box": {"x": 0.108, "y": 0.090, "pair": 0.098},
    "rubikscube": {"x": 0.094, "y": 0.094, "pair": 0.092},
    "bread": {"x": 0.112, "y": 0.092, "pair": 0.100},
    "phone": {"x": 0.116, "y": 0.082, "pair": 0.100},
    "remotecontrol": {"x": 0.124, "y": 0.078, "pair": 0.104},
    "pillbottle": {"x": 0.096, "y": 0.096, "pair": 0.094},
    "coffee_box": {"x": 0.108, "y": 0.090, "pair": 0.098},
}


class PickRelativePositionFixedAssetBaseLyt0310(PickRelativePositionBase):
    # Subclasses should set the object count and fixed asset identity.
    object_count = None
    fixed_modelname = None
    fixed_model_id = None
    fixed_asset_alias = None

    def _category_variant_ids(self):
        # Return all usable variant ids for the fixed category.
        if self.fixed_modelname is None:
            raise RuntimeError("fixed_modelname must be set")

        # Primitive blocks use locally defined variants instead of mesh metadata.
        if self.fixed_modelname == "primitive_block":
            return [variant["variant_id"] for variant in PRIMITIVE_BLOCK_VARIANTS]

        # Find the stable whitelist entry for this fixed model.
        candidate_ids = None
        for modelname, model_ids in STABLE_SIMPLE_PICK_ASSET_SPECS:
            if modelname == self.fixed_modelname:
                candidate_ids = list(model_ids)
                break
        if candidate_ids is None:
            raise RuntimeError(f"{self.fixed_modelname} is not in the stable asset whitelist")

        # Keep only variants that are both whitelisted and actually available on disk.
        available_ids = set(self._get_available_model_ids(self.fixed_modelname))
        variant_ids = [model_id for model_id in candidate_ids if model_id in available_ids]
        if not variant_ids:
            raise RuntimeError(f"No available variants found for {self.fixed_modelname}")
        return variant_ids

    def _category_noun(self):
        # Convert the fixed asset alias/model name into a human-readable noun.
        alias = self.fixed_asset_alias or self.fixed_modelname.split("_", 1)[-1].replace("-", "_")
        return CATEGORY_NOUNS.get(alias, alias.replace("_", " "))

    def _apply_category_noun(self, phrase):
        # Replace the generic word "object" with the category-specific noun.
        noun = self._category_noun()
        return str(phrase).replace("object", noun)

    def _relative_min_safe_gaps(self):
        # Get category-specific safe spacing thresholds.
        alias = self.fixed_asset_alias or self.fixed_modelname.split("_", 1)[-1].replace("-", "_")
        spacing = RELATIVE_SAFE_SPACING.get(alias)

        if spacing is not None:
            min_gap_x = float(spacing["x"])
            min_gap_y = float(spacing["y"])
            min_pair_gap = float(spacing["pair"])
        elif self.fixed_modelname == "primitive_block":
            # Fallback spacing for primitive blocks based on their physical size.
            half_x, half_y, _ = PRIMITIVE_BLOCK_VARIANTS[0]["half_size"]
            size_x = 2.0 * float(half_x)
            size_y = 2.0 * float(half_y)
            min_gap_x = max(0.095, size_x + 0.060)
            min_gap_y = max(0.095, size_y + 0.060)
            min_pair_gap = max(0.095, np.hypot(size_x, size_y) + 0.050)
        else:
            # Generic fallback for unknown categories.
            min_gap_x, min_gap_y, min_pair_gap = 0.095, 0.085, 0.090

        # Scale spacing by grid mode to balance clarity and placement feasibility.
        mode = self._grid_mode()
        if mode == "2x2":
            scale = 0.78
        elif mode == "2x3":
            scale = 0.92
        else:
            scale = 0.86
        return min_gap_x * scale, min_gap_y * scale, min_pair_gap * scale

    def _layout_respects_safe_spacing(self, occupied_slots, positions):
        # Verify that a sampled layout respects category-specific spacing.
        slot_to_coord = self._slot_to_grid_map()
        min_gap_x, min_gap_y, min_pair_gap = self._relative_min_safe_gaps()

        # Compute used grid columns and rows.
        used_columns = sorted({slot_to_coord[slot_name][0] for slot_name in occupied_slots})
        used_rows = sorted({slot_to_coord[slot_name][1] for slot_name in occupied_slots})

        # Compute mean x position for each occupied column.
        column_centers = []
        for grid_x in used_columns:
            values = [
                float(positions[slot_name][0])
                for slot_name in occupied_slots
                if slot_to_coord[slot_name][0] == grid_x
            ]
            column_centers.append(float(np.mean(values)))

        # Compute mean y position for each occupied row.
        row_centers = []
        for grid_y in used_rows:
            values = [
                float(positions[slot_name][1])
                for slot_name in occupied_slots
                if slot_to_coord[slot_name][1] == grid_y
            ]
            row_centers.append(float(np.mean(values)))

        # Neighboring columns must be separated enough in world x.
        if any((right - left) < min_gap_x for left, right in zip(column_centers, column_centers[1:])):
            return False

        # Neighboring rows must be separated enough in world y.
        if any((upper - lower) < min_gap_y for lower, upper in zip(row_centers, row_centers[1:])):
            return False

        # Every pair of objects must also satisfy a radial distance threshold.
        points = [np.asarray(positions[slot_name][:2], dtype=float) for slot_name in occupied_slots]
        for idx, point_i in enumerate(points):
            for point_j in points[idx + 1:]:
                if float(np.linalg.norm(point_i - point_j)) < min_pair_gap:
                    return False
        return True

    def _sample_scene_asset(self):
        # Sample the concrete fixed asset variant for this episode.
        if self.fixed_modelname == "primitive_block":
            # Cycle through primitive block color variants by episode number.
            variant = PRIMITIVE_BLOCK_VARIANTS[int(self.ep_num) % len(PRIMITIVE_BLOCK_VARIANTS)]
            return {
                "modelname": "primitive_block",
                "model_id": variant["variant_id"],
                "asset_key": f"primitive_block/base{variant['variant_id']}",
                "asset_alias": self.fixed_asset_alias or "block",
                "primitive_kind": "box",
                "half_size": variant["half_size"],
                "color": variant["color"],
            }

        # Cycle through stable mesh variants by episode number.
        variant_ids = self._category_variant_ids()
        model_id = int(variant_ids[int(self.ep_num) % len(variant_ids)])
        return {
            "modelname": self.fixed_modelname,
            "model_id": model_id,
            "asset_key": f"{self.fixed_modelname}/base{model_id}",
            "asset_alias": self.fixed_asset_alias or self.fixed_modelname,
        }

    def _create_asset_object(self, pose, slot_name, asset_spec):
        # Create either a primitive block or a regular mesh object.
        if asset_spec.get("primitive_kind") == "box":
            obj = create_box(
                scene=self.scene,
                pose=pose,
                half_size=asset_spec["half_size"],
                color=asset_spec["color"],
                name=f"{self.fixed_asset_alias or 'block'}_{slot_name}",
            )
            obj.set_mass(0.05)
            self.add_prohibit_area(obj, padding=0.05)
            return obj

        # For mesh assets, fall back to the parent actor creation logic.
        return PickRelativePositionBase._create_asset_object(self, pose, slot_name, asset_spec)

    def _play_once_pre_grasp_dis(self):
        # Cans need a slightly larger pre-grasp distance.
        if self.fixed_asset_alias == "can" or self.fixed_modelname == "071_can":
            return 0.11
        return super()._play_once_pre_grasp_dis()

    def _play_once_grasp_dis(self):
        # Cans use a slightly larger final grasp distance.
        if self.fixed_asset_alias == "can" or self.fixed_modelname == "071_can":
            return 0.02
        return super()._play_once_grasp_dis()

    def _play_once_lift_z(self):
        # Lift higher in dense 3x3 scenes for clearer success.
        if self._grid_mode() == "3x3":
            return 0.14
        return super()._play_once_lift_z()

    def _build_layout_from_xy(self, direction_key, reference_slot, target_slot, occupied_slots, xy_positions):
        # Convert sampled 2D positions into full poses and 3D positions.
        poses = {}
        positions = {}

        # Primitive blocks need their z position lifted by their half-height.
        is_primitive_block = self.fixed_modelname == "primitive_block"
        block_half_z = None
        if is_primitive_block:
            asset_spec = self._sample_scene_asset()
            block_half_z = float(asset_spec["half_size"][2])

        for slot_name in occupied_slots:
            pos_x, pos_y = xy_positions[slot_name]

            if is_primitive_block:
                # Primitive blocks are placed upright with small random yaw.
                pos_z = TABLE_Z + block_half_z
                poses[slot_name] = rand_pose(
                    xlim=[pos_x, pos_x],
                    ylim=[pos_y, pos_y],
                    zlim=[pos_z, pos_z],
                    qpos=[1, 0, 0, 0],
                    rotate_rand=True,
                    rotate_lim=[0, 0, 0.75],
                )
            else:
                # Mesh assets use the shared asset quaternion and random rotation.
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
        }

    def _fixed_layout_bounds(self):
        # Workspace bounds for fixed-asset layouts.
        mode = self._grid_mode()
        if mode == "2x2":
            return (-0.205, 0.205), (-0.195, 0.035)
        if mode == "2x3":
            return (-0.225, 0.225), (-0.205, 0.045)
        return (-0.228, 0.228), (-0.215, 0.075)

    def _fixed_gap_ranges(self):
        # Grid gap ranges are based on safe spacing plus a small margin.
        min_gap_x, min_gap_y, _ = self._relative_min_safe_gaps()
        mode = self._grid_mode()

        if mode == "2x2":
            return (
                max(0.090, min_gap_x + 0.006),
                0.185,
            ), (
                max(0.085, min_gap_y + 0.006),
                0.170,
            )

        if mode == "2x3":
            return (
                max(0.088, min_gap_x + 0.006),
                0.160,
            ), (
                max(0.082, min_gap_y + 0.006),
                0.160,
            )

        return (
            max(0.080, min_gap_x + 0.004),
            0.132,
        ), (
            max(0.076, min_gap_y + 0.004),
            0.122,
        )

    def _fixed_shift_ranges(self):
        # Small random row/column shifts make the grid less perfectly regular.
        mode = self._grid_mode()
        if mode == "2x2":
            return 0.014, 0.012
        if mode == "2x3":
            return 0.018, 0.012
        return 0.016, 0.014

    def _sample_layout(self, direction_key):
        # Sample a fixed-asset relative-position layout.
        all_slots = self._slot_names()
        valid_reference_slots = self._valid_reference_slots(direction_key)
        if not valid_reference_slots:
            raise RuntimeError(f"No valid reference slots for direction {direction_key}")

        slot_to_coord = self._slot_to_grid_map()
        grid_to_slot = self._grid_to_slot_map()
        (x_min_bound, x_max_bound), (y_min_bound, y_max_bound) = self._fixed_layout_bounds()
        gap_x_range, gap_y_range = self._fixed_gap_ranges()
        column_shift_mag, row_shift_mag = self._fixed_shift_ranges()

        # Larger scenes need more sampling attempts.
        if self.object_count == 2:
            max_tries = 160
        elif self.object_count in {3, 4}:
            max_tries = 260
        elif self.object_count == 6:
            max_tries = 360
        else:
            max_tries = 520

        for _ in range(max_tries):
            # Pick a reference slot and compute the target slot from the desired direction.
            reference_slot = str(np.random.choice(valid_reference_slots))
            ref_x, ref_y = slot_to_coord[reference_slot]
            delta_x, delta_y = DIRECTION_TO_DELTA[direction_key]
            target_slot = grid_to_slot[(ref_x + delta_x, ref_y + delta_y)]

            # Sample distractor slots if there are more than two objects.
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

            # Sort slots so generated metadata stays deterministic.
            occupied_slots = [reference_slot, target_slot] + distractor_slots
            occupied_slots.sort(key=self._slot_sort_key)

            # Randomly sample grid spacing.
            grid_gap_x = float(np.random.uniform(*gap_x_range))
            grid_gap_y = float(np.random.uniform(*gap_y_range))

            # Randomly offset each used row and column slightly.
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

            # Convert grid slots into local 2D positions before adding the global anchor.
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

            # Choose an anchor so all objects fit within workspace bounds.
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
                slot_name: np.array(
                    [position[0] + anchor_x, position[1] + anchor_y],
                    dtype=float,
                )
                for slot_name, position in xy_positions.items()
            }

            # Build the full layout and validate both spacing and camera visibility.
            layout = self._build_layout_from_xy(
                direction_key,
                reference_slot,
                target_slot,
                occupied_slots,
                shifted_positions,
            )
            if not self._layout_respects_safe_spacing(occupied_slots, layout["positions"]):
                continue
            if not self._is_visually_consistent_layout(layout):
                continue
            return layout

        raise RuntimeError(f"Failed to sample a valid {self.object_count}-object relative layout")

    def _is_visually_consistent_layout(self, layout):
        # Check projected object centers against the head-camera image bounds.
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

        for slot_name in layout["occupied_slots"]:
            pixel_x, pixel_y = projected_points[slot_name][:2]
            if not (min_x <= pixel_x <= max_x and min_y <= pixel_y <= max_y):
                return False
        return True

    def _is_visually_consistent_rendered_layout(self, layout, objects_by_slot):
        # Check actual rendered object centroids against the head-camera image bounds.
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

        for slot_name in layout["occupied_slots"]:
            pixel_x, pixel_y = rendered_points[slot_name]
            if not (min_x <= pixel_x <= max_x and min_y <= pixel_y <= max_y):
                return False
        return True

    def load_actors(self):
        # Search for a valid fixed-asset relative-position scene.
        layout = None
        metadata = None
        objects_by_slot = None
        desired_direction_key = self._desired_direction_key()
        self.scene_asset = self._sample_scene_asset()

        # Debug counters make failures easier to diagnose.
        debug_counts = {
            "sample": 0,
            "layout_visual": 0,
            "rendered": 0,
            "metadata": 0,
        }
        last_error = None

        # Larger scenes receive more candidate attempts.
        max_candidates = 120 if self.object_count <= 4 else 220
        for _ in range(max_candidates):
            try:
                candidate_layout = self._sample_layout(desired_direction_key)
            except RuntimeError as exc:
                debug_counts["sample"] += 1
                last_error = str(exc)
                continue

            # Validate projected layout before creating actors.
            if not self._is_visually_consistent_layout(candidate_layout):
                debug_counts["layout_visual"] += 1
                continue

            # Create candidate actors for rendered validation.
            prohibit_area_length = len(self.prohibited_area)
            candidate_objects = {}
            for slot_name in candidate_layout["occupied_slots"]:
                candidate_objects[slot_name] = self._create_asset_object(
                    candidate_layout["poses"][slot_name],
                    slot_name,
                    self.scene_asset,
                )

            # Reject and remove candidate actors if rendered visibility is poor.
            if not self._is_visually_consistent_rendered_layout(candidate_layout, candidate_objects):
                debug_counts["rendered"] += 1
                self._remove_candidate_objects(candidate_objects, prohibit_area_length)
                continue

            # Build instruction metadata and accept only unambiguous layouts.
            candidate_metadata = self._build_layout_metadata(candidate_layout)
            if candidate_metadata is not None:
                layout = candidate_layout
                metadata = candidate_metadata
                objects_by_slot = candidate_objects
                break

            debug_counts["metadata"] += 1
            self._remove_candidate_objects(candidate_objects, prohibit_area_length)

        # Include debugging details when sampling fails.
        if layout is None or metadata is None or objects_by_slot is None:
            details = (
                f"sample={debug_counts['sample']}, "
                f"layout_visual={debug_counts['layout_visual']}, "
                f"rendered={debug_counts['rendered']}, "
                f"metadata={debug_counts['metadata']}"
            )
            if last_error is not None:
                details += f", last_sample_error={last_error}"
            raise RuntimeError(f"Failed to sample an unambiguous relative layout ({details})")

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

        # Choose the arm based on target x position, using reference x as fallback near center.
        target_x = float(self.target_object.get_pose().p[0])
        reference_x = float(self.reference_object.get_pose().p[0])
        if abs(target_x) >= 0.02:
            arm_name = "right" if target_x > 0 else "left"
        else:
            arm_name = "right" if reference_x >= 0 else "left"

        self.arm_tag = ArmTag(arm_name)

        # Record initial target height for success checking.
        self.start_z = float(self.target_object.get_pose().p[2])

    def play_once(self):
        # Run the parent pick-and-lift demo, then add category metadata.
        info = super().play_once()
        info["layout"]["scene_asset_alias"] = self._category_noun()
        return info