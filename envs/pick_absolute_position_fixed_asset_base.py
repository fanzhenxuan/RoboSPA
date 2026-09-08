import numpy as np

from .pick_absolute_position_base import (
    ASSET_QPOS,
    HEAD_CAMERA_MARGIN_X_RATIO,
    HEAD_CAMERA_MARGIN_Y_RATIO,
    HEAD_CAMERA_MIN_OBJECT_PIXELS,
    HEAD_CAMERA_SETTLE_STEPS,
    POSITION_CYCLE,
    TABLE_Z,
    PickAbsolutePositionMultiviewBase,
)
from .pick_relative_position_fixed_asset_base import CATEGORY_NOUNS, PRIMITIVE_BLOCK_VARIANTS
from .stable_simple_pick_assets import STABLE_SIMPLE_PICK_ASSET_SPECS
from .stable_task6_assets import TASK6_STABLE_CATEGORY_SPECS
from .utils import create_box, rand_pose


FIXED_HEAD_CAMERA_MIN_CENTROID_DIST = 42.0
FIXED_HEAD_CAMERA_MAX_BBOX_OVERLAP_RATIO = 0.40

FIXED_CATEGORY_WORLD_RADIUS = {
    "primitive_block": 0.022,
    "075_bread": 0.048,
    "100_seal": 0.045,
    "057_toycar": 0.042,
    "077_phone": 0.052,
    "079_remotecontrol": 0.042,
    "073_rubikscube": 0.044,
    "073_rubikscube": 0.044,
    "113_coffee-box": 0.050,
    "112_tea-box": 0.044,
    "105_sauce-can": 0.047,
}

FIXED_CATEGORY_WORLD_CLEARANCE = {
    2: 0.016,
    3: 0.013,
    4: 0.011,
    5: 0.009,
    6: 0.008,
    7: 0.007,
}

ABSOLUTE_SAFE_SPACING = {
    "block": {"x": 0.095, "y": 0.095, "pair": 0.095},
    "bread": {"x": 0.112, "y": 0.092, "pair": 0.100},
    "seal": {"x": 0.094, "y": 0.086, "pair": 0.088},
    "toycar": {"x": 0.098, "y": 0.086, "pair": 0.090},
    "phone": {"x": 0.126, "y": 0.094, "pair": 0.112},
    "rubikscube": {"x": 0.098, "y": 0.098, "pair": 0.094},
    "coffee_box": {"x": 0.110, "y": 0.092, "pair": 0.100},
    "tea_box": {"x": 0.098, "y": 0.092, "pair": 0.094},
    "sauce_can": {"x": 0.098, "y": 0.092, "pair": 0.094},
}

LOCAL_CATEGORY_NOUNS = {
    "seal": "seal",
    "toycar": "toy car",
    "teabox": "tea box",
    "sauce_can": "sauce can",
}


class PickAbsolutePositionFixedAssetMultiviewBase(PickAbsolutePositionMultiviewBase):
    object_count = None
    fixed_modelname = None
    fixed_model_id = None
    fixed_asset_alias = None

    def _category_variant_ids(self):
        if self.fixed_modelname is None:
            raise RuntimeError("fixed_modelname must be set")
        if self.fixed_modelname == "primitive_block":
            return [variant["variant_id"] for variant in PRIMITIVE_BLOCK_VARIANTS]

        candidate_ids = None
        for modelname, model_ids in STABLE_SIMPLE_PICK_ASSET_SPECS:
            if modelname == self.fixed_modelname:
                candidate_ids = list(model_ids)
                break
        if candidate_ids is None:
            extra_ids = TASK6_STABLE_CATEGORY_SPECS.get(self.fixed_modelname)
            if extra_ids:
                candidate_ids = list(extra_ids)
        if candidate_ids is None:
            raise RuntimeError(f"{self.fixed_modelname} is not in the stable asset whitelist")

        available_ids = set(self._get_available_model_ids(self.fixed_modelname))
        variant_ids = [model_id for model_id in candidate_ids if model_id in available_ids]
        if not variant_ids:
            raise RuntimeError(f"No available variants found for {self.fixed_modelname}")
        return variant_ids

    def _category_noun(self):
        alias = self.fixed_asset_alias or self.fixed_modelname.split("_", 1)[-1].replace("-", "_")
        return LOCAL_CATEGORY_NOUNS.get(alias, CATEGORY_NOUNS.get(alias, alias.replace("_", " ")))

    def _absolute_min_safe_gaps(self):
        alias = self.fixed_asset_alias or self.fixed_modelname.split("_", 1)[-1].replace("-", "_")
        spacing = ABSOLUTE_SAFE_SPACING.get(alias)
        if spacing is not None:
            min_gap_x = float(spacing["x"])
            min_gap_y = float(spacing["y"])
            min_pair_gap = float(spacing["pair"])
            count = int(self.object_count or 0)
            if count <= 2:
                scale = 0.90
            elif count == 3:
                scale = 0.95
            else:
                scale = 1.0
            return min_gap_x * scale, min_gap_y * scale, min_pair_gap * scale
        return 0.10, 0.085, 0.095

    def _candidate_respects_safe_spacing(self, candidate, positions):
        min_gap_x, min_gap_y, min_pair_gap = self._absolute_min_safe_gaps()
        for existing in positions:
            dx = abs(float(candidate[0] - existing[0]))
            dy = abs(float(candidate[1] - existing[1]))
            if dx < min_gap_x and dy < min_gap_y:
                return False
            if float(np.linalg.norm(candidate[:2] - existing[:2])) < min_pair_gap:
                return False
        return True

    def _sample_scene_asset(self):
        if self.fixed_modelname == "primitive_block":
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

        variant_ids = self._category_variant_ids()
        model_id = int(variant_ids[int(self.ep_num) % len(variant_ids)])
        return {
            "modelname": self.fixed_modelname,
            "model_id": model_id,
            "asset_key": f"{self.fixed_modelname}/base{model_id}",
            "asset_alias": self.fixed_asset_alias or self.fixed_modelname,
        }

    def _create_asset_object(self, pose, asset_spec):
        if asset_spec.get("primitive_kind") == "box":
            obj = create_box(
                scene=self.scene,
                pose=pose,
                half_size=asset_spec["half_size"],
                color=asset_spec["color"],
                name=self.fixed_asset_alias or 'block',
            )
            obj.set_mass(0.05)
            self.add_prohibit_area(obj, padding=0.05)
            return obj
        return PickAbsolutePositionMultiviewBase._create_asset_object(self, pose, asset_spec)

    def _desired_position_key(self):
        return POSITION_CYCLE[self.ep_num % len(POSITION_CYCLE)]

    def _projected_layout_score(self, projected):
        points = np.asarray([point[:2] for point in projected["points"]], dtype=float)
        width = float(projected["width"])
        height = float(projected["height"])
        margin_x = width * HEAD_CAMERA_MARGIN_X_RATIO
        margin_y = height * HEAD_CAMERA_MARGIN_Y_RATIO
        usable_width = max(1.0, width - 2 * margin_x)
        usable_height = max(1.0, height - 2 * margin_y)

        norm_points = np.zeros_like(points)
        norm_points[:, 0] = (points[:, 0] - margin_x) / usable_width
        norm_points[:, 1] = (points[:, 1] - margin_y) / usable_height
        norm_points = np.clip(norm_points, 0.0, 1.0)

        x_span = float(norm_points[:, 0].max() - norm_points[:, 0].min())
        y_span = float(norm_points[:, 1].max() - norm_points[:, 1].min())

        min_pair_dist = 0.0
        if len(points) > 1:
            min_pair_dist = min(
                float(np.linalg.norm(points[i] - points[j])) / max(1.0, min(usable_width, usable_height))
                for i in range(len(points))
                for j in range(i + 1, len(points))
            )

        cells = set()
        cols = set()
        rows = set()
        for x, y in norm_points:
            col = min(2, max(0, int(x * 3.0)))
            row = min(2, max(0, int(y * 3.0)))
            cells.add((col, row))
            cols.add(col)
            rows.add(row)

        cell_score = float(len(cells)) / float(max(1, len(points)))
        col_score = float(len(cols)) / 3.0
        row_score = float(len(rows)) / 3.0
        mean_offset = float(np.linalg.norm(np.mean(norm_points, axis=0) - np.array([0.5, 0.5], dtype=float)))
        center_score = 1.0 - min(1.0, mean_offset / 0.75)

        return (
            0.9 * min_pair_dist
            + 1.9 * x_span
            + 1.9 * y_span
            + 0.9 * cell_score
            + 0.6 * col_score
            + 0.6 * row_score
            + 0.2 * center_score
        )

    def _rendered_object_layout(self, objects):
        head_camera_info = self._head_camera_projection_config()
        if head_camera_info is None:
            return None

        for _ in range(HEAD_CAMERA_SETTLE_STEPS):
            self.scene.step()
        self._update_render()
        self.cameras.update_picture()
        head_camera = head_camera_info["camera"]
        segmentation = np.asarray(head_camera.get_picture("Segmentation")[..., 1], dtype=np.int32)

        rendered = []
        for obj in objects:
            actor_id = int(obj.actor.get_per_scene_id())
            mask = segmentation == actor_id
            pixel_count = int(mask.sum())
            if pixel_count < HEAD_CAMERA_MIN_OBJECT_PIXELS:
                return None

            pixel_y, pixel_x = np.nonzero(mask)
            rendered.append(
                {
                    "bbox": (
                        int(np.min(pixel_x)),
                        int(np.max(pixel_x)),
                        int(np.min(pixel_y)),
                        int(np.max(pixel_y)),
                    ),
                    "centroid": np.array(
                        [float(np.mean(pixel_x)), float(np.mean(pixel_y))],
                        dtype=float,
                    ),
                    "pixels": pixel_count,
                }
            )

        return {
            "objects": rendered,
            "width": int(head_camera_info["width"]),
            "height": int(head_camera_info["height"]),
        }

    def _is_visually_clear_rendered_layout(self, rendered):
        if rendered is None:
            return False

        width = float(rendered["width"])
        height = float(rendered["height"])
        margin_x = width * HEAD_CAMERA_MARGIN_X_RATIO
        margin_y = height * HEAD_CAMERA_MARGIN_Y_RATIO
        objects = rendered["objects"]

        for obj in objects:
            x0, x1, y0, y1 = obj["bbox"]
            if x0 < margin_x or x1 > width - margin_x:
                return False
            if y0 < margin_y or y1 > height - margin_y:
                return False

        min_centroid_dist = {2: 18.0, 3: 16.0, 4: 14.0, 5: 12.0, 6: 10.0, 7: 8.0}[self.object_count]
        if self.fixed_modelname == "primitive_block":
            min_centroid_dist = {2: 16.0, 3: 14.0, 4: 12.0, 5: 10.0, 6: 8.0, 7: 6.0}[self.object_count]

        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                centroid_dist = float(np.linalg.norm(objects[i]["centroid"] - objects[j]["centroid"]))
                if centroid_dist < min_centroid_dist:
                    return False
                if self._bbox_overlap_ratio(objects[i]["bbox"], objects[j]["bbox"]) > 0.0:
                    return False
        return True

    def _sample_positions(self, position_key):
        if self.object_count not in {2, 3, 4, 5, 6, 7}:
            raise ValueError("object_count must be one of {2, 3, 4, 5, 6, 7}")

        is_primitive_block = self.fixed_modelname == "primitive_block"
        x_min_bound, x_max_bound = -0.240, 0.240
        y_min_bound, y_max_bound = -0.225, 0.095

        asset_spec = self._sample_scene_asset()
        block_half_z = None
        if is_primitive_block:
            block_half_z = float(asset_spec["half_size"][2])
            asset_radius = float(np.linalg.norm(np.asarray(asset_spec["half_size"][:2], dtype=float)))
        else:
            asset_radius = FIXED_CATEGORY_WORLD_RADIUS[self.fixed_modelname]
        min_world_dist = 2.0 * asset_radius + FIXED_CATEGORY_WORLD_CLEARANCE[self.object_count]

        valid_candidates = []
        if self.fixed_modelname == "079_remotecontrol" and self.object_count >= 6:
            base_positions = [
                np.array([-0.215, -0.160, TABLE_Z], dtype=float),
                np.array([0.000, -0.160, TABLE_Z], dtype=float),
                np.array([0.215, -0.160, TABLE_Z], dtype=float),
                np.array([-0.215, 0.060, TABLE_Z], dtype=float),
                np.array([0.000, 0.060, TABLE_Z], dtype=float),
                np.array([0.215, 0.060, TABLE_Z], dtype=float),
            ]
            for _ in range(64):
                offset_x = float(np.random.uniform(-0.012, 0.012))
                offset_y = float(np.random.uniform(-0.012, 0.012))
                positions = []
                for pos in base_positions[: self.object_count]:
                    positions.append(np.array([pos[0] + offset_x, pos[1] + offset_y, pos[2]], dtype=float))
                if not self._has_unique_extreme(positions, position_key):
                    continue
                projected = self._project_positions_to_head_camera(positions)
                if not self._positions_visible_in_head_camera(projected):
                    continue
                poses = []
                for pos in positions:
                    poses.append(
                        rand_pose(
                            xlim=[float(pos[0]), float(pos[0])],
                            ylim=[float(pos[1]), float(pos[1])],
                            zlim=[float(pos[2]), float(pos[2])],
                            qpos=ASSET_QPOS,
                            rotate_rand=False,
                        )
                    )
                return {"positions": positions, "poses": poses, "score": None}
        max_trials = 1500 if is_primitive_block else 1000
        for _ in range(max_trials):
            positions = []
            attempts = 0
            while len(positions) < self.object_count and attempts < 240:
                attempts += 1
                pos_x = float(np.random.uniform(x_min_bound, x_max_bound))
                pos_y = float(np.random.uniform(y_min_bound, y_max_bound))
                pos_z = TABLE_Z + block_half_z if is_primitive_block else TABLE_Z
                candidate = np.array([pos_x, pos_y, pos_z], dtype=float)
                if any(np.linalg.norm(candidate[:2] - existing[:2]) < min_world_dist for existing in positions):
                    continue
                if not self._candidate_respects_safe_spacing(candidate, positions):
                    continue
                positions.append(candidate)
            if len(positions) != self.object_count:
                continue
            if not self._has_unique_extreme(positions, position_key):
                continue

            projected = self._project_positions_to_head_camera(positions)
            if not self._positions_visible_in_head_camera(projected):
                continue

            poses = []
            for pos in positions:
                if is_primitive_block:
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
                    rotate_lim = [0, float(np.pi), 0]
                    if self.fixed_modelname == "079_remotecontrol" and self.object_count >= 6:
                        rotate_lim = [0, float(np.pi) * 0.35, 0]
                    poses.append(
                        rand_pose(
                            xlim=[float(pos[0]), float(pos[0])],
                            ylim=[float(pos[1]), float(pos[1])],
                            zlim=[float(pos[2]), float(pos[2])],
                            qpos=ASSET_QPOS,
                            rotate_rand=True,
                            rotate_lim=rotate_lim,
                        )
                    )
            valid_candidates.append((positions, poses))
            if len(valid_candidates) >= 48:
                break

        if not valid_candidates:
            raise RuntimeError(
                f"Failed to sample a clear {self.object_count}-object absolute layout"
            )

        chosen_positions, chosen_poses = valid_candidates[int(np.random.randint(len(valid_candidates)))]
        return {"positions": chosen_positions, "poses": chosen_poses, "score": None}

    def play_once(self):
        info = super().play_once()
        info["layout"]["scene_asset_alias"] = self._category_noun()
        return info
