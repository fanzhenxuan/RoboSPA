import glob
import os

import numpy as np

from ._base_task import Base_Task
from .stable_simple_pick_assets import (
    build_interleaved_stable_asset_catalog,
    stable_asset_catalog_index,
)
from .utils import *

ASSET_QPOS = [0.5, 0.5, 0.5, 0.5]
TABLE_Z = 0.741
POSITION_CYCLE = ["leftmost", "rightmost", "frontmost", "backmost"]
HEAD_CAMERA_MARGIN_X_RATIO = 0.05
HEAD_CAMERA_MARGIN_Y_RATIO = 0.05
HEAD_CAMERA_MIN_OBJECT_PIXELS = 120
HEAD_CAMERA_SETTLE_STEPS = 180
HEAD_CAMERA_MIN_CENTROID_DIST = 60.0
HEAD_CAMERA_MAX_BBOX_OVERLAP_RATIO = 0.35
ABSOLUTE_POSITION_MIN_EXTREME_GAP = 0.06
ABSOLUTE_POSITION_VIEW_NAMES = ("front", "back", "left", "right")
ABSOLUTE_POSITION_VIEW_CAMERA = {
    "back": {
        "position": [-0.032, -0.45, 1.35],
        "forward": [0.0, 0.6, -0.8],
        "left": [-1.0, 0.0, 0.0],
    },
    "front": {
        "position": [-0.032, 0.45, 1.35],
        "forward": [0.0, -0.6, -0.8],
        "left": [1.0, 0.0, 0.0],
    },
    "left": {
        "position": [-0.45, -0.032, 1.35],
        "forward": [0.6, 0.0, -0.8],
        "left": [0.0, 1.0, 0.0],
    },
    "right": {
        "position": [0.45, -0.032, 1.35],
        "forward": [-0.6, 0.0, -0.8],
        "left": [0.0, -1.0, 0.0],
    },
}
ABSOLUTE_POSITION_VIEW_TEXT = {
    "front": "opposite",
    "back": "robot's",
    "left": "left",
    "right": "right",
}
ABSOLUTE_POSITION_RENDER_CAMERA_VIEW = "back"
ABSOLUTE_POSITION_VIEW_POSITION_KEY_MAP = {
    "back": {
        "leftmost": "leftmost",
        "rightmost": "rightmost",
        "frontmost": "frontmost",
        "backmost": "backmost",
    },
    "front": {
        "leftmost": "rightmost",
        "rightmost": "leftmost",
        "frontmost": "backmost",
        "backmost": "frontmost",
    },
    "left": {
        "leftmost": "backmost",
        "rightmost": "frontmost",
        "frontmost": "leftmost",
        "backmost": "rightmost",
    },
    "right": {
        "leftmost": "frontmost",
        "rightmost": "backmost",
        "frontmost": "rightmost",
        "backmost": "leftmost",
    },
}


class PickAbsolutePositionMultiviewBase(Base_Task):
    object_count = None
    asset_catalog_cache = None

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def _select_observation_view(self):
        if isinstance(self.scene_config, dict):
            scene_view_name = self.scene_config.get("view_name")
            if scene_view_name in ABSOLUTE_POSITION_VIEW_NAMES:
                return scene_view_name

        # Shuffle the four viewpoints within each block of four episodes so the
        # final dataset is as balanced as possible while retaining randomness.
        task_name = self.task_name or ""
        task_seed = sum((idx + 1) * ord(ch) for idx, ch in enumerate(task_name))
        block_idx = int(self.ep_num) // len(ABSOLUTE_POSITION_VIEW_NAMES)
        within_block_idx = int(self.ep_num) % len(ABSOLUTE_POSITION_VIEW_NAMES)
        block_seed = int(task_seed + 7919 * block_idx + 17 * int(self.object_count or 0))
        view_rng = np.random.RandomState(block_seed % (2**31 - 1))
        block_views = list(ABSOLUTE_POSITION_VIEW_NAMES)
        view_rng.shuffle(block_views)
        return block_views[within_block_idx]

    def _apply_observation_view(self, view_name):
        head_camera_id = getattr(self.cameras, "head_camera_id", None)
        if head_camera_id is None:
            self.view_name = None
            return

        if view_name not in ABSOLUTE_POSITION_VIEW_CAMERA:
            raise ValueError(f"Unsupported absolute-position view: {view_name}")

        # Keep rendered videos in the robot's ego-centric view while using
        # `view_name` only for instruction-side viewpoint remapping.
        camera_view_name = ABSOLUTE_POSITION_RENDER_CAMERA_VIEW
        view_cfg = ABSOLUTE_POSITION_VIEW_CAMERA[camera_view_name]
        cam_pos = np.asarray(view_cfg["position"], dtype=float)
        cam_forward = np.asarray(view_cfg["forward"], dtype=float)
        cam_forward /= np.linalg.norm(cam_forward)
        cam_left = np.asarray(view_cfg["left"], dtype=float)
        cam_left /= np.linalg.norm(cam_left)
        cam_up = np.cross(cam_forward, cam_left)

        mat44 = np.eye(4, dtype=float)
        mat44[:3, :3] = np.stack([cam_forward, cam_left, cam_up], axis=1)
        mat44[:3, 3] = cam_pos

        self.cameras.static_camera_list[head_camera_id].entity.set_pose(sapien.Pose(mat44))
        self.view_name = view_name
        self.camera_view_name = camera_view_name

    def _serialize_scene_config(self):
        if self.view_name is None:
            return None
        return {"view_name": self.view_name}

    def save_traj_data(self, idx):
        file_path = os.path.join(self.save_dir, "_traj_data", f"episode{idx}.pkl")
        traj_data = {
            "left_joint_path": deepcopy(self.left_joint_path),
            "right_joint_path": deepcopy(self.right_joint_path),
            "scene_config": self._serialize_scene_config(),
        }
        save_pkl(file_path, traj_data)

    def _get_available_model_ids(self, modelname):
        asset_path = os.path.join("assets/objects", modelname)
        json_files = sorted(glob.glob(os.path.join(asset_path, "model_data*.json")))

        model_ids = []
        for file_path in json_files:
            base_name = os.path.basename(file_path)
            if base_name == "model_data.json":
                model_ids.append(None)
                continue
            try:
                model_ids.append(int(base_name.replace("model_data", "").replace(".json", "")))
            except ValueError:
                continue
        return model_ids

    def _build_asset_catalog(self):
        cache = self.__class__.asset_catalog_cache
        if cache is not None:
            return cache

        catalog = build_interleaved_stable_asset_catalog(self._get_available_model_ids)
        if self.object_count >= 5:
            allowed_modelnames = {
                '047_mouse', '048_stapler', '071_can', '073_rubikscube', '075_bread', '077_phone', '079_remotecontrol', '080_pillbottle', '081_playingcards', '107_soap', '112_tea-box', '113_coffee-box'
            }
            catalog = [asset_spec for asset_spec in catalog if asset_spec['modelname'] in allowed_modelnames]

        self.__class__.asset_catalog_cache = catalog
        return catalog

    def _sample_scene_asset(self):
        catalog = self._build_asset_catalog()
        if len(catalog) >= 50:
            asset_idx = stable_asset_catalog_index(self.task_name, self.ep_num, len(catalog))
        else:
            asset_idx = int(self.ep_num % len(catalog))
        return dict(catalog[asset_idx])

    def _head_camera_projection_config(self):
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

    def _project_positions_to_head_camera(self, positions):
        camera_config = self._head_camera_projection_config()
        if camera_config is None:
            return None

        intrinsic = camera_config["intrinsic"]
        extrinsic = camera_config["extrinsic"]
        projected = []

        for position in positions:
            world_point = np.concatenate([np.asarray(position, dtype=float), [1.0]])
            camera_point = extrinsic @ world_point
            depth = float(camera_point[2])
            if depth <= 1e-6:
                return None

            pixel = intrinsic @ camera_point[:3]
            projected.append(
                np.array(
                    [float(pixel[0] / depth), float(pixel[1] / depth), depth],
                    dtype=float,
                )
            )

        return {
            "points": projected,
            "width": camera_config["width"],
            "height": camera_config["height"],
        }

    def _positions_visible_in_head_camera(self, projected):
        if projected is None:
            return False

        width = projected["width"]
        height = projected["height"]
        margin_x = width * HEAD_CAMERA_MARGIN_X_RATIO
        margin_y = height * HEAD_CAMERA_MARGIN_Y_RATIO

        for u, v, _ in projected["points"]:
            if not (margin_x <= u <= width - margin_x):
                return False
            if not (margin_y <= v <= height - margin_y):
                return False
        return True

    def _image_point_to_table_position(self, u, v):
        camera_config = self._head_camera_projection_config()
        if camera_config is None:
            return None

        intrinsic_inv = np.linalg.inv(camera_config["intrinsic"])
        extrinsic = np.asarray(camera_config["extrinsic"], dtype=float)
        if extrinsic.shape != (3, 4):
            return None

        ray_camera = np.asarray(
            intrinsic_inv @ np.array([float(u), float(v), 1.0], dtype=float),
            dtype=float,
        ).reshape(-1)
        if ray_camera.shape[0] != 3:
            return None

        rotation = extrinsic[:, :3]
        translation = extrinsic[:, 3]
        system = np.column_stack((rotation[:, 0], rotation[:, 1], -ray_camera))
        rhs = -(rotation[:, 2] * TABLE_Z + translation)
        try:
            solution = np.linalg.solve(system, rhs)
        except np.linalg.LinAlgError:
            return None

        world_x = float(solution[0])
        world_y = float(solution[1])
        ray_scale = float(solution[2])
        if ray_scale <= 0:
            return None
        return np.array([world_x, world_y, TABLE_Z], dtype=float)

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

        min_pair_dist = 1.0
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
            3.6 * min_pair_dist
            + 1.2 * x_span
            + 1.2 * y_span
            + 0.8 * cell_score
            + 0.5 * col_score
            + 0.5 * row_score
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

    def _bbox_overlap_ratio(self, bbox_a, bbox_b):
        ax0, ax1, ay0, ay1 = bbox_a
        bx0, bx1, by0, by1 = bbox_b
        inter_x0 = max(ax0, bx0)
        inter_x1 = min(ax1, bx1)
        inter_y0 = max(ay0, by0)
        inter_y1 = min(ay1, by1)
        if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
            return 0.0

        inter = float((inter_x1 - inter_x0) * (inter_y1 - inter_y0))
        area_a = float(max(1, (ax1 - ax0) * (ay1 - ay0)))
        area_b = float(max(1, (bx1 - bx0) * (by1 - by0)))
        return inter / min(area_a, area_b)

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

        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                centroid_dist = float(np.linalg.norm(objects[i]["centroid"] - objects[j]["centroid"]))
                if centroid_dist < HEAD_CAMERA_MIN_CENTROID_DIST:
                    return False
                if self._bbox_overlap_ratio(objects[i]["bbox"], objects[j]["bbox"]) > HEAD_CAMERA_MAX_BBOX_OVERLAP_RATIO:
                    return False
        return True

    def _remove_candidate_objects(self, objects, prohibit_area_length):
        for obj in objects:
            obj.actor.remove_from_scene()
        del self.prohibited_area[prohibit_area_length:]
        self.scene.step()
        self.scene.update_render()

    def _create_asset_object(self, pose, asset_spec):
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

    def _desired_position_key(self):
        return POSITION_CYCLE[self.ep_num % len(POSITION_CYCLE)]

    def _target_index(self, positions, position_key):
        xs = [float(pos[0]) for pos in positions]
        ys = [float(pos[1]) for pos in positions]

        if position_key == "leftmost":
            return int(np.argmin(xs))
        if position_key == "rightmost":
            return int(np.argmax(xs))
        if position_key == "frontmost":
            return int(np.argmax(ys))
        if position_key == "backmost":
            return int(np.argmin(ys))
        raise ValueError(f"Unsupported position key: {position_key}")

    def _has_unique_extreme(self, positions, position_key):
        xs = np.array([float(pos[0]) for pos in positions], dtype=float)
        ys = np.array([float(pos[1]) for pos in positions], dtype=float)

        if position_key == "leftmost":
            sorted_values = np.sort(xs)
            return (sorted_values[1] - sorted_values[0]) > ABSOLUTE_POSITION_MIN_EXTREME_GAP
        if position_key == "rightmost":
            sorted_values = np.sort(xs)
            return (sorted_values[-1] - sorted_values[-2]) > ABSOLUTE_POSITION_MIN_EXTREME_GAP
        if position_key == "frontmost":
            sorted_values = np.sort(ys)
            return (sorted_values[-1] - sorted_values[-2]) > ABSOLUTE_POSITION_MIN_EXTREME_GAP
        if position_key == "backmost":
            sorted_values = np.sort(ys)
            return (sorted_values[1] - sorted_values[0]) > ABSOLUTE_POSITION_MIN_EXTREME_GAP
        return False

    def _sample_positions(self, position_key):
        if self.object_count not in {2, 3, 4, 5}:
            raise ValueError("object_count must be one of {2, 3, 4, 5}")

        x_min_bound, x_max_bound = -0.215, 0.215
        y_min_bound, y_max_bound = -0.215, 0.070
        grid_points = [(-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
        min_cols = {2: 2, 3: 2, 4: 3, 5: 3}[self.object_count]
        min_rows = {2: 1, 3: 2, 4: 2, 5: 3}[self.object_count]
        min_world_distance_candidates = {
            2: [0.145, 0.135, 0.125],
            3: [0.135, 0.125, 0.115],
            4: [0.125, 0.115, 0.105],
            5: [0.118, 0.108, 0.098],
        }[self.object_count]
        min_x_span = {2: 0.34, 3: 0.40, 4: 0.48, 5: 0.56}[self.object_count]
        min_y_span = {2: 0.12, 3: 0.24, 4: 0.32, 5: 0.40}[self.object_count]

        chosen_positions = None
        chosen_score = None

        for min_world_dist in min_world_distance_candidates:
            valid_candidates = []
            for _ in range(420):
                chosen_idx = np.random.choice(len(grid_points), size=self.object_count, replace=False)
                chosen_grids = [grid_points[int(idx)] for idx in chosen_idx]
                used_cols = {grid_x for grid_x, _ in chosen_grids}
                used_rows = {grid_y for _, grid_y in chosen_grids}
                if len(used_cols) < min_cols or len(used_rows) < min_rows:
                    continue

                grid_gap_x = float(np.random.uniform(0.135, 0.185))
                grid_gap_y = float(np.random.uniform(0.110, 0.148))
                column_shift = {col: float(np.random.uniform(-0.022, 0.022)) for col in (-1, 0, 1)}
                row_shift = {row: float(np.random.uniform(-0.020, 0.020)) for row in (-1, 0, 1)}

                offsets = [
                    (
                        grid_x * grid_gap_x + column_shift[grid_x],
                        grid_y * grid_gap_y + row_shift[grid_y],
                    )
                    for grid_x, grid_y in chosen_grids
                ]
                offset_xs = [offset[0] for offset in offsets]
                offset_ys = [offset[1] for offset in offsets]
                anchor_x_min = x_min_bound - max(offset_xs)
                anchor_x_max = x_max_bound - min(offset_xs)
                anchor_y_min = y_min_bound - max(offset_ys)
                anchor_y_max = y_max_bound - min(offset_ys)
                if anchor_x_min >= anchor_x_max or anchor_y_min >= anchor_y_max:
                    continue

                anchor_x = float(np.random.uniform(anchor_x_min, anchor_x_max))
                anchor_y = float(np.random.uniform(anchor_y_min, anchor_y_max))
                positions = [
                    np.array([anchor_x + offset_x, anchor_y + offset_y, TABLE_Z], dtype=float)
                    for offset_x, offset_y in offsets
                ]

                if any(
                    np.linalg.norm(positions[i][:2] - positions[j][:2]) < min_world_dist
                    for i in range(len(positions))
                    for j in range(i + 1, len(positions))
                ):
                    continue
                if not self._has_unique_extreme(positions, position_key):
                    continue

                projected = self._project_positions_to_head_camera(positions)
                if not self._positions_visible_in_head_camera(projected):
                    continue

                width = float(projected["width"])
                height = float(projected["height"])
                margin_x = width * HEAD_CAMERA_MARGIN_X_RATIO
                margin_y = height * HEAD_CAMERA_MARGIN_Y_RATIO
                usable_width = max(1.0, width - 2 * margin_x)
                usable_height = max(1.0, height - 2 * margin_y)
                points = np.asarray([point[:2] for point in projected["points"]], dtype=float)
                norm_points = np.zeros_like(points)
                norm_points[:, 0] = (points[:, 0] - margin_x) / usable_width
                norm_points[:, 1] = (points[:, 1] - margin_y) / usable_height
                norm_points = np.clip(norm_points, 0.0, 0.999999)
                x_span = float(norm_points[:, 0].max() - norm_points[:, 0].min())
                y_span = float(norm_points[:, 1].max() - norm_points[:, 1].min())
                if x_span < min_x_span or y_span < min_y_span:
                    continue

                score = self._projected_layout_score(projected)
                valid_candidates.append((score, [pos.copy() for pos in positions]))

            if valid_candidates:
                valid_candidates.sort(key=lambda item: item[0], reverse=True)
                top_candidates = valid_candidates[: min(12, len(valid_candidates))]
                chosen_score, chosen_positions = top_candidates[int(np.random.randint(len(top_candidates)))]
                break

        if chosen_positions is None:
            raise RuntimeError(f"Failed to sample a valid {self.object_count}-object absolute layout")

        poses = []
        for pos in chosen_positions:
            poses.append(
                rand_pose(
                    xlim=[float(pos[0]), float(pos[0])],
                    ylim=[float(pos[1]), float(pos[1])],
                    zlim=[TABLE_Z, TABLE_Z],
                    qpos=ASSET_QPOS,
                    rotate_rand=True,
                    rotate_lim=[0, float(np.pi), 0],
                )
            )

        return {"positions": chosen_positions, "poses": poses, "score": chosen_score}

    def _relation_phrase(self, position_key):
        if self.object_count == 2:
            if position_key == "leftmost":
                return "the left object"
            if position_key == "rightmost":
                return "the right object"
            if position_key == "frontmost":
                return "the front object"
            if position_key == "backmost":
                return "the back object"
        if position_key == "leftmost":
            return "the leftmost object"
        if position_key == "rightmost":
            return "the rightmost object"
        if position_key == "frontmost":
            return "the frontmost object"
        if position_key == "backmost":
            return "the backmost object"
        raise ValueError(f"Unsupported position key: {position_key}")

    def _view_relative_position_key(self, position_key):
        view_name = self.view_name or "back"
        return ABSOLUTE_POSITION_VIEW_POSITION_KEY_MAP.get(view_name, {}).get(position_key, position_key)

    def load_actors(self):
        self._apply_observation_view(self._select_observation_view())
        self.position_key = self._desired_position_key()
        self.scene_asset = self._sample_scene_asset()

        for _ in range(120):
            sampled = self._sample_positions(self.position_key)
            prohibit_area_length = len(self.prohibited_area)
            candidate_objects = []
            for pose in sampled["poses"]:
                candidate_objects.append(self._create_asset_object(pose, self.scene_asset))

            rendered = self._rendered_object_layout(candidate_objects)
            if not self._is_visually_clear_rendered_layout(rendered):
                self._remove_candidate_objects(candidate_objects, prohibit_area_length)
                continue

            self.objects = candidate_objects
            self.object_positions = [obj.get_pose().p.copy() for obj in self.objects]
            self.target_index = self._target_index(self.object_positions, self.position_key)
            self.target_object = self.objects[self.target_index]

            target_x = float(self.target_object.get_pose().p[0])
            self.arm_tag = ArmTag("right" if target_x > 0 else "left")
            self.start_z = float(self.target_object.get_pose().p[2])
            self.observed_position_key = self._view_relative_position_key(self.position_key)
            self.position_phrase = self._relation_phrase(self.observed_position_key)
            return

        raise RuntimeError(
            f"Failed to sample a visually clear {self.object_count}-object absolute layout"
        )

    def play_once(self):
        arm_tag = self.arm_tag
        self.move(
            self.grasp_actor(
                self.target_object,
                arm_tag=arm_tag,
                pre_grasp_dis=self._play_once_pre_grasp_dis(),
                grasp_dis=self._play_once_grasp_dis(),
            )
        )
        self.move(
            self.move_by_displacement(
                arm_tag=arm_tag,
                z=self._play_once_lift_z(),
                move_axis="arm",
            )
        )

        self.info["layout"] = {
            "object_count": self.object_count,
            "target_index": self.target_index,
            "position_key": self.position_key,
            "observed_position_key": self.observed_position_key,
            "scene_asset": self.scene_asset["asset_key"],
            "view_name": self.view_name,
            "camera_view_name": getattr(self, "camera_view_name", ABSOLUTE_POSITION_RENDER_CAMERA_VIEW),
            "object_positions": [
                [float(pos[0]), float(pos[1]), float(pos[2])]
                for pos in self.object_positions
            ],
        }
        self.info["info"] = {
            "{P}": self.position_phrase,
            "{V}": ABSOLUTE_POSITION_VIEW_TEXT[self.view_name],
            "{a}": str(arm_tag),
        }
        return self.info

    def _play_once_pre_grasp_dis(self):
        return 0.09

    def _play_once_grasp_dis(self):
        return 0.01

    def _play_once_lift_z(self):
        return 0.12

    def _target_actor_name(self):
        if hasattr(self.target_object, "get_name"):
            return self.target_object.get_name()
        actor = getattr(self.target_object, "actor", None)
        if actor is not None and hasattr(actor, "get_name"):
            return actor.get_name()
        return None

    def check_success(self):
        target_pose = self.target_object.get_pose().p
        lifted = (float(target_pose[2]) - self.start_z) > 0.06
        target_name = self._target_actor_name()
        contact_positions = self.get_gripper_actor_contact_position(target_name) if target_name else []
        gripper_closed = self.is_left_gripper_close() if self.arm_tag == "left" else self.is_right_gripper_close()
        grasped = bool(gripper_closed and len(contact_positions) > 0)
        return bool(lifted and grasped)
