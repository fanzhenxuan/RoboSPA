from ._base_task import Base_Task
from .utils import *

import os
import glob
import sapien
import numpy as np
from copy import deepcopy


class Pick_Mixed_Objects_Distance_A_2(Base_Task):
    """
    Heterogeneous-object distance-reasoning pick task.
    Keep the original distance-task structure and only change object loading logic.
    """

    OBJECT_SPECS = [
        # {'modelname': '050_bell', 'skip_model_ids': []},
        {'modelname': '021_cup', 'skip_model_ids': [6,10,11,12]},
        {'modelname': '075_bread', 'skip_model_ids': []},
        # {'modelname': '113_coffee-box', 'skip_model_ids': []},
        {'modelname': '071_can', 'skip_model_ids': []},
        {'modelname': '048_stapler', 'skip_model_ids': []}]

    N_OBJ = 4

    OBJ_QPOS = [0.707, 0.707, 0, 0]
    ROTATE_RAND = True
    ROTATE_LIM = [0, 1.57, 0]
    CONVEX = True

    PRE_GRASP_DIS = 0.12
    GRASP_DIS = 0.01
    LIFT_Z = 0.08

    POSE_XLIM = [-0.22, 0.22]
    POSE_YLIM = [-0.18, 0.08]
    POSE_Z = 0.741

    MIN_DIST_SQ = 0.025
    # MIN_DIST_SQ = 0.0144
    CLEAR_DIST_GAP = 0.03
    # CLEAR_DIST_GAP = 0.058
    PROHIBIT_PADDING = 0.068

    def setup_demo(self, **kwargs):
        super()._init_task_env_(**kwargs)

    def _ordinal_number(self, n):
        if 10 <= n % 100 <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suffix}"

    def _object_phrase(self, modelname):
        return modelname.split("_", 1)[1].replace("-", " ")

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

    def _rand_one_pose(self):
        return rand_pose(
            xlim=self.POSE_XLIM,
            ylim=self.POSE_YLIM,
            zlim=[self.POSE_Z],
            qpos=self.OBJ_QPOS,
            ylim_prop=True,
            rotate_rand=self.ROTATE_RAND,
            rotate_lim=self.ROTATE_LIM,
        )

    def _check_pose_valid(self, candidate_pose, old_pose_lst):
        for old_pose in old_pose_lst:
            if np.sum((candidate_pose.p[:2] - old_pose.p[:2]) ** 2) < self.MIN_DIST_SQ:
                return False
        return True

    def _select_objects_for_episode(self):
        selected_specs = [dict(spec) for spec in self.OBJECT_SPECS[:self.N_OBJ]]
        for spec in selected_specs:
            spec["model_id"] = self._sample_model_id(spec)
            spec["phrase"] = self._object_phrase(spec["modelname"])

        shuffle_order = list(np.random.permutation(len(selected_specs)))
        selected_specs = [selected_specs[i] for i in shuffle_order]
        reference_idx = int(np.random.randint(len(selected_specs)))
        return selected_specs, reference_idx

    def _sample_object_poses(self, n_obj, reference_idx):
        while True:
            obj_pose_lst = []

            for _ in range(n_obj):
                obj_pose = self._rand_one_pose()

                retry = 0
                while not self._check_pose_valid(obj_pose, obj_pose_lst):
                    obj_pose = self._rand_one_pose()
                    retry += 1
                    if retry > 300:
                        break

                if retry > 300:
                    break

                obj_pose_lst.append(deepcopy(obj_pose))

            if len(obj_pose_lst) != n_obj:
                continue

            xs = [pose.p[0] for pose in obj_pose_lst]
            ys = [pose.p[1] for pose in obj_pose_lst]
            xs_sorted = sorted(xs)

            too_ordered = True
            for i in range(len(xs_sorted) - 1):
                if abs(xs_sorted[i + 1] - xs_sorted[i]) > 0.12:
                    too_ordered = False
                    break
            if max(ys) - min(ys) > 0.04:
                too_ordered = False

            if too_ordered:
                continue

            ref_xy = obj_pose_lst[reference_idx].p[:2]
            dist_info = []

            for obj_idx in range(n_obj):
                if obj_idx == reference_idx:
                    continue
                dist = np.linalg.norm(obj_pose_lst[obj_idx].p[:2] - ref_xy)
                dist_info.append((obj_idx, dist))

            dist_info.sort(key=lambda x: x[1], reverse=True)

            valid_rank_infos = []
            for rank_idx in range(len(dist_info)):
                enough_gap = True

                if rank_idx > 0:
                    if abs(dist_info[rank_idx][1] - dist_info[rank_idx - 1][1]) < self.CLEAR_DIST_GAP:
                        enough_gap = False

                if rank_idx < len(dist_info) - 1:
                    if abs(dist_info[rank_idx][1] - dist_info[rank_idx + 1][1]) < self.CLEAR_DIST_GAP:
                        enough_gap = False

                if enough_gap:
                    valid_rank_infos.append((rank_idx, dist_info))

            if len(valid_rank_infos) == 0:
                continue

            return obj_pose_lst, valid_rank_infos

    def load_actors(self):
        selected_specs, reference_idx = self._select_objects_for_episode()
        obj_pose_lst, valid_rank_infos = self._sample_object_poses(self.N_OBJ, reference_idx)

        self.object_specs = selected_specs
        self.objs = []
        self.OBJECT_TEXTS = []

        for i in range(self.N_OBJ):
            spec = self.object_specs[i]
            pose = obj_pose_lst[i]
            obj = create_actor(
                scene=self,
                pose=pose,
                modelname=spec["modelname"],
                convex=self.CONVEX,
                model_id=spec["model_id"],
            )
            self.objs.append(obj)
            self.OBJECT_TEXTS.append(spec["phrase"])

        self.delay(2)

        for obj in self.objs:
            self.add_prohibit_area(obj, padding=self.PROHIBIT_PADDING)
        self.prohibited_area.append([-0.17, -0.22, 0.17, -0.12])

        chosen_rank_idx, chosen_dist_info = valid_rank_infos[
            np.random.randint(len(valid_rank_infos))
        ]

        self.reference_idx = reference_idx
        self.distance_rank = self._ordinal_number(chosen_rank_idx + 1)
        self.target_idx = chosen_dist_info[chosen_rank_idx][0]
        self.target_obj = self.objs[self.target_idx]

        self.reference_text = self.OBJECT_TEXTS[self.reference_idx]
        self.target_text = self.OBJECT_TEXTS[self.target_idx]

    def play_once(self):
        self.last_gripper = None
        arm_tag = self.pick_target_object(self.target_obj)

        self.info["info"] = {
            "{A}": self.reference_text,
            "{B}": self.distance_rank,
            "{a}": str(arm_tag),
        }
        return self.info

    def pick_target_object(self, obj):
        obj_pose = obj.get_pose().p
        arm_tag = ArmTag("left" if obj_pose[0] < 0 else "right")

        if self.last_gripper is not None and self.last_gripper != arm_tag:
            self.move(
                self.grasp_actor(
                    obj,
                    arm_tag=arm_tag,
                    pre_grasp_dis=self.PRE_GRASP_DIS,
                    grasp_dis=self.GRASP_DIS,
                ),
                self.back_to_origin(arm_tag=arm_tag.opposite),
            )
        else:
            self.move(
                self.grasp_actor(
                    obj,
                    arm_tag=arm_tag,
                    pre_grasp_dis=self.PRE_GRASP_DIS,
                    grasp_dis=self.GRASP_DIS,
                )
            )

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=self.LIFT_Z))
        self.last_gripper = arm_tag
        return str(arm_tag)

    def check_success(self):
        obj_pose = self.target_obj.get_pose().p
        return obj_pose[2] > 0.82
