from ._base_task import Base_Task
from .utils import *

import sapien
import numpy as np
from copy import deepcopy


class Pick_Pill_Bottles_Distance_4(Base_Task):
    """
    Pillbottle距离推理pick任务
    设计原则：
    1. 保持此前mug距离任务的整体架构不变
    2. 仅修改上桌mug的选择逻辑：
       - 先随机选择一个参照颜色
       - 该颜色只放置1个对应base
       - 桌面上其余5个pillbottle从其他颜色对应的base里随机挑选
    3. 参照pillbottle固定为这个唯一颜色的pillbottle，再在其余pillbottle中随机选择“第x远”的pillbottle作为目标
    4. info仍然保持3个：参照颜色、距离序数、arm
    """

    OBJECT_NAME = "080_pillbottle"

    BASE_ID_TO_COLOR = {
        1: "orange pillbottle",
        2: "white pillbottle",
        3: "orange pillbottle",
        4: "brown pillbottle",
        5: "green pillbottle",
    }

    N_OBJ = 6

    # 参考官方pillbottle任务的姿态风格
    OBJ_QPOS = [0.707, 0.707, 0, 0]
    ROTATE_RAND = True
    ROTATE_LIM = [0, 1.57, 0]
    CONVEX = True

    PRE_GRASP_DIS = 0.08
    GRASP_DIS = 0.01
    LIFT_Z = 0.08

    # 保持pillbottle任务里的取值范围不变
    POSE_XLIM = [-0.22, 0.22]
    POSE_YLIM = [-0.18, 0.08]
    POSE_Z = 0.741

    # 按mug数量做轻微稳健调整
    MIN_DIST_SQ = 0.027
    CLEAR_DIST_GAP = 0.065
    PROHIBIT_PADDING = 0.07

    def setup_demo(self, **kwargs):
        super()._init_task_env_(**kwargs)

    def _ordinal_number(self, n):
        if 10 <= n % 100 <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suffix}"

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

        # if abs(candidate_pose.p[0]) < 0.05:
        #     return False

        # if np.sum((candidate_pose.p[:2] - np.array([0, -0.1])) ** 2) < 0.01:
        #     return False

        return True

    def _select_objects_for_episode(self):
        all_color_texts = sorted(set(self.BASE_ID_TO_COLOR.values()))

        valid_reference_colors = []
        for color_text in all_color_texts:
            other_base_ids = [
                base_id
                for base_id, base_color in self.BASE_ID_TO_COLOR.items()
                if base_color != color_text
            ]
            # 只要还有“其他颜色”的base可选，就允许作为参照颜色
            if len(other_base_ids) > 0:
                valid_reference_colors.append(color_text)

        assert len(valid_reference_colors) > 0, f"没有可用于当前{self.OBJECT_NAME}数量的参照颜色"

        reference_color_text = np.random.choice(valid_reference_colors)

        reference_base_candidates = [
            base_id
            for base_id, base_color in self.BASE_ID_TO_COLOR.items()
            if base_color == reference_color_text
        ]
        reference_base_id = int(np.random.choice(reference_base_candidates))

        other_base_ids = [
            base_id
            for base_id, base_color in self.BASE_ID_TO_COLOR.items()
            if base_color != reference_color_text
        ]

        need_num = self.N_OBJ - 1
        use_replace = len(other_base_ids) < need_num

        chosen_other_base_ids = list(
            np.random.choice(other_base_ids, size=need_num, replace=use_replace)
        )

        selected_pairs = [(reference_base_id, reference_color_text)]
        for base_id in chosen_other_base_ids:
            base_id = int(base_id)
            selected_pairs.append((base_id, self.BASE_ID_TO_COLOR[base_id]))

        shuffle_order = list(np.random.permutation(len(selected_pairs)))
        selected_pairs = [selected_pairs[i] for i in shuffle_order]
        reference_idx = shuffle_order.index(0)

        return selected_pairs, reference_idx

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
        selected_pairs, reference_idx = self._select_objects_for_episode()
        obj_pose_lst, valid_rank_infos = self._sample_object_poses(self.N_OBJ, reference_idx)

        self.BASE_IDS = [base_id for base_id, _ in selected_pairs]
        self.COLOR_TEXTS = [color_text for _, color_text in selected_pairs]

        self.objs = []
        for i in range(self.N_OBJ):
            pose = obj_pose_lst[i]
            obj = create_actor(
                scene=self,
                pose=pose,
                modelname=self.OBJECT_NAME,
                convex=self.CONVEX,
                model_id=self.BASE_IDS[i],
            )
            self.objs.append(obj)

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

        self.reference_text = self.COLOR_TEXTS[self.reference_idx]
        self.target_text = self.COLOR_TEXTS[self.target_idx]

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
