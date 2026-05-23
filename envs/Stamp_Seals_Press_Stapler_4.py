from ._base_task import Base_Task
from .utils import *
from ._GLOBAL_CONFIGS import *
import numpy as np


class Stamp_Seals_Press_Stapler_4(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        # independent progress counters
        # left_seal_stage: 0 -> 1 -> 2
        # right_seal_stage: 0 -> 1 -> 2
        # stapler_stage: 0 -> 1
        self.left_seal_stage = 0
        self.right_seal_stage = 0
        self.stapler_stage = 0

        # overall bookkeeping only for compatibility
        self.stage_sum = 4
        self.stage = 0
        self.task_success = [0] * self.stage_sum

        self.required_left_seal_stage = 2
        self.required_right_seal_stage = 2
        self.required_stapler_stage = 0

        # de-bounce flags
        self.has_left_left_stamp_area = True
        self.has_left_right_stamp_area = True
        self.has_left_press_area = True

        self._left_seal_arm_tag = None
        self._right_seal_arm_tag = None
        self._stapler_arm_tag = None

        # -----------------------------
        # 1) left seal position
        # -----------------------------
        left_seal_rand_pos = rand_pose(
            xlim=[-0.25, -0.12],
            ylim=[-0.15, 0.05],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=False,
        )

        max_trials = 100
        trials = 0

        while abs(left_seal_rand_pos.p[0]) < 0.05 and trials < max_trials:
            left_seal_rand_pos = rand_pose(
                xlim=[-0.25, -0.12],
                ylim=[-0.15, 0.05],
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=False,
            )
            trials += 1

        if abs(left_seal_rand_pos.p[0]) < 0.05:
            raise RuntimeError("Failed to sample a valid left_seal_rand_pos within 100 tries.")

        self.left_seal_id = np.random.choice([0, 2, 3, 4, 6], 1)[0]
        self.left_seal = create_actor(
            scene=self,
            pose=left_seal_rand_pos,
            modelname="100_seal",
            convex=True,
            model_id=self.left_seal_id,
        )
        self.left_seal.set_mass(0.05)

        # -----------------------------
        # 2) left pad
        # -----------------------------
        left_target_rand_pose = rand_pose(
            xlim=[-0.25, -0.12],
            ylim=[-0.15, 0.05],
            qpos=[1, 0, 0, 0],
            rotate_rand=False,
        )

        max_trials = 100
        trials = 0

        while (
            np.sqrt(
                (left_target_rand_pose.p[0] - left_seal_rand_pos.p[0]) ** 2
                + (left_target_rand_pose.p[1] - left_seal_rand_pos.p[1]) ** 2
            ) < 0.1
        ) and trials < max_trials:
            left_target_rand_pose = rand_pose(
                xlim=[-0.25, -0.12],
                ylim=[-0.15, 0.1],
                qpos=[1, 0, 0, 0],
                rotate_rand=False,
            )
            trials += 1

        if (
            np.sqrt(
                (left_target_rand_pose.p[0] - left_seal_rand_pos.p[0]) ** 2
                + (left_target_rand_pose.p[1] - left_seal_rand_pos.p[1]) ** 2
            ) < 0.1
        ):
            raise RuntimeError("Failed to sample a valid left_target_rand_pose within 100 tries.")

        colors = {
            "Red": (1, 0, 0),
            "Green": (0, 1, 0),
            "Blue": (0, 0, 1),
            "Yellow": (1, 1, 0),
            "Cyan": (0, 1, 1),
            "Magenta": (1, 0, 1),
            "Black": (0, 0, 0),
            "Gray": (0.5, 0.5, 0.5),
            "Orange": (1, 0.5, 0),
            "Purple": (0.5, 0, 0.5),
            "Brown": (0.65, 0.4, 0.16),
            "Pink": (1, 0.75, 0.8),
            "Lime": (0.5, 1, 0),
            "Olive": (0.5, 0.5, 0),
            "Teal": (0, 0.5, 0.5),
            "Maroon": (0.5, 0, 0),
            "Navy": (0, 0, 0.5),
            "Coral": (1, 0.5, 0.31),
            "Turquoise": (0.25, 0.88, 0.82),
            "Indigo": (0.29, 0, 0.51),
            "Beige": (0.96, 0.91, 0.81),
            "Tan": (0.82, 0.71, 0.55),
            "Silver": (0.75, 0.75, 0.75),
        }
        color_items = list(colors.items())
        left_idx = np.random.choice(len(color_items))
        self.left_pad_color_name, self.left_pad_color_value = color_items[left_idx]
        right_idx = np.random.choice(len(color_items))
        self.right_pad_color_name, self.right_pad_color_value = color_items[right_idx]

        self.left_target = create_visual_box(
            scene=self,
            pose=left_target_rand_pose,
            half_size=[0.035, 0.035, 0.0005],
            color=self.left_pad_color_value,
            name="box",
        )

        if self.required_right_seal_stage > 0:
            # -----------------------------
            # 3) right seal position
            # -----------------------------
            right_seal_rand_pos = rand_pose(
                xlim=[0.10, 0.22],
                ylim=[-0.15, 0.05],
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=False,
            )

            max_trials = 100
            trials = 0

            while abs(right_seal_rand_pos.p[0]) < 0.05 and trials < max_trials:
                right_seal_rand_pos = rand_pose(
                    xlim=[0.10, 0.22],
                    ylim=[-0.15, 0.05],
                    qpos=[0.5, 0.5, 0.5, 0.5],
                    rotate_rand=False,
                )
                trials += 1

            if abs(right_seal_rand_pos.p[0]) < 0.05:
                raise RuntimeError("Failed to sample a valid right_seal_rand_pos within 100 tries.")

            self.right_seal_id = np.random.choice([0, 2, 3, 4, 6], 1)[0]
            self.right_seal = create_actor(
                scene=self,
                pose=right_seal_rand_pos,
                modelname="100_seal",
                convex=True,
                model_id=self.right_seal_id,
            )
            self.right_seal.set_mass(0.05)

            # -----------------------------
            # 4) right pad
            # -----------------------------
            right_target_rand_pose = rand_pose(
                xlim=[0.10, 0.22],
                ylim=[-0.15, 0.05],
                qpos=[1, 0, 0, 0],
                rotate_rand=False,
            )

            max_trials = 100
            trials = 0

            while (
                np.sqrt(
                    (right_target_rand_pose.p[0] - right_seal_rand_pos.p[0]) ** 2
                    + (right_target_rand_pose.p[1] - right_seal_rand_pos.p[1]) ** 2
                ) < 0.1
            ) and trials < max_trials:
                right_target_rand_pose = rand_pose(
                    xlim=[0.10, 0.22],
                    ylim=[-0.15, 0.1],
                    qpos=[1, 0, 0, 0],
                    rotate_rand=False,
                )
                trials += 1

            if (
                np.sqrt(
                    (right_target_rand_pose.p[0] - right_seal_rand_pos.p[0]) ** 2
                    + (right_target_rand_pose.p[1] - right_seal_rand_pos.p[1]) ** 2
                ) < 0.1
            ):
                raise RuntimeError("Failed to sample a valid right_target_rand_pose within 100 tries.")

            self.right_target = create_visual_box(
                scene=self,
                pose=right_target_rand_pose,
                half_size=[0.035, 0.035, 0.0005],
                color=self.right_pad_color_value,
                name="box",
            )

        if self.required_stapler_stage > 0:
            # -----------------------------
            # 5) stapler
            # -----------------------------
            stapler_rand_pos = rand_pose(
                xlim=[-0.03, 0.04],
                ylim=[-0.15, 0.05],
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=True,
                rotate_lim=[0, np.pi, 0],
            )

            self.stapler_id = np.random.choice([0, 1, 2, 3, 4, 5, 6], 1)[0]
            self.stapler = create_actor(
                self,
                pose=stapler_rand_pos,
                modelname="048_stapler",
                convex=True,
                model_id=self.stapler_id,
                is_static=True,
            )

        self.add_prohibit_area(self.left_seal, padding=0.1)
        self.add_prohibit_area(self.left_target, padding=0.1)
        if self.required_right_seal_stage > 0:
            self.add_prohibit_area(self.right_seal, padding=0.1)
            self.add_prohibit_area(self.right_target, padding=0.1)
        if self.required_stapler_stage > 0:
            self.add_prohibit_area(self.stapler, padding=0.05)

    def _get_left_seal_arm_tag(self):
        if self._left_seal_arm_tag is None:
            self._left_seal_arm_tag = ArmTag("left" if self.left_seal.get_pose().p[0] < 0 else "right")
        return self._left_seal_arm_tag

    def _get_right_seal_arm_tag(self):
        if self._right_seal_arm_tag is None:
            self._right_seal_arm_tag = ArmTag("right" if self.right_seal.get_pose().p[0] > 0 else "left")
        return self._right_seal_arm_tag

    def _get_stapler_arm_tag(self):
        if self._stapler_arm_tag is None:
            self._stapler_arm_tag = ArmTag("left" if self.stapler.get_pose().p[0] < 0 else "right")
        return self._stapler_arm_tag

    def _seal_on_target(self, seal, target, eps=0.01):
        seal_pose = seal.get_pose().p
        target_pos = target.get_pose().p
        return (
            np.all(np.abs(seal_pose[:2] - target_pos[:2]) < np.array([eps, eps]))
            and abs(seal_pose[2] - target_pos[2]) < 0.01
        )

    def _seal_has_left_target(self, seal, target, eps=0.02):
        seal_pose = seal.get_pose().p
        target_pos = target.get_pose().p
        return (
            np.any(np.abs(seal_pose[:2] - target_pos[:2]) > np.array([eps, eps]))
            or abs(seal_pose[2] - target_pos[2]) > 0.01
        )

    def _is_press_success(self):
        stapler_pose = self.stapler.get_contact_point(2)[:3]
        positions = self.get_gripper_actor_contact_position("048_stapler")
        eps_xy = np.array([0.03, 0.03])

        for position in positions:
            if (
                np.all(np.abs(position[:2] - stapler_pose[:2]) < eps_xy)
                and abs(position[2] - stapler_pose[2]) < 0.03
            ):
                return True
        return False

    def _has_left_press_area(self):
        stapler_pose = self.stapler.get_contact_point(2)[:3]
        positions = self.get_gripper_actor_contact_position("048_stapler")

        if len(positions) == 0:
            return True

        for position in positions:
            if (
                np.all(np.abs(position[:2] - stapler_pose[:2]) < np.array([0.03, 0.03]))
                and abs(position[2] - stapler_pose[2]) < 0.03
            ):
                return False

        return True

    def update_progress(self):
        # -----------------------------
        # left seal progress
        # -----------------------------
        if self.required_left_seal_stage > 0:
            if self._seal_has_left_target(self.left_seal, self.left_target):
                self.has_left_left_stamp_area = True

            if (
                self.left_seal_stage < self.required_left_seal_stage
                and self.has_left_left_stamp_area
                and self._seal_on_target(self.left_seal, self.left_target)
            ):
                self.left_seal_stage += 1
                self.has_left_left_stamp_area = False

        # -----------------------------
        # right seal progress
        # -----------------------------
        if self.required_right_seal_stage > 0:
            if self._seal_has_left_target(self.right_seal, self.right_target):
                self.has_left_right_stamp_area = True

            if (
                self.right_seal_stage < self.required_right_seal_stage
                and self.has_left_right_stamp_area
                and self._seal_on_target(self.right_seal, self.right_target)
            ):
                self.right_seal_stage += 1
                self.has_left_right_stamp_area = False

        # -----------------------------
        # stapler progress
        # -----------------------------
        if self.required_stapler_stage > 0:
            if self._has_left_press_area():
                self.has_left_press_area = True

            if self.stapler_stage < self.required_stapler_stage and self.has_left_press_area and self._is_press_success():
                self.stapler_stage += 1
                self.has_left_press_area = False

        task_success = []
        for i in range(self.required_left_seal_stage):
            task_success.append(int(self.left_seal_stage >= i + 1))
        for i in range(self.required_right_seal_stage):
            task_success.append(int(self.right_seal_stage >= i + 1))
        for i in range(self.required_stapler_stage):
            task_success.append(int(self.stapler_stage >= i + 1))

        self.task_success = task_success
        self.stage = sum(self.task_success)

    def _do_one_seal_stage(self, seal, target, arm_tag, prev_arm_tag=None, lift_after_grasp=True, z_offset=0.05):
        grasp_action = self.grasp_actor(
            seal,
            arm_tag=arm_tag,
            pre_grasp_dis=0.1,
            contact_point_id=[4, 5, 6, 7],
        )
        retreat_z = 0.10

        if prev_arm_tag is not None and prev_arm_tag != arm_tag:
            # 先让旧手臂原地竖直上抬，避免低空扫到seal或pad
            self.move(
                self.move_by_displacement(arm_tag=prev_arm_tag, z=retreat_z)
            )
            # 再让新手臂去抓，同时旧手臂收回
            self.move(
                grasp_action,
                self.back_to_origin(arm_tag=prev_arm_tag),
            )
        else:
            self.move(grasp_action)

        # if prev_arm_tag is not None and prev_arm_tag != arm_tag:
        #     self.move(
        #         grasp_action,
        #         self.back_to_origin(arm_tag=prev_arm_tag),
        #     )
        # else:
        #     self.move(grasp_action)

        if lift_after_grasp:
            self.move(self.move_by_displacement(arm_tag=arm_tag, z=z_offset))

        self.update_progress()
        self.move(
            self.place_actor(
                seal,
                arm_tag=arm_tag,
                target_pose=target.get_pose(),
                pre_dis=0.1,
                constrain="auto",
            )
        )
        self.move(self.open_gripper(arm_tag=arm_tag))
        self.update_progress()

    def _do_stapler_press_once(self, stapler_arm_tag, prev_arm_tag=None):
        stapler_hover_action = self.grasp_actor(
            self.stapler,
            arm_tag=stapler_arm_tag,
            pre_grasp_dis=0.1,
            grasp_dis=0.1,
            contact_point_id=2,
        )

        stapler_press_action = self.grasp_actor(
            self.stapler,
            arm_tag=stapler_arm_tag,
            pre_grasp_dis=0.02,
            grasp_dis=0.02,
            contact_point_id=2,
        )

        if prev_arm_tag is not None and prev_arm_tag != stapler_arm_tag:
            self.move(
                stapler_hover_action,
                self.back_to_origin(arm_tag=prev_arm_tag),
            )
        else:
            self.move(stapler_hover_action)
        self.update_progress()

        self.move(self.close_gripper(arm_tag=stapler_arm_tag))
        self.move(stapler_press_action)
        self.update_progress()

    def _move_arm_up_and_back(self, arm_tag):
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.10))
        # self.move(self.back_to_origin(arm_tag=arm_tag))

    def play_once(self):
        left_seal_arm_tag = self._get_left_seal_arm_tag()
        prev_arm_tag = None

        for _ in range(self.required_left_seal_stage):
            self._do_one_seal_stage(
                self.left_seal,
                self.left_target,
                left_seal_arm_tag,
                prev_arm_tag=prev_arm_tag,
                lift_after_grasp=True,
            )
            prev_arm_tag = left_seal_arm_tag

        if self.required_right_seal_stage > 0:
            right_seal_arm_tag = self._get_right_seal_arm_tag()
            for i in range(self.required_right_seal_stage):
                self._do_one_seal_stage(
                    self.right_seal,
                    self.right_target,
                    right_seal_arm_tag,
                    prev_arm_tag=prev_arm_tag if i == 0 else None,
                    lift_after_grasp=True,
                )
                prev_arm_tag = right_seal_arm_tag

        if self.required_stapler_stage > 0:
            stapler_arm_tag = self._get_stapler_arm_tag()
            self._do_stapler_press_once(
                stapler_arm_tag,
                prev_arm_tag=prev_arm_tag,
            )
            prev_arm_tag = stapler_arm_tag
        elif self.required_right_seal_stage > 0:
            self._move_arm_up_and_back(prev_arm_tag)
        elif self.required_left_seal_stage > 0:
            self._move_arm_up_and_back(prev_arm_tag)

        self.info["info"] = {
            # "{A}": f"100_seal/base{self.left_seal_id}",
            # "{B}": f"100_seal/base{self.right_seal_id}",
            "{a}": str(left_seal_arm_tag),
            "{b}": str(right_seal_arm_tag),
        }
        return self.info

    def check_success(self):
        self.update_progress()
        return all(self.task_success)
