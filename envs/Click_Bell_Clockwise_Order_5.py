from ._base_task import Base_Task
from .utils import *
import numpy as np


class Click_Bell_Clockwise_Order_5(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def _sample_circle_positions(self):
        visible_x = (-0.255, 0.255)
        visible_y = (-0.225, 0.055)
        min_pair_dist = 0.085

        for _ in range(400):
            base_radius = np.random.uniform(0.088, 0.138)
            x_scale = np.random.uniform(0.80, 1.18)
            y_scale = np.random.uniform(0.80, 1.14)

            rel_points = []
            for idx in range(5):
                base_angle = (np.pi / 2.0) - idx * (2.0 * np.pi / 5.0)
                angle_jitter = np.random.uniform(-0.05, 0.05) if idx == 0 else np.random.uniform(-0.22, 0.22)
                theta = base_angle + angle_jitter
                radius = base_radius * np.random.uniform(0.85, 1.18)
                x = x_scale * radius * np.cos(theta) + np.random.uniform(-0.012, 0.012)
                y = y_scale * radius * np.sin(theta) + np.random.uniform(-0.012, 0.012)
                rel_points.append(np.array([x, y], dtype=float))

            ys = [float(point[1]) for point in rel_points]
            if ys[0] <= max(ys[1:]) + 0.012:
                continue

            cw_angles = [((np.pi / 2.0) - np.arctan2(point[1], point[0])) % (2.0 * np.pi) for point in rel_points]
            if not all(cw_angles[i] + 0.12 < cw_angles[i + 1] for i in range(4)):
                continue

            valid = True
            for i in range(5):
                for j in range(i + 1, 5):
                    if np.linalg.norm(rel_points[i] - rel_points[j]) < min_pair_dist:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue

            rel_x_min = min(float(point[0]) for point in rel_points)
            rel_x_max = max(float(point[0]) for point in rel_points)
            rel_y_min = min(float(point[1]) for point in rel_points)
            rel_y_max = max(float(point[1]) for point in rel_points)

            center_x_min = visible_x[0] - rel_x_min
            center_x_max = visible_x[1] - rel_x_max
            center_y_min = visible_y[0] - rel_y_min
            center_y_max = visible_y[1] - rel_y_max
            if center_x_min >= center_x_max or center_y_min >= center_y_max:
                continue

            center_x = np.random.uniform(center_x_min, center_x_max)
            center_y = np.random.uniform(center_y_min, center_y_max)
            points = [point + np.array([center_x, center_y], dtype=float) for point in rel_points]

            return [
                rand_pose(
                    xlim=[float(point[0]), float(point[0])],
                    ylim=[float(point[1]), float(point[1])],
                    rotate_rand=False,
                    qpos=[0.5, 0.5, 0.5, 0.5],
                )
                for point in points
            ]

        raise RuntimeError("Failed to sample a valid five-bell circle layout")

    def _arm_for_bell(self, bell):
        return ArmTag("right" if float(bell.get_pose().p[0]) >= 0 else "left")

    def _arm_closed(self, arm_tag):
        return self.is_left_gripper_close() if str(arm_tag) == "left" else self.is_right_gripper_close()

    def _is_click_success(self, bell, arm_tag):
        if not self._arm_closed(arm_tag):
            return False
        bell_pose = bell.get_contact_point(0)[:3]
        positions = self.get_gripper_actor_contact_position("050_bell")
        eps_xy = np.array([0.025, 0.025], dtype=float)
        for position in positions:
            if np.all(np.abs(position[:2] - bell_pose[:2]) < eps_xy) and abs(position[2] - bell_pose[2]) < 0.03:
                return True
        return False

    def _has_left_click_area(self, bell):
        bell_pose = bell.get_contact_point(0)[:3]
        positions = self.get_gripper_actor_contact_position("050_bell")
        if len(positions) == 0:
            return True
        eps_xy = np.array([0.025, 0.025], dtype=float)
        for position in positions:
            if np.all(np.abs(position[:2] - bell_pose[:2]) < eps_xy) and abs(position[2] - bell_pose[2]) < 0.03:
                return False
        return True

    def _prepare_new_arm(self, prev_arm_tag, new_arm_tag):
        if prev_arm_tag is None:
            self.move(self.close_gripper(new_arm_tag, pos=0.0))
            return
        if str(new_arm_tag) != str(prev_arm_tag):
            self.move(
                self.back_to_origin(prev_arm_tag),
                self.close_gripper(new_arm_tag, pos=0.0),
            )
            return
        self.move(self.close_gripper(new_arm_tag, pos=0.0))

    def load_actors(self):
        self.stage_sum = 5
        self.stage = 0
        self.task_success = [False] * self.stage_sum
        self.has_left_press_area = True
        self.sequence_names = [
            "top bell",
            "upper-right bell",
            "lower-right bell",
            "lower-left bell",
            "upper-left bell",
        ]

        bell_poses = self._sample_circle_positions()
        self.bell_id = int(np.random.choice([0, 1]))
        self.bells = []
        self.bell_ids = []
        for pose in bell_poses:
            bell = create_actor(
                scene=self,
                pose=pose,
                modelname="050_bell",
                convex=True,
                model_id=self.bell_id,
                is_static=True,
            )
            self.add_prohibit_area(bell, padding=0.07)
            self.bells.append(bell)
            self.bell_ids.append(self.bell_id)

    def update_progress(self, arm_tag):
        if self.stage >= self.stage_sum:
            return
        target_bell = self.bells[self.stage]
        if self._has_left_click_area(target_bell):
            self.has_left_press_area = True
        if self.has_left_press_area and self._is_click_success(target_bell, arm_tag):
            self.task_success[self.stage] = True
            self.stage += 1
            self.has_left_press_area = False

    def _do_one_click(self, bell, arm_tag):
        self.move(
            self.grasp_actor(
                bell,
                arm_tag=arm_tag,
                pre_grasp_dis=0.10,
                grasp_dis=0.10,
                contact_point_id=0,
            )
        )
        self.move(self.move_by_displacement(arm_tag, z=-0.045))
        self.update_progress(arm_tag)
        self.move(self.move_by_displacement(arm_tag, z=0.055))
        self.update_progress(arm_tag)

    def play_once(self):
        prev_arm_tag = None
        for bell in self.bells:
            arm_tag = self._arm_for_bell(bell)
            self._prepare_new_arm(prev_arm_tag, arm_tag)
            self._do_one_click(bell, arm_tag)
            prev_arm_tag = arm_tag

        self.info["layout"] = {
            "object_count": self.stage_sum,
            "sequence_names": self.sequence_names,
            "start_index": 0,
            "order": list(range(self.stage_sum)),
            "bell_assets": [f"050_bell/base{bell_id}" for bell_id in self.bell_ids],
            "bell_positions": [
                [float(bell.get_pose().p[0]), float(bell.get_pose().p[1]), float(bell.get_pose().p[2])]
                for bell in self.bells
            ],
        }
        self.info["info"] = {
            "{B}": "top bell",
        }
        return self.info

    def check_success(self):
        return all(self.task_success)
