from ._base_task import Base_Task
from .utils import *
from ._GLOBAL_CONFIGS import *
import numpy as np


class Press_Stapler_Memory_5(Base_Task):

    BUTTON_COUNT = 5

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        # task_success[i] records whether button i is clicked correctly.
        self.task_success = [0 for _ in range(self.BUTTON_COUNT)]

        # Random target clicks: each button 1~5 times.
        self.target_press_counts = [int(np.random.randint(1, 6)) for _ in range(self.BUTTON_COUNT)]
        self.press_counts = [0 for _ in range(self.BUTTON_COUNT)]
        self.press_flags = [False for _ in range(self.BUTTON_COUNT)]

        # Place all "buttons" in one row (same random y in one sample).
        self.row_y = float(np.random.uniform(-0.12, 0.04))
        x_positions = self._get_line_x_positions(self.BUTTON_COUNT)

        # Use stapler as temporary button proxy (button asset missing).
        self.buttons = []
        self.button_ids = []
        self.button_model_name = "048_stapler"
        for x in x_positions:
            button_id = int(np.random.choice([0, 1, 2, 3, 4, 5, 6]))
            button = create_actor(
                self,
                pose=rand_pose(
                    xlim=[x, x],
                    ylim=[self.row_y, self.row_y],
                    qpos=[0.5, 0.5, 0.5, 0.5],
                    rotate_rand=True,
                    rotate_lim=[0, np.pi, 0],
                ),
                modelname=self.button_model_name,
                convex=True,
                model_id=button_id,
                is_static=True,
            )
            self.buttons.append(button)
            self.button_ids.append(button_id)
            self.add_prohibit_area(button, padding=0.05)

        self.used_arms = set()

    def _get_line_x_positions(self, num_buttons):
        if num_buttons == 1:
            return [0.0]
        return np.linspace(-0.24, 0.24, num_buttons).tolist()

    def _is_button_pressed(self, idx):
        button_pose = self.buttons[idx].get_contact_point(2)[:3]
        positions = self.get_gripper_actor_contact_position(self.button_model_name)
        eps_xy = np.array([0.03, 0.03])

        for position in positions:
            if (
                np.all(np.abs(position[:2] - button_pose[:2]) < eps_xy)
                and abs(position[2] - button_pose[2]) < 0.03
            ):
                return True
        return False

    def update_progress(self):
        # Edge-based click counting (same idea as RMBench press_button).
        for idx in range(self.BUTTON_COUNT):
            is_pressed = self._is_button_pressed(idx)

            if is_pressed and (not self.press_flags[idx]):
                self.press_flags[idx] = True
                self.press_counts[idx] += 1
            elif (not is_pressed) and self.press_flags[idx]:
                self.press_flags[idx] = False

            self.task_success[idx] = int(self.press_counts[idx] == self.target_press_counts[idx])

    def _build_instruction_info(self):
        info = {}

        for idx, cnt in enumerate(self.target_press_counts):
            key = f"{{{chr(ord('A') + idx)}}}"   # {a}, {b}, {c}, ...
            info[key] = str(cnt)

        return info

    def play_once(self):
        # Click from left to right.
        click_order = sorted(range(self.BUTTON_COUNT), key=lambda idx: self.buttons[idx].get_pose().p[0])
        self.last_gripper = None

        for idx in click_order:
            button = self.buttons[idx]
            arm_tag = ArmTag("left" if button.get_pose().p[0] < 0 else "right")
            self.used_arms.add(str(arm_tag))
            target_count = self.target_press_counts[idx]

            # If the next button uses a different arm, switch hands explicitly.
            if self.last_gripper is not None and self.last_gripper != arm_tag:
                self.move(
                    self.grasp_actor(
                        button,
                        arm_tag=arm_tag,
                        pre_grasp_dis=0.08,
                        grasp_dis=0.08,
                        contact_point_id=2,
                    ),
                    self.back_to_origin(arm_tag=arm_tag.opposite),
                )
            else:
                self.move(
                    self.grasp_actor(
                        button,
                        arm_tag=arm_tag,
                        pre_grasp_dis=0.08,
                        grasp_dis=0.08,
                        contact_point_id=2,
                    )
                )
            self.update_progress()

            for _ in range(target_count):
                self.move(
                    self.grasp_actor(
                        button,
                        arm_tag=arm_tag,
                        pre_grasp_dis=0.02,
                        grasp_dis=0.02,
                        contact_point_id=2,
                    )
                )
                self.update_progress()

                self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.05, move_axis="arm"))
                self.update_progress()

            self.last_gripper = arm_tag

        self.info["info"] = self._build_instruction_info()
        return self.info

    def check_success(self):
        self.update_progress()
        return all(v == 1 for v in self.task_success)