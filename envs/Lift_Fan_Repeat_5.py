from ._base_task import Base_Task
from .utils import *
import sapien
import math
from copy import deepcopy
import numpy as np


class Lift_Fan_Repeat_5(Base_Task):

    def setup_demo(self, is_test=False, **kwargs):
        # Initialize the task environment with the given keyword arguments.
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        # Track whether each of the five lift-and-return stages has succeeded.
        self.task_success = [0, 0, 0, 0, 0]

        # The task requires completing five repeated lift-and-place-back cycles.
        self.stage_sum = 5
        self.stage = 0

        # Flags used to record whether the fan has been lifted away from its origin
        # during each stage. A stage is completed only after the fan is lifted and
        # then returned to its initial pose.
        self.left_after_stage1 = False
        self.left_after_stage2 = False
        self.left_after_stage3 = False
        self.left_after_stage4 = False
        self.left_after_stage5 = False

        # Randomly sample the fan's initial pose within a small workspace region.
        rand_pos = rand_pose(
            xlim=[-0.1, 0.1],
            ylim=[-0.15, -0.05],
            # qpos=[0.0, 0.0, 0.707, 0.707],
            qpos=[0.707, 0.707, 0.0, 0.0],
            rotate_rand=False,
            # rotate_lim=[0, 2 * np.pi, 0],
        )

        # Randomly choose one of the available fan asset variants.
        id_list = [4, 5]
        self.fan_id = np.random.choice(id_list)

        # Create the fan actor in the scene.
        self.fan = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="099_fan",
            convex=True,
            model_id=self.fan_id,
        )

        # Make the fan lightweight so it can be lifted easily.
        self.fan.set_mass(0.01)

        # Prevent other objects from being placed too close to the fan.
        self.add_prohibit_area(self.fan, padding=0.07)

        # Record the fan's initial pose so the task can check whether it has
        # been returned to its starting position and orientation.
        self.init_pose_p = self.fan.get_pose().p.copy()
        self.init_pose_q = self.fan.get_pose().q.copy()
        self.target_pose = self.init_pose_p.tolist() + self.init_pose_q.tolist()

    def _fan_at_origin(self):
        # Get the fan's current position and orientation.
        fan_pose = self.fan.get_pose().p
        fan_qpose = self.fan.get_pose().q.copy()

        # Convert the saved initial pose into numpy arrays for comparison.
        target_pose = np.array(self.init_pose_p)
        target_qpose = np.array(self.init_pose_q)

        # Quaternions q and -q represent the same rotation.
        # Flip the current quaternion when needed so the comparison is consistent.
        if np.dot(fan_qpose, target_qpose) < 0:
            fan_qpose *= -1

        # Check whether both position and orientation are close enough to
        # the original pose.
        return (
            np.all(np.abs(fan_pose - target_pose) < np.array([0.04, 0.04, 0.04]))
            and np.all(np.abs(fan_qpose - target_qpose) < np.array([0.05, 0.05, 0.05, 0.05]))
        )

    def _fan_lifted(self):
        # The fan is considered lifted if its height is sufficiently above
        # its initial height.
        fan_z = self.fan.get_pose().p[2]
        origin_z = self.init_pose_p[2]
        return fan_z > origin_z + 0.05

    def update_progress(self):
        # Stage 0: wait for the fan to be lifted, then returned to origin.
        if self.stage == 0:
            if self._fan_lifted():
                self.left_after_stage1 = True
            elif self.left_after_stage1 and self._fan_at_origin():
                self.stage = 1
            return

        # Stage 1: repeat the lift-and-return check for the second cycle.
        if self.stage == 1:
            if self._fan_lifted():
                self.left_after_stage2 = True
            elif self.left_after_stage2 and self._fan_at_origin():
                self.stage = 2
            return

        # Stage 2: repeat the lift-and-return check for the third cycle.
        if self.stage == 2:
            if self._fan_lifted():
                self.left_after_stage3 = True
            elif self.left_after_stage3 and self._fan_at_origin():
                self.stage = 3
            return

        # Stage 3: repeat the lift-and-return check for the fourth cycle.
        if self.stage == 3:
            if self._fan_lifted():
                self.left_after_stage4 = True
            elif self.left_after_stage4 and self._fan_at_origin():
                self.stage = 4
            return

        # Stage 4: repeat the lift-and-return check for the fifth cycle.
        if self.stage == 4:
            if self._fan_lifted():
                self.left_after_stage5 = True
            elif self.left_after_stage5 and self._fan_at_origin():
                self.stage = 5
            return

    def _lift_and_put_back_once(self, arm_tag, lift_z=0.08):
        # Grasp the fan using the selected arm.
        self.move(self.grasp_actor(self.fan, arm_tag=arm_tag, pre_grasp_dis=0.05))

        # Lift the fan upward by the specified distance.
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=lift_z, move_axis="arm"))

        # Update progress after the lift motion.
        self.update_progress()

        # Move the fan back down near its original position.
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=-lift_z + 0.005, move_axis="world"))  # place back down

        # Release the fan after placing it back.
        self.move(self.open_gripper(arm_tag))  # release grip

        # Update progress after the fan is released.
        self.update_progress()

        # Run one extra progress update to catch any delayed settling at the origin.
        self.update_progress()

    def play_once(self):
        # Use the right arm if the fan starts on the right side of the workspace,
        # otherwise use the left arm.
        arm_tag = ArmTag("right" if self.fan.get_pose().p[0] > 0 else "left")

        # Lift and place the fan back down five times.
        self._lift_and_put_back_once(arm_tag, lift_z=0.08)
        self._lift_and_put_back_once(arm_tag, lift_z=0.08)
        self._lift_and_put_back_once(arm_tag, lift_z=0.08)
        self._lift_and_put_back_once(arm_tag, lift_z=0.08)
        self._lift_and_put_back_once(arm_tag, lift_z=0.08)

        # Store task information for instruction generation or evaluation.
        self.info["info"] = {
            # "{A}": f"099_fan/base{self.fan_id}",
            # "{B}":self.stage_sum,
            # "{a}": str(arm_tag),
            "{A}": f"099_fan/base{self.fan_id}",
            "{B}": str(self.stage_sum),
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        # Update the progress before checking success.
        self.update_progress()

        # Convert the current stage number into per-stage success flags.
        for i in range(self.stage_sum):
            self.task_success[i] = int(self.stage >= i + 1)

        # print("self.stage", self.stage)
        # print("self.task_success", self.task_success)

        # The task succeeds only after all five lift-and-return cycles are completed.
        return self.stage >= self.stage_sum