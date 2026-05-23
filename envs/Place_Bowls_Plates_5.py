from ._base_task import Base_Task
from .utils import *
import sapien
import numpy as np
from copy import deepcopy


class Place_Bowls_Plates_5(Base_Task):

    def setup_demo(self, **kwags):
        # Initialize the task environment with the provided keyword arguments.
        super()._init_task_env_(**kwags)

    def load_actors(self):
        # Randomly place the plate on either the left or right side of the workspace.
        plate_side = np.random.choice(["left", "right"])
        plate_x = -0.12 if plate_side == "left" else 0.12

        # Sample a fixed-orientation pose for the plate.
        plate_pose = rand_pose(
            xlim=[plate_x - 0.03, plate_x + 0.03],
            ylim=[-0.15, -0.10],
            rotate_rand=False,
            qpos=[0.5, 0.5, 0.5, 0.5],
        )

        # Create the target plate as a static actor.
        self.plate_id = 0
        self.plate = create_actor(
            self,
            pose=plate_pose,
            modelname="003_plate",
            scale=[0.025, 0.025, 0.025],
            is_static=True,
            convex=True,
        )

        bowl_pose_lst = []

        def check_bowl_pose_valid(bowl_pose):
            # Avoid placing bowls too close to the center line,
            # which can make arm selection and motion ambiguous.
            if abs(bowl_pose.p[0]) < 0.08:
                return False

            # Keep bowls away from the plate at initialization.
            if np.sum((bowl_pose.p[:2] - self.plate.get_pose().p[:2]) ** 2) < 0.0169:
                return False

            # Keep each bowl far enough from previously sampled bowls.
            for old_pose in bowl_pose_lst:
                if np.sum((bowl_pose.p[:2] - old_pose.p[:2]) ** 2) < 0.0169:
                    return False

            return True

        # Sample five valid bowl poses.
        for _ in range(5):
            bowl_pose = rand_pose(
                xlim=[-0.28, 0.28],
                ylim=[-0.05, 0.20],
                qpos=[0.5, 0.5, 0.5, 0.5],
                ylim_prop=True,
                rotate_rand=False,
            )

            # Resample the bowl pose until it satisfies spacing constraints
            # or the trial budget is exhausted.
            max_trials = 200
            trials = 0
            while not check_bowl_pose_valid(bowl_pose) and trials < max_trials:
                bowl_pose = rand_pose(
                    xlim=[-0.28, 0.28],
                    ylim=[-0.05, 0.20],
                    qpos=[0.5, 0.5, 0.5, 0.5],
                    ylim_prop=True,
                    rotate_rand=False,
                )
                trials += 1

            # Store a copy of the accepted bowl pose.
            bowl_pose_lst.append(deepcopy(bowl_pose))

        # Sort bowls from back to front / lower y to higher y.
        # This provides a deterministic stacking order.
        bowl_pose_lst = sorted(bowl_pose_lst, key=lambda x: x.p[1])

        def create_bowl(bowl_pose):
            # Create one bowl actor with a fixed model variant.
            return create_actor(
                self,
                pose=bowl_pose,
                modelname="002_bowl",
                model_id=3,
                convex=True,
            )

        # Create five bowl actors.
        self.bowl1 = create_bowl(bowl_pose_lst[0])
        self.bowl2 = create_bowl(bowl_pose_lst[1])
        self.bowl3 = create_bowl(bowl_pose_lst[2])
        self.bowl4 = create_bowl(bowl_pose_lst[3])
        self.bowl5 = create_bowl(bowl_pose_lst[4])

        # Store all bowls in stacking order.
        self.bowls = [
            self.bowl1,
            self.bowl2,
            self.bowl3,
            self.bowl4,
            self.bowl5,
        ]

        # Track whether each bowl is successfully included in the stack.
        self.task_success = [0, 0, 0, 0, 0]

        # Register prohibited areas around the plate and bowls.
        self.add_prohibit_area(self.plate, padding=0.10)
        self.add_prohibit_area(self.bowl1, padding=0.07)
        self.add_prohibit_area(self.bowl2, padding=0.07)
        self.add_prohibit_area(self.bowl3, padding=0.07)
        self.add_prohibit_area(self.bowl4, padding=0.07)
        self.add_prohibit_area(self.bowl5, padding=0.07)

        # Add an extra prohibited area near the target placement region.
        target_pose = [-0.1, -0.15, 0.1, -0.05]
        self.prohibited_area.append(target_pose)

        # Target orientation used when placing bowls.
        self.quat_of_target_pose = [0, 0.707, 0.707, 0]

    def move_bowl(self, actor, target_pose):
        # Choose the arm based on the bowl's current x position.
        actor_pose = actor.get_pose().p
        arm_tag = ArmTag("left" if actor_pose[0] < 0 else "right")

        # If using the same arm as before, directly grasp the bowl.
        if self.las_arm is None or arm_tag == self.las_arm:
            self.move(
                self.grasp_actor(
                    actor,
                    arm_tag=arm_tag,
                    contact_point_id=[0, 2][int(arm_tag == "left")],
                    pre_grasp_dis=0.1,
                )
            )
        else:
            # If switching arms, grasp with the new arm while sending the old arm home.
            self.move(
                self.grasp_actor(
                    actor,
                    arm_tag=arm_tag,
                    contact_point_id=[0, 2][int(arm_tag == "left")],
                    pre_grasp_dis=0.1,
                ),
                self.back_to_origin(arm_tag=arm_tag.opposite),
            )

        # Lift the bowl before moving to the placement target.
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1, move_axis="arm"))

        # Place the bowl at the target pose, using the fixed target orientation.
        self.move(
            self.place_actor(
                actor,
                target_pose=target_pose.tolist() + self.quat_of_target_pose,
                arm_tag=arm_tag,
                functional_point_id=0,
                pre_dis=0.09,
                dis=0,
                constrain="align",
            )
        )

        # Lift the gripper after placing the bowl.
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.09, move_axis="arm"))

        # Record the last used arm for future arm-switch handling.
        self.las_arm = arm_tag
        return arm_tag

    def _is_bowl_on_plate(self, bowl_pos, plate_pos, eps_xy):
        # Check whether a bowl is horizontally aligned with the plate.
        return np.all(np.abs(bowl_pos[:2] - plate_pos[:2]) < eps_xy)

    def _is_bowl_on_bowl(self, upper_pos, lower_pos, eps_xy, z_min=0.005, z_max=0.08):
        # Check horizontal alignment between two bowls.
        xy_ok = np.all(np.abs(upper_pos[:2] - lower_pos[:2]) < eps_xy)

        # Check that the upper bowl is above the lower bowl by a plausible stack height.
        z_diff = upper_pos[2] - lower_pos[2]
        z_ok = z_min < z_diff < z_max

        return xy_ok and z_ok

    def _get_bowl_stack_chain(self):
        # Get the plate position and define tolerances for plate/bowl alignment.
        plate_pose = self.plate.get_pose().p
        eps_plate_xy = np.array([0.05, 0.05])
        eps_stack_xy = np.array([0.04, 0.04])

        # Collect bowl indices, actor references, and current positions.
        bowl_infos = []
        for idx, bowl in enumerate(self.bowls):
            bowl_infos.append({
                "idx": idx,
                "actor": bowl,
                "pose": bowl.get_pose().p,
            })

        # Sort bowls from lowest to highest by z position.
        bowl_infos.sort(key=lambda x: x["pose"][2])

        # Find bowls that can serve as the base bowl on the plate.
        base_candidates = []
        for info in bowl_infos:
            if self._is_bowl_on_plate(info["pose"], plate_pose, eps_plate_xy):
                base_candidates.append(info)

        # No stack exists if no bowl is on the plate.
        if len(base_candidates) == 0:
            return []

        # Start the chain from the lowest bowl on the plate.
        current = min(base_candidates, key=lambda x: x["pose"][2])
        chain = [current["idx"]]
        used = {current["idx"]}

        # Repeatedly search for the next bowl stacked on top of the current bowl.
        while True:
            candidates = []
            for info in bowl_infos:
                if info["idx"] in used:
                    continue
                if self._is_bowl_on_bowl(info["pose"], current["pose"], eps_stack_xy):
                    candidates.append(info)

            if len(candidates) == 0:
                break

            # Choose the closest valid bowl above the current one as the next stack element.
            current = min(candidates, key=lambda x: x["pose"][2] - chain.__len__())
            chain.append(current["idx"])
            used.add(current["idx"])

        return chain

    def update_progress(self):
        # Reset per-bowl success flags.
        self.task_success = [0, 0, 0, 0, 0]

        # Mark every bowl that appears in the detected stack chain.
        chain = self._get_bowl_stack_chain()
        for idx in chain:
            self.task_success[idx] = 1

    def play_once(self):
        # Reset the last-used arm before executing the demonstration.
        self.las_arm = None

        # First place bowl1 onto the plate.
        plate_target_pose = np.array(self.plate.get_functional_point(0)[:3])
        arm_tag1 = self.move_bowl(self.bowl1, plate_target_pose)

        # Stack each following bowl on top of the previous bowl.
        arm_tag2 = self.move_bowl(self.bowl2, self.bowl1.get_pose().p + np.array([0, 0, 0.05]))
        arm_tag3 = self.move_bowl(self.bowl3, self.bowl2.get_pose().p + np.array([0, 0, 0.05]))
        arm_tag4 = self.move_bowl(self.bowl4, self.bowl3.get_pose().p + np.array([0, 0, 0.05]))
        arm_tag5 = self.move_bowl(self.bowl5, self.bowl4.get_pose().p + np.array([0, 0, 0.05]))

        # Store placeholders for instruction generation and evaluation.
        self.info["info"] = {
            "{A}": f"003_plate/base{self.plate_id}",
            "{B}": "002_bowl/base3",
            "{a}": str(arm_tag1),
            "{b}": str(arm_tag2),
            "{c}": str(arm_tag3),
            "{d}": str(arm_tag4),
            "{e}": str(arm_tag5),
        }
        return self.info

    def check_success(self):
        # Refresh stack-progress flags.
        self.update_progress()

        # The task succeeds only if all bowls are stacked and both grippers are open.
        return (
            self.task_success == [1, 1, 1, 1, 1]
            and self.is_left_gripper_open()
            and self.is_right_gripper_open()
        )