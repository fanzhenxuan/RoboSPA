from ._base_task import Base_Task
from .utils import *
import sapien
import numpy as np
from copy import deepcopy


class Place_Bowls_Plates_3(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        # =========================
        # 1. 创建 plate
        # =========================
        plate_side = np.random.choice(["left", "right"])
        # print("盘子在", plate_side)
        plate_x = -0.12 if plate_side == "left" else 0.12

        plate_pose = rand_pose(
            xlim=[plate_x - 0.03, plate_x + 0.03],
            ylim=[-0.15, -0.10],
            rotate_rand=False,
            qpos=[0.5, 0.5, 0.5, 0.5],
        )

        self.plate_id = 0
        self.plate = create_actor(
            self,
            pose=plate_pose,
            modelname="003_plate",
            scale=[0.025, 0.025, 0.025],
            is_static=True,
            convex=True,
        )

        # =========================
        # 2. 创建三个 bowl
        # =========================
        bowl_pose_lst = []

        def check_bowl_pose_valid(bowl_pose):
            if abs(bowl_pose.p[0]) < 0.08:
                return False

            if np.sum((bowl_pose.p[:2] - self.plate.get_pose().p[:2]) ** 2) < 0.0169:
                return False

            for old_pose in bowl_pose_lst:
                if np.sum((bowl_pose.p[:2] - old_pose.p[:2]) ** 2) < 0.0169:
                    return False

            return True

        for _ in range(3):
            bowl_pose = rand_pose(
                xlim=[-0.28, 0.28],
                ylim=[-0.05, 0.20],
                qpos=[0.5, 0.5, 0.5, 0.5],
                ylim_prop=True,
                rotate_rand=False,
            )
            while not check_bowl_pose_valid(bowl_pose):
                bowl_pose = rand_pose(
                    xlim=[-0.28, 0.28],
                    ylim=[-0.05, 0.20],
                    qpos=[0.5, 0.5, 0.5, 0.5],
                    ylim_prop=True,
                    rotate_rand=False,
                )
            bowl_pose_lst.append(deepcopy(bowl_pose))

        bowl_pose_lst = sorted(bowl_pose_lst, key=lambda x: x.p[1])

        def create_bowl(bowl_pose):
            return create_actor(
                self,
                pose=bowl_pose,
                modelname="002_bowl",
                model_id=3,
                convex=True,
            )

        self.bowl1 = create_bowl(bowl_pose_lst[0])
        self.bowl2 = create_bowl(bowl_pose_lst[1])
        self.bowl3 = create_bowl(bowl_pose_lst[2])

        self.bowls = [
            self.bowl1,
            self.bowl2,
            self.bowl3,
        ]

        self.task_success = [0, 0, 0]

        # =========================
        # 3. prohibit area
        # =========================
        self.add_prohibit_area(self.plate, padding=0.10)
        self.add_prohibit_area(self.bowl1, padding=0.07)
        self.add_prohibit_area(self.bowl2, padding=0.07)
        self.add_prohibit_area(self.bowl3, padding=0.07)

        target_pose = [-0.1, -0.15, 0.1, -0.05]
        self.prohibited_area.append(target_pose)

        self.quat_of_target_pose = [0, 0.707, 0.707, 0]

    def move_bowl(self, actor, target_pose):
        actor_pose = actor.get_pose().p
        arm_tag = ArmTag("left" if actor_pose[0] < 0 else "right")

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
            self.move(
                self.grasp_actor(
                    actor,
                    arm_tag=arm_tag,
                    contact_point_id=[0, 2][int(arm_tag == "left")],
                    pre_grasp_dis=0.1,
                ),
                self.back_to_origin(arm_tag=arm_tag.opposite),
            )

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1, move_axis="arm"))

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

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.09, move_axis="arm"))
        self.las_arm = arm_tag
        return arm_tag

    def _is_bowl_on_plate(self, bowl_pos, plate_pos, eps_xy):
        return np.all(np.abs(bowl_pos[:2] - plate_pos[:2]) < eps_xy)

    def _is_bowl_on_bowl(self, upper_pos, lower_pos, eps_xy, z_min=0.005, z_max=0.08):
        xy_ok = np.all(np.abs(upper_pos[:2] - lower_pos[:2]) < eps_xy)
        z_diff = upper_pos[2] - lower_pos[2]
        z_ok = z_min < z_diff < z_max
        return xy_ok and z_ok

    def _get_bowl_stack_chain(self):
        """
        返回盘子上的连续碗堆链条，元素是 bowls 的编号 index（0~2）
        不强制 bowl1/bowl2/... 的顺序，只看空间堆叠关系。
        """
        plate_pose = self.plate.get_pose().p
        eps_plate_xy = np.array([0.05, 0.05])
        eps_stack_xy = np.array([0.04, 0.04])

        bowl_infos = []
        for idx, bowl in enumerate(self.bowls):
            bowl_infos.append({
                "idx": idx,
                "actor": bowl,
                "pose": bowl.get_pose().p,
            })

        bowl_infos.sort(key=lambda x: x["pose"][2])

        base_candidates = []
        for info in bowl_infos:
            if self._is_bowl_on_plate(info["pose"], plate_pose, eps_plate_xy):
                base_candidates.append(info)

        if len(base_candidates) == 0:
            return []

        current = min(base_candidates, key=lambda x: x["pose"][2])
        chain = [current["idx"]]
        used = {current["idx"]}

        while True:
            candidates = []
            for info in bowl_infos:
                if info["idx"] in used:
                    continue
                if self._is_bowl_on_bowl(info["pose"], current["pose"], eps_stack_xy):
                    candidates.append(info)

            if len(candidates) == 0:
                break

            current = min(candidates, key=lambda x: x["pose"][2] - current["pose"][2])
            chain.append(current["idx"])
            used.add(current["idx"])

        return chain

    def update_progress(self):
        self.task_success = [0, 0, 0]

        chain = self._get_bowl_stack_chain()
        for idx in chain:
            self.task_success[idx] = 1

    def play_once(self):
        self.las_arm = None

        plate_target_pose = np.array(self.plate.get_functional_point(0)[:3])
        arm_tag1 = self.move_bowl(self.bowl1, plate_target_pose)
        arm_tag2 = self.move_bowl(self.bowl2, self.bowl1.get_pose().p + np.array([0, 0, 0.05]))
        arm_tag3 = self.move_bowl(self.bowl3, self.bowl2.get_pose().p + np.array([0, 0, 0.05]))

        self.info["info"] = {
            "{A}": f"003_plate/base{self.plate_id}",
            "{B}": "002_bowl/base3",
            # "{C}": "002_bowl/base3",
            # "{D}": "002_bowl/base3",
            "{a}": str(arm_tag1),
            "{b}": str(arm_tag2),
            "{c}": str(arm_tag3),
        }
        return self.info

    def check_success(self):
        self.update_progress()
      
        return (
            self.task_success == [1, 1, 1]
            and self.is_left_gripper_open()
            and self.is_right_gripper_open()
        )