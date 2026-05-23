from ._base_task import Base_Task
from .utils import *
import sapien
import math
from ._GLOBAL_CONFIGS import *
from copy import deepcopy
import numpy as np


class Move_Playing_Cards_Away_3(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def _sample_card_pose(self, existing_xy=None, min_dist=0.05, max_try=100):
        if existing_xy is None:
            existing_xy = []

        for _ in range(max_try):
            rand_pos = rand_pose(
                xlim=[-0.20, 0.20],
                ylim=[-0.20, 0.10],
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=True,
                rotate_lim=[0, 3.14, 0],
            )

            x, y = rand_pos.p[0], rand_pos.p[1]

            ok = True
            for ex, ey in existing_xy:
                if np.linalg.norm([x - ex, y - ey]) < min_dist:
                    ok = False
                    break

            if ok:
                return rand_pos

        raise RuntimeError(f"Failed to sample card pose with min_dist={min_dist}")

    def load_actors(self):
        self.task_success = [0, 0, 0]
        self.playingcards_list = []
        self.playingcards_id_list = []
        existing_xy = []

        for i in range(3):
            rand_pos = self._sample_card_pose(
                existing_xy=existing_xy,
                min_dist=0.15,
                max_try=200,
            )
            playingcards_id = np.random.choice([0, 1, 2], 1)[0]

            card = create_actor(
                scene=self,
                pose=rand_pos,
                modelname="081_playingcards",
                convex=True,
                model_id=playingcards_id,
            )

            self.playingcards_list.append(card)
            self.playingcards_id_list.append(playingcards_id)
            self.add_prohibit_area(card, padding=0.01)

            existing_xy.append([rand_pos.p[0], rand_pos.p[1]])

        self.target_pose = self.playingcards_list[0].get_pose()

    def pick_and_place_block(
        self,
        actor,
        target_xy,
        is_last=False,
        lift_z=0.10,
        place_down_z=-0.07,
        pre_grasp_dis=0.10,
        grasp_dis=0.01,
    ):
        arm_tag = ArmTag("right" if target_xy[0] > 0 else "left")

        if hasattr(self, "last_arm_tag") and self.last_arm_tag is not None and self.last_arm_tag != arm_tag:
            self.move(
                self.grasp_actor(
                    actor,
                    arm_tag=arm_tag,
                    pre_grasp_dis=pre_grasp_dis,
                    grasp_dis=grasp_dis,
                ),
                self.back_to_origin(arm_tag=arm_tag.opposite)
            )
        else:
            self.move(
                self.grasp_actor(
                    actor,
                    arm_tag=arm_tag,
                    pre_grasp_dis=pre_grasp_dis,
                    grasp_dis=grasp_dis,
                )
            )

        self.move(
            self.move_by_displacement(
                arm_tag,
                z=lift_z,
            )
        )

        cur_xy = actor.get_pose().p[:2]
        delta_xy = np.array(target_xy) - cur_xy

        self.move(
            self.move_by_displacement(
                arm_tag,
                x=float(delta_xy[0]),
                y=float(delta_xy[1]),
            )
        )

        self.move(
            self.move_by_displacement(
                arm_tag,
                z=place_down_z,
            )
        )

        self.move(self.open_gripper(arm_tag))

        if not is_last:
            self.move(
                self.move_by_displacement(
                    arm_tag,
                    z=lift_z,
                )
            )

        self.last_arm_tag = arm_tag

        return arm_tag

    def play_once(self):
        self.last_arm_tag = None

        cards_with_ids = list(zip(self.playingcards_list, self.playingcards_id_list))
        cards_with_ids.sort(key=lambda x: x[0].get_pose().p[0], reverse=False)

        arm_tags = []
        for i, (card, card_id) in enumerate(cards_with_ids):
            cur_pose = card.get_pose().p
            cur_x, cur_y = cur_pose[0], cur_pose[1]

            target_x = 0.3 if cur_x > 0 else -0.3
            target_xy = [target_x, float(cur_y)]

            arm_tag = self.pick_and_place_block(
                actor=card,
                target_xy=target_xy,
                is_last=(i == len(cards_with_ids) - 1),
            )
            arm_tags.append(str(arm_tag))

        self.info["info"] = {
            # "{A}": "three playing cards",
            # "{a}": " / ".join(arm_tags),
        }
        return self.info

    def update_progress(self):
        for i, card in enumerate(self.playingcards_list):
            x = card.get_pose().p[0]
            self.task_success[i] = int(abs(x) >= 0.23)

    def check_success(self):
        self.update_progress()
        return self.task_success == [1, 1, 1]