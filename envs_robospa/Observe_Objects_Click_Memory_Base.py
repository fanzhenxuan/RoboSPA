from copy import deepcopy
from ._base_task import Base_Task
from .utils import *
import sapien
import numpy as np
import math


class ObserveObjectsClickMemoryBase(Base_Task):
    """
    Task:
    1. 参考平台上从左到右放 N 个静态参考物体（定义点击顺序）
    2. 桌面上只放这 N 个对应物体，不放干扰物
    3. observe 2 秒
    4. 挡住参考平台
    5. 机器人按顺序 click 这 N 个桌面目标物体
    6. 点错直接 fail

    当前版本：
    - bread / stapler / can / playingcards / bell 五选 N
    - 无干扰物
    - 每个物体只点击一次，因此不需要“离开再计数”的防抖逻辑
    """

    STEP_NUM = 3

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        # =========================
        # 基础状态
        # =========================
        self.stage_sum = self.STEP_NUM
        self.stage = 0
        self.fail_flag = False
        self.wall = None

        self.get_obs_cnt = 0
        self.task_success = [0]*self.stage_sum
        # 所有桌面物体（也是目标物体）
        self.table_objects = []
        self.table_modelnames = []
        self.table_ids = []

        self.target_objects = []
        self.target_modelnames = []
        self.target_ids = []

        # 参考平台上的静态参考物体
        self.ref_objects = []
        self.ref_modelnames = []
        self.ref_ids = []

        # =========================
        # 参考平台
        # =========================
        self.ref_platform = create_box(
            self.scene,
            sapien.Pose(p=[0, 0.15, 0.82]),
            half_size=[0.24, 0.05, 0.01],
            color=(0.8, 0.8, 0.8),
            name="ref_platform",
            is_static=True,
        )

        # =========================
        # 候选物体
        # =========================
        candidate_models = [
            "075_bread",
            "048_stapler",
            "071_can",
            "081_playingcards",
            "050_bell",
        ]

        if self.STEP_NUM > len(candidate_models):
            raise ValueError(f"STEP_NUM={self.STEP_NUM} is larger than number of candidate models.")

        def sample_model_id(modelname):
            if modelname == "075_bread":
                return int(np.random.choice([0, 1, 3, 5, 6], 1)[0])
            elif modelname == "048_stapler":
                return int(np.random.choice([0, 1, 2, 3, 4, 5, 6], 1)[0])
            elif modelname == "071_can":
                return int(np.random.choice([0, 1, 2, 3, 5, 6], 1)[0])
            elif modelname == "081_playingcards":
                return int(np.random.choice([0, 1, 2], 1)[0])
            elif modelname == "050_bell":
                return int(np.random.choice([0, 1], 1)[0])
            else:
                raise ValueError(f"Unsupported modelname: {modelname}")

        # 随机挑 STEP_NUM 个类别
        shuffled_models = deepcopy(candidate_models)
        np.random.shuffle(shuffled_models)
        selected_models = shuffled_models[:self.STEP_NUM]

        selected_pairs = []
        for modelname in selected_models:
            selected_pairs.append((modelname, sample_model_id(modelname)))

        # =========================
        # 参考平台上放 STEP_NUM 个静态参考物体
        # 从左到右定义点击顺序
        # =========================
        if self.STEP_NUM == 1:
            ref_xs = [0.0]
        else:
            ref_xs = np.linspace(-0.18, 0.18, self.STEP_NUM).tolist()

        for i, (modelname, model_id) in enumerate(selected_pairs):
            if modelname == "075_bread":
                ref_pose = rand_pose(
                    xlim=[ref_xs[i], ref_xs[i]],
                    ylim=[0.15, 0.15],
                    zlim=[0.832, 0.832],
                    qpos=[0.707, 0.707, 0.0, 0.0],
                    rotate_rand=True,
                    rotate_lim=[0, np.pi / 4, 0],
                )
            else:
                ref_pose = rand_pose(
                    xlim=[ref_xs[i], ref_xs[i]],
                    ylim=[0.15, 0.15],
                    zlim=[0.85, 0.85],
                    qpos=[0.707, 0.707, 0, 0],
                )

            ref_obj = create_actor(
                scene=self,
                pose=ref_pose,
                modelname=modelname,
                convex=True,
                model_id=model_id,
                is_static=True,
            )
            self.ref_objects.append(ref_obj)
            self.ref_modelnames.append(modelname)
            self.ref_ids.append(model_id)

        # =========================
        # 桌面采样区域（放大版）
        # =========================
        def sample_pose(xlim=[-0.27, 0.27], ylim=[-0.21, -0.05], rotate=False):
            return rand_pose(
                xlim=xlim,
                ylim=ylim,
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=rotate,
                rotate_lim=[0, 3.14, 0] if rotate else None,
            )

        def far_enough(p0, p1, min_dist=0.10):
            return np.linalg.norm(p0.p[:2] - p1.p[:2]) >= min_dist

        def need_rotate(modelname):
            return modelname in ["048_stapler", "081_playingcards", "075_bread"]

        def sample_valid_table_pose(existing_poses, modelname, max_trials=300):
            trials = 0
            pose = sample_pose(rotate=need_rotate(modelname))
            while trials < max_trials:
                cond_center = abs(pose.p[0]) >= 0.05
                cond_far = True
                for old_pose in existing_poses:
                    if not far_enough(pose, old_pose, min_dist=0.10):
                        cond_far = False
                        break
                if cond_center and cond_far:
                    return pose
                pose = sample_pose(rotate=need_rotate(modelname))
                trials += 1
            raise RuntimeError(f"Failed to sample valid table pose for {modelname} within {max_trials} tries.")

        # =========================
        # 桌面上只放 STEP_NUM 个目标物体
        # =========================
        existing_poses = []
        for modelname, model_id in selected_pairs:
            pose = sample_valid_table_pose(existing_poses, modelname)
            actor = create_actor(
                scene=self,
                pose=pose,
                modelname=modelname,
                convex=True,
                model_id=model_id,
                is_static=True,
            )
            existing_poses.append(pose)

            self.table_objects.append(actor)
            self.table_modelnames.append(modelname)
            self.table_ids.append(model_id)

            self.target_objects.append(actor)
            self.target_modelnames.append(modelname)
            self.target_ids.append(model_id)

        # =========================
        # prohibit area
        # =========================
        for obj, modelname in zip(self.table_objects, self.table_modelnames):
            if modelname == "075_bread":
                self.add_prohibit_area(obj, padding=0.08)
            elif modelname == "048_stapler":
                self.add_prohibit_area(obj, padding=0.05)
            elif modelname == "071_can":
                self.add_prohibit_area(obj, padding=0.10)
            elif modelname == "081_playingcards":
                self.add_prohibit_area(obj, padding=0.10)
            elif modelname == "050_bell":
                self.add_prohibit_area(obj, padding=0.07)
        self.orig_left_endpose = self.get_arm_pose("left")
        self.orig_right_endpose = self.get_arm_pose("right")        
    def add_wall(self):
        if self.wall is not None:
            return
        self.wall = create_box(
            self.scene,
            sapien.Pose(p=[0, 0.05, 0.95]),
            half_size=[0.32, 0.005, 0.2],
            color=(1, 0.9, 0.9),
            name="wall",
            is_static=True,
        )

    # -------------------------
    # click config
    # -------------------------
    def _get_click_config(self, modelname):
        if modelname == "075_bread":
            return {
                "contact_point_id": 0,
                "pre_grasp_dis": 0.08,
                "grasp_dis": 0.08,
                "press_depth": 0.05,
                "xy_eps": 0.03,
                "z_eps": 0.03,
            }
        elif modelname == "048_stapler":
            return {
                "contact_point_id": 2,
                "pre_grasp_dis": 0.10,
                "grasp_dis": 0.10,
                "press_depth": 0.075,
                "xy_eps": 0.03,
                "z_eps": 0.03,
            }
        elif modelname == "071_can":
            return {
                "contact_point_id": 8,
                "pre_grasp_dis": 0.08,
                "grasp_dis": 0.08,
                "press_depth": 0.05,
                "xy_eps": 0.03,
                "z_eps": 0.04,
            }
        elif modelname == "081_playingcards":
            return {
                "contact_point_id": 0,
                "pre_grasp_dis": 0.08,
                "grasp_dis": 0.08,
                "press_depth": 0.05,
                "xy_eps": 0.03,
                "z_eps": 0.03,
            }
        elif modelname == "050_bell":
            return {
                "contact_point_id": 0,
                "pre_grasp_dis": 0.10,
                "grasp_dis": 0.10,
                "press_depth": 0.045,
                "xy_eps": 0.025,
                "z_eps": 0.03,
            }
        else:
            raise ValueError(f"Unsupported modelname: {modelname}")

    def _arm_closed(self, arm_tag: ArmTag):
        return self.is_left_gripper_close() if str(arm_tag) == "left" else self.is_right_gripper_close()

    def _is_click_success(self, actor, modelname, arm_tag: ArmTag):
        if not self._arm_closed(arm_tag):
            return False

        cfg = self._get_click_config(modelname)
        click_pose = actor.get_contact_point(cfg["contact_point_id"])[:3]
        positions = self.get_gripper_actor_contact_position(modelname)

        for position in positions:
            if (
                np.all(np.abs(position[:2] - click_pose[:2]) < np.array([cfg["xy_eps"], cfg["xy_eps"]]))
                and abs(position[2] - click_pose[2]) < cfg["z_eps"]
            ):
                return True
        return False

    # -------------------------
    # progress
    # -------------------------
    def update_progress(self):
        if self.fail_flag:
            return
        if self.stage >= self.stage_sum:
            return

        current_target = self.target_objects[self.stage]
        current_modelname = self.target_modelnames[self.stage]
        current_arm = ArmTag("right" if current_target.get_pose().p[0] > 0 else "left")

        # 当前阶段正确点击就直接 +1
        if self._is_click_success(current_target, current_modelname, current_arm):
            self.stage += 1
            self.task_success[self.stage-1] = 1

    # -------------------------
    # 单次点击动作
    # -------------------------
    def _do_one_click(self, actor, modelname, arm_tag, prev_arm_tag=None):
        cfg = self._get_click_config(modelname)

        if prev_arm_tag is not None and arm_tag != prev_arm_tag:
            self.move(
                self.grasp_actor(
                    actor,
                    arm_tag=arm_tag,
                    pre_grasp_dis=cfg["pre_grasp_dis"],
                    grasp_dis=cfg["grasp_dis"],
                    contact_point_id=cfg["contact_point_id"],
                ),
                self.back_to_origin(arm_tag=arm_tag.opposite),
            )
        else:
            self.move(
                self.grasp_actor(
                    actor,
                    arm_tag=arm_tag,
                    pre_grasp_dis=cfg["pre_grasp_dis"],
                    grasp_dis=cfg["grasp_dis"],
                    contact_point_id=cfg["contact_point_id"],
                )
            )

        self.move(self.move_by_displacement(arm_tag, z=-cfg["press_depth"]))
        self.update_progress()

        self.move(self.move_by_displacement(arm_tag, z=cfg["press_depth"]))

    # -------------------------
    # main episode
    # -------------------------
    def play_once(self):
        # observe 2 秒
        self.delay(delay_time=2, save_freq=-1)

        # 挡住参考平台
        self.add_wall()

        # 短暂停顿
        self.delay(delay_time=1, save_freq=-1)

        prev_arm_tag = None
        for i in range(self.stage_sum):
            if self.fail_flag:
                break

            actor = self.target_objects[i]
            modelname = self.target_modelnames[i]
            arm_tag = ArmTag("right" if actor.get_pose().p[0] > 0 else "left")

            # 闭合夹爪，模拟按压
            self.move(self.close_gripper(arm_tag=arm_tag, pos=0))

            self._do_one_click(
                actor=actor,
                modelname=modelname,
                arm_tag=arm_tag,
                prev_arm_tag=prev_arm_tag,
            )
            prev_arm_tag = arm_tag
        self.get_obs_cnt = 10000
        self.info["info"] = {}
        return self.info

    def check_success(self):
        if self.fail_flag:
            return False
        self.update_progress()
        print(f"task_success: {self.task_success}, stage: {self.stage}/{self.stage_sum}")
        self.get_obs_cnt += 1
        if self.get_obs_cnt == 500:
            self.add_wall()
        elif self.get_obs_cnt < 500:
            current_left_endpose = self.get_arm_pose("left")
            current_right_endpose = self.get_arm_pose("right")
            if np.linalg.norm(np.array(current_left_endpose[:3]) - np.array(self.orig_left_endpose[:3])) > 0.03 or \
               np.linalg.norm(np.array(current_right_endpose[:3]) - np.array(self.orig_right_endpose[:3])) > 0.03:
                print("Arm position deviation detected!")
                self.fail_flag = True
            return False
        return all(x ==1 for x in self.task_success)
        