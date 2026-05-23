from ._base_task import Base_Task
from .utils import *
import sapien
from copy import deepcopy


class Place_Phone_Press_Stapler_5(Base_Task):

    def setup_demo(self, is_test=False, **kwargs):
        super()._init_task_env_(table_xy_bias=[0.3, 0], **kwargs)


    def load_actors(self):
        # =========================
        # 1. 左侧大垃圾桶
        # =========================
        self.dustbin = create_actor(
            self,
            pose=sapien.Pose([-0.45, 0, 0], [0.5, 0.5, 0.5, 0.5]),
            modelname="011_dustbin",
            convex=True,
            is_static=True,
        )

        # =========================
        # 2. 左侧桌面垃圾桶
        # =========================
        deskbin_qpos = [0.651892, 0.651428, 0.274378, 0.274584]
        deskbin_pose = rand_pose(
            xlim=[-0.2, -0.07],
            ylim=[-0.2, 0],
            qpos=deskbin_qpos,
            rotate_rand=True,
            rotate_lim=[0, np.pi / 8.5, 0],
        )
        self.deskbin_init_q = deskbin_pose.q

        self.deskbin_id = np.random.choice([0, 3, 7, 8, 9, 10], 1)[0]
        self.deskbin = create_actor(
            self,
            pose=deskbin_pose,
            modelname="063_tabletrashbin",
            model_id=self.deskbin_id,
            convex=True,
        )

        # =========================
        # 3. 垃圾球
        # =========================
        self.garbage_num = 1
        self.sphere_lst = []
        for i in range(self.garbage_num):
            sphere_pose = sapien.Pose(
                [
                    deskbin_pose.p[0] + np.random.rand() * 0.02 - 0.01,
                    deskbin_pose.p[1] + np.random.rand() * 0.02 - 0.01,
                    0.78 + i * 0.005,
                ],
                [1, 0, 0, 0],
            )
            sphere = create_sphere(
                self.scene,
                pose=sphere_pose,
                radius=0.008,
                color=[1, 0, 0],
                name="garbage",
            )
            self.sphere_lst.append(sphere)
            self.sphere_lst[-1].find_component_by_type(
                sapien.physx.PhysxRigidDynamicComponent
            ).mass = 0.0001

        # =========================
        # 4. 倒垃圾动作
        # =========================
        self.middle_pose = [0, -0.1, 0.741 + self.table_z_bias, 1, 0, 0, 0]

        left_arm = ArmTag("left")
        pour_pose_1 = [-0.45, -0.05, 1.05, -0.694654, -0.178228, 0.165979, -0.676862]
        pour_pose_2 = [
            -0.45,
            -0.05 - np.random.rand() * 0.02,
            1.05 - np.random.rand() * 0.02,
            -0.694654,
            -0.178228,
            0.165979,
            -0.676862,
        ]
        self.pour_actions = (
            left_arm,
            [
                Action(left_arm, "move", pour_pose_1),
                Action(left_arm, "move", pour_pose_2),
            ],
        )

        # =========================
        # 5. phone 与 stand
        # =========================
        ori_quat = [
            [0.707, 0.707, 0, 0],
            [0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5, -0.5],
            [0.5, -0.5, 0.5, -0.5],
        ]

        phone_x_lim = [0.05, 0.25]
        stand_x_lim = [0, 0.15]
        stand_qpos = [0.707, 0.707, 0, 0]

        self.phone_id = np.random.choice([0, 1, 2, 4], 1)[0]
        phone_pose = rand_pose(
            xlim=phone_x_lim,
            ylim=[-0.2, 0.0],
            qpos=ori_quat[self.phone_id],
            rotate_rand=True,
            rotate_lim=[0, 0.7, 0],
        )
        self.phone = create_actor(
            scene=self,
            pose=phone_pose,
            modelname="077_phone",
            convex=True,
            model_id=self.phone_id,
        )
        self.phone.set_mass(0.01)

        def sample_stand_pose():
            return rand_pose(
                xlim=stand_x_lim,
                ylim=[0, 0.2],
                qpos=stand_qpos,
                rotate_rand=False,
            )

        stand_pose = sample_stand_pose()
        # while np.sqrt(np.sum((phone_pose.p[:2] - stand_pose.p[:2]) ** 2)) < 0.15:
        #     stand_pose = sample_stand_pose()
        
        max_trials = 100
        trials = 0
        
        while np.sqrt(np.sum((phone_pose.p[:2] - stand_pose.p[:2]) ** 2)) < 0.15 and trials < max_trials:
            stand_pose = sample_stand_pose()
            trials += 1
        
        if np.sqrt(np.sum((phone_pose.p[:2] - stand_pose.p[:2]) ** 2)) < 0.15:
            raise RuntimeError("Failed to sample a valid stand_pose within 100 tries.")

        self.stand_id = np.random.choice([1, 2], 1)[0]
        self.stand = create_actor(
            scene=self,
            pose=stand_pose,
            modelname="078_phonestand",
            convex=True,
            model_id=self.stand_id,
            is_static=True,
        )

        self.add_prohibit_area(self.phone, padding=0.01)
        self.add_prohibit_area(self.stand, padding=0.01)

        # =========================
        # 6. stapler
        # =========================
        stapler_pose = rand_pose(
            xlim=[0.0, 0.2],
            # ylim=[-0.1, 0.05],
            ylim=[-0.2, -0.05],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=True,
            rotate_lim=[0, np.pi, 0],
        )

        self.stapler_id = np.random.choice([0, 1, 2, 3, 4, 5, 6], 1)[0]
        self.stapler = create_actor(
            self,
            pose=stapler_pose,
            modelname="048_stapler",
            convex=True,
            model_id=self.stapler_id,
            is_static=True,
        )
        self.add_prohibit_area(self.stapler, padding=0.01)

        # =========================
        # 7. 任务进度初始化
        # =========================
        self.stage_sum = 5
        self.stage = 0
        self.has_left_press_area = True
        self.task_success = [0, 0, 0, 0, 0]


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
        eps_xy = np.array([0.03, 0.03])
        if len(positions) == 0:
            return True

        for position in positions:
            if (
                np.all(np.abs(position[:2] - stapler_pose[:2]) < eps_xy)
                and abs(position[2] - stapler_pose[2]) < 0.03
            ):
                return False

        return True

    def update_progress(self):
        # =========================
        # 1. phone -> stand
        # =========================
        phone_func_pose = np.array(self.phone.get_functional_point(0))
        stand_func_pose = np.array(self.stand.get_functional_point(0))
        eps = np.array([0.045, 0.04, 0.04])

        phone_done = int(np.all(np.abs(phone_func_pose - stand_func_pose)[:3] < eps))

        # =========================
        # 2. stapler 三次点击进度更新
        # =========================
        if self.stage < 3:
            if self._has_left_press_area():
                self.has_left_press_area = True

            if self.has_left_press_area and self._is_press_success():
                self.stage += 1
                self.has_left_press_area = False

        # =========================
        # 3. dump garbage
        # =========================
        dump_done = 1
        for i in range(self.garbage_num):
            pose = self.sphere_lst[i].get_pose().p
            if not (0.13 <= pose[2] <= 0.25):
                dump_done = 0
                break

        # =========================
        # 4. 统一记录 task_success
        # phone, press1, press2, press3, dump
        # =========================
        self.task_success[0] = phone_done
        self.task_success[1] = int(self.stage >= 1)
        self.task_success[2] = int(self.stage >= 2)
        self.task_success[3] = int(self.stage >= 3)
        self.task_success[4] = dump_done


    def play_once(self):
        # 1. place phone on stand
        arm_tag = ArmTag("left" if self.phone.get_pose().p[0] < 0 else "right")

        self.move(
            self.grasp_actor(self.phone, arm_tag=arm_tag, pre_grasp_dis=0.08)
        )

        stand_func_pose = self.stand.get_functional_point(0)
        self.move(
            self.place_actor(
                self.phone,
                arm_tag=arm_tag,
                target_pose=stand_func_pose,
                functional_point_id=0,
                dis=0,
                constrain="align",
            )
        )

        # 2. click stapler three times
        stapler_arm = ArmTag("left" if self.stapler.get_pose().p[0] < 0 else "right")

        stapler_hover_action = lambda: self.grasp_actor(
            self.stapler,
            arm_tag=stapler_arm,
            pre_grasp_dis=0.1,
            grasp_dis=0.1,
            contact_point_id=2,
        )

        stapler_press_action = lambda: self.grasp_actor(
            self.stapler,
            arm_tag=stapler_arm,
            pre_grasp_dis=0.02,
            grasp_dis=0.02,
            contact_point_id=2,
        )

        for i in range(3):
            if i == 0 and stapler_arm != arm_tag:
                self.move(
                    stapler_hover_action(),
                    self.back_to_origin(arm_tag=arm_tag),
                )
            else:
                self.move(stapler_hover_action())

            self.update_progress()

            self.move(self.close_gripper(arm_tag=stapler_arm))
            self.move(stapler_press_action())
            self.update_progress()

        # 3. dump garbage
        place_deskbin_arm_tag = ArmTag("left")

        deskbin_grasp_action = self.grasp_actor(
            self.deskbin,
            arm_tag=place_deskbin_arm_tag,
            pre_grasp_dis=0.08,
            contact_point_id=1,
        )

        if place_deskbin_arm_tag == stapler_arm:
            self.move(deskbin_grasp_action)
        else:
            self.move(
                deskbin_grasp_action,
                self.back_to_origin(arm_tag=stapler_arm),
            )

        self.move(
            self.move_by_displacement(
                arm_tag=place_deskbin_arm_tag,
                z=0.08,
                move_axis="arm",
            )
        )

        for _ in range(3):
            self.move(self.pour_actions)

        self.delay(6)
        self.info["info"] = {
            "{A}": f"077_phone/base{self.phone_id}",
            "{B}": f"078_phonestand/base{self.stand_id}",
            "{C}": f"048_stapler/base{self.stapler_id}",
            "{D}": f"063_tabletrashbin/base{self.deskbin_id}",
            "{E}": "011_dustbin/base0",
            "{a}": str(arm_tag),
            "{c}": str(stapler_arm),
            "{d}": "left",
        }

        return self.info
 
    def check_success(self):
        self.update_progress()
        return self.task_success == [1, 1, 1, 1, 1]