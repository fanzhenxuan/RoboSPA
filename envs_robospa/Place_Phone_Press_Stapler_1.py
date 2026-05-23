from ._base_task import Base_Task
from .utils import *
import sapien
from copy import deepcopy


class Place_Phone_Press_Stapler_1(Base_Task):

    def setup_demo(self, is_test=False, **kwargs):
        super()._init_task_env_(table_xy_bias=[0.3, 0], **kwargs)

    def load_actors(self):
        # =========================
        # 1. phone 与 stand
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
        # 2. 任务进度初始化
        # =========================
        self.stage_sum = 1
        self.task_success = [0]

    def update_progress(self):
        # =========================
        # 1. phone -> stand
        # =========================
        phone_func_pose = np.array(self.phone.get_functional_point(0))
        stand_func_pose = np.array(self.stand.get_functional_point(0))
        eps = np.array([0.045, 0.04, 0.04])

        phone_done = int(np.all(np.abs(phone_func_pose - stand_func_pose)[:3] < eps))

        # =========================
        # 2. 统一记录 task_success
        # phone
        # =========================
        self.task_success[0] = phone_done

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

        self.info["info"] = {
            "{A}": f"077_phone/base{self.phone_id}",
            "{B}": f"078_phonestand/base{self.stand_id}",
            "{a}": str(arm_tag),
        }

        return self.info

    def check_success(self):
        self.update_progress()
        return self.task_success == [1]