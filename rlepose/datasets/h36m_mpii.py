import random

import torch
import torch.utils.data as data
from rlepose.models.builder import DATASET
from easydict import EasyDict

from .h36m import H36m
from .mpii import Mpii

s_mpii_2_hm36_jt = [6, 2, 1, 0, 3, 4, 5, -1, 8,
                    -1, 9, 13, 14, 15, 12, 11, 10, 7]
s_36_jt_num = 18


@DATASET.register_module
class H36mMpii(data.Dataset):
    CLASSES = ['person']
    EVAL_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    num_joints = 18
    num_bones = 17
    bbox_3d_shape = (2000, 2000, 2000)
    joints_name = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck',
                   'Nose', 'Head', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'Thorax')
    action_name = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Posing', 'Purchases',
                   'Sitting', 'SittingDown', 'Smoking', 'Photo', 'Waiting', 'Walking', 'WalkDog', 'WalkTogether']
    skeleton = ((1, 0), (2, 0),
                (3, 1), (4, 2), (5, 3), (6, 4),
                (7, 0),
                (8, 0), (9, 0),
                (10, 8), (11, 9), (12, 10), (13, 11)
                )
    data_domain = set([
        'type',
        'target_uvd',
        'target_uvd_weight'
    ])

    def __init__(self,
                 train=True,
                 skip_empty=True,
                 dpg=False,
                 lazy_import=False,
                 **cfg):
        self._train = train
        self._preset_cfg = cfg['PRESET']

        cfg = EasyDict(cfg)
        if train:
            self.db0 = H36m(
                cfg=cfg,
                ann_file=cfg.SET_LIST[0].TRAIN_SET,
                train=train)
            self.db1 = Mpii(
                cfg=cfg,
                ann_file=f'{cfg.SET_LIST[1].TRAIN_SET}.json',
                train=True)

            self._subsets = [self.db0, self.db1]
        else:
            self.db0 = H36m(
                cfg=cfg,
                ann_file=cfg.TEST_SET,
                train=train)

            self._subsets = [self.db0]

        self._subset_size = [len(item) for item in self._subsets]
        self.cumulative_sizes = self.cumsum(self._subset_size)
        self._db0_size = len(self.db0)

        if train:
            self._db1_size = len(self.db1)
            self.tot_size = 2 * self._db0_size
        else:
            self.tot_size = self._db0_size
        self.joint_pairs = self.db0.joint_pairs
        self.evaluate = self.db0.evaluate

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            r.append(e + s)
            s += e
        return r

    def __len__(self):
        return self.tot_size

    def __getitem__(self, idx):
        assert idx >= 0
        if idx < self._db0_size:
            dataset_idx = 0
            sample_idx = idx
        else:
            assert self._train
            dataset_idx = 1

            _rand = random.uniform(0, 1)
            sample_idx = int(_rand * self._db1_size)

        sample = self._subsets[dataset_idx][sample_idx]
        img, target, img_id, bbox = sample

        if dataset_idx == 1:
            # Mpii
            target_uvd_origin = target.pop('target_uvd')
            target_uvd_weight_origin = target.pop('target_uvd_weight')

            # --- 💡 로직 분기 시작 💡 ---

            # 1. RLELoss (좌표) 데이터인 경우 (dim() == 1)
            #    (이것이 rlepose의 원래 로직입니다)
            if target_uvd_origin.dim() == 1:
                target_uvd = torch.zeros(self.num_joints, 3)
                target_uvd_weight = torch.zeros(self.num_joints, 3)

                # 원래 assert 로직 유지 (16 * 3 = 48)
                assert target_uvd_origin.shape[0] == 16 * 3, f"RLE coord shape error: {target_uvd_origin.shape}"
                target_uvd_origin = target_uvd_origin.reshape(16, 3)
                target_uvd_weight_origin = target_uvd_weight_origin.reshape(16, 3)
                
                for i in range(s_36_jt_num):
                    id1 = i
                    id2 = s_mpii_2_hm36_jt[i]
                    if id2 >= 0:
                        target_uvd[id1, :2] = target_uvd_origin[id2, :2].clone()
                        target_uvd_weight[id1, :2] = target_uvd_weight_origin[id2, :2].clone()

                target['target_uvd'] = target_uvd.reshape(-1)
                target['target_uvd_weight'] = target_uvd_weight.reshape(-1)
            
            # 2. MSELoss (히트맵) 데이터인 경우 (dim() == 3)
            #    (이것이 ConvNextPose를 위한 새 로직입니다)
            elif target_uvd_origin.dim() == 3:
                num_mpii_joints, h, w = target_uvd_origin.shape
                # 히트맵 assert 로직 (16, 32, 32)
                assert num_mpii_joints == 16, f"Heatmap shape error: {target_uvd_origin.shape}"
                
                target_uvd = torch.zeros(self.num_joints, h, w)           # (18, 32, 32)
                target_uvd_weight = torch.zeros(self.num_joints, 1, 1)    # (18, 1, 1)

                for i in range(s_36_jt_num): # 0~17 (H36M 관절 인덱스)
                    id_h36m = i
                    id_mpii = s_mpii_2_hm36_jt[i] # H36M 인덱스에 매핑되는 MPII 인덱스
                    
                    if id_mpii >= 0:
                        # 매핑되는 관절이 있으면 해당 히트맵 채널을 그대로 복사
                        target_uvd[id_h36m, :, :] = target_uvd_origin[id_mpii, :, :].clone()
                        # 가중치도 복사
                        target_uvd_weight[id_h36m] = target_uvd_weight_origin[id_mpii].clone()
                
                target['target_uvd'] = target_uvd
                target['target_uvd_weight'] = target_uvd_weight
            
            # 3. 예외 처리
            else:
                raise ValueError(f"Unexpected target_uvd shape from MPII: {target_uvd_origin.shape}")

        # --- 로직 분기 끝 ---

        assert set(target.keys()).issubset(self.data_domain), (set(target.keys()), self.data_domain)
        target.pop('type')

        return img, target, img_id, bbox

        # if dataset_idx == 1:
        #     # Mpii
        #     target_uvd_origin = target.pop('target_uvd')
        #     target_uvd_weight_origin = target.pop('target_uvd_weight')

        #     target_uvd = torch.zeros(self.num_joints, 3)
        #     target_uvd_weight = torch.zeros(self.num_joints, 3)

        #     assert target_uvd_origin.dim() == 1 and target_uvd_origin.shape[0] == 16 * 3, target_uvd_origin.shape
        #     target_uvd_origin = target_uvd_origin.reshape(16, 3)
        #     target_uvd_weight_origin = target_uvd_weight_origin.reshape(16, 3)
        #     for i in range(s_36_jt_num):
        #         id1 = i
        #         id2 = s_mpii_2_hm36_jt[i]
        #         if id2 >= 0:
        #             target_uvd[id1, :2] = target_uvd_origin[id2, :2].clone()
        #             target_uvd_weight[id1, :2] = target_uvd_weight_origin[id2, :2].clone()

        #     target['target_uvd'] = target_uvd.reshape(-1)
        #     target['target_uvd_weight'] = target_uvd_weight.reshape(-1)

        # assert set(target.keys()).issubset(self.data_domain), (set(target.keys()), self.data_domain)
        # target.pop('type')

        # return img, target, img_id, bbox
