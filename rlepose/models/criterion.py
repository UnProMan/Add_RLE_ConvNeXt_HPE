import math

import torch
import torch.nn as nn

from .builder import LOSS


@LOSS.register_module
class MSELoss(nn.Module):
    ''' MSE Loss
    '''
    def __init__(self):
        super(MSELoss, self).__init__()
        self.criterion = nn.MSELoss()

    # def forward(self, output, labels):
    #     pred_hm = output['heatmap']
    #     gt_hm = labels['target_hm']
    #     gt_hm_weight = labels['target_hm_weight']
    #     loss = 0.5 * self.criterion(pred_hm.mul(gt_hm_weight), gt_hm.mul(gt_hm_weight))

    #     return loss

    def forward(self, output, labels):
        # 3D 예측 히트맵 (e.g., [B, 576, 32, 32])
        pred_hm_3d_flat = output['heatmap']
        # 2D 정답 히트맵 (e.g., [B, 18, 32, 32])
        gt_hm_2d = labels['target_uvd']
        # 2D 정답 가중치 (e.g., [B, 18, 1, 1])
        gt_hm_weight = labels['target_uvd_weight']

        # --- 3D 예측을 2D로 변환하는 로직 ---
        
        # 1. 정답(gt_hm_2d)에서 관절 수(num_joints)를 가져옵니다.
        num_joints = gt_hm_2d.shape[1] # e.g., 18

        # 2. 예측(pred_hm_3d_flat) 채널이 정답 채널로 나누어지는지 확인
        if pred_hm_3d_flat.shape[1] % num_joints != 0:
            raise ValueError(f"Prediction channels ({pred_hm_3d_flat.shape[1]})"
                             f" not divisible by GT channels ({num_joints})")

        # 3. 깊이(Z) 차원(depth_dim)을 계산합니다.
        depth_dim = pred_hm_3d_flat.shape[1] // num_joints # e.g., 576 // 18 = 32

        # 4. 평탄화된 3D 예측을 [B, J, D, H, W]로 재구성합니다.
        # e.g., [B, 576, 32, 32] -> [B, 18, 32, 32, 32]
        pred_hm_3d = pred_hm_3d_flat.reshape(-1, 
                                             num_joints, 
                                             depth_dim, 
                                             pred_hm_3d_flat.shape[2], 
                                             pred_hm_3d_flat.shape[3])

        # 5. 깊이(Z) 차원(dim=2)을 따라 합산(sum)하여 2D 히트맵으로 만듭니다.
        # (torch.max(pred_hm_3d, dim=2)[0] 를 사용해도 됩니다)
        pred_hm_2d = torch.sum(pred_hm_3d, dim=2)
        # pred_hm_2d shape: [B, 18, 32, 32]

        # 6. 이제 2D 예측과 2D 정답을 사용하여 손실을 계산합니다.
        loss = 0.5 * self.criterion(pred_hm_2d.mul(gt_hm_weight), gt_hm_2d.mul(gt_hm_weight))

        return loss


@LOSS.register_module
class RLELoss(nn.Module):
    ''' RLE Regression Loss
    '''

    def __init__(self, OUTPUT_3D=False, size_average=True):
        super(RLELoss, self).__init__()
        self.size_average = size_average
        self.amp = 1 / math.sqrt(2 * math.pi)

    def logQ(self, gt_uv, pred_jts, sigma):
        return torch.log(sigma / self.amp) + torch.abs(gt_uv - pred_jts) / (math.sqrt(2) * sigma + 1e-9)

    def forward(self, output, labels):
        nf_loss = output.nf_loss
        pred_jts = output.pred_jts
        sigma = output.sigma
        gt_uv = labels['target_uv'].reshape(pred_jts.shape)
        gt_uv_weight = labels['target_uv_weight'].reshape(pred_jts.shape)

        nf_loss = nf_loss * gt_uv_weight[:, :, :1]

        residual = True
        if residual:
            Q_logprob = self.logQ(gt_uv, pred_jts, sigma) * gt_uv_weight
            loss = nf_loss + Q_logprob

        if self.size_average and gt_uv_weight.sum() > 0:
            return loss.sum() / len(loss)
        else:
            return loss.sum()


@LOSS.register_module
class RLELoss3D(nn.Module):
    ''' RLE Regression Loss 3D
    '''

    def __init__(self, OUTPUT_3D=False, size_average=True):
        super(RLELoss3D, self).__init__()
        self.size_average = size_average
        self.amp = 1 / math.sqrt(2 * math.pi)

    def logQ(self, gt_uv, pred_jts, sigma):
        return torch.log(sigma / self.amp) + torch.abs(gt_uv - pred_jts) / (math.sqrt(2) * sigma + 1e-9)

    def forward(self, output, labels):
        nf_loss = output.nf_loss
        pred_jts = output.pred_jts
        sigma = output.sigma
        gt_uv = labels['target_uvd'].reshape(pred_jts.shape)
        gt_uv_weight = labels['target_uvd_weight'].reshape(pred_jts.shape)
        nf_loss = nf_loss * gt_uv_weight

        residual = True
        if residual:
            Q_logprob = self.logQ(gt_uv, pred_jts, sigma) * gt_uv_weight
            loss = nf_loss + Q_logprob

        if self.size_average and gt_uv_weight.sum() > 0:
            return loss.sum() / len(loss)
        else:
            return loss.sum()
