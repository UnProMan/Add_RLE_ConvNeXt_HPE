import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from .transforms import get_max_pred_batch

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class NullWriter(object):
    def write(self, arg):
        pass


class DataLogger(object):
    """Average data logger."""
    def __init__(self):
        self.clear()

    def clear(self):
        self.value = 0
        self.sum = 0
        self.cnt = 0
        self.avg = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.cnt += n
        self._cal_avg()

    def _cal_avg(self):
        self.avg = self.sum / self.cnt


def calc_iou(pred, target):
    """Calculate mask iou"""
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().data.numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().data.numpy()

    pred = pred >= 0.5
    target = target >= 0.5

    intersect = (pred == target) * pred * target
    union = np.maximum(pred, target)

    if pred.ndim == 2:
        iou = np.sum(intersect) / np.sum(union)
    elif pred.ndim == 3 or pred.ndim == 4:
        n_samples = pred.shape[0]
        intersect = intersect.reshape(n_samples, -1)
        union = union.reshape(n_samples, -1)

        iou = np.mean(np.sum(intersect, axis=1) / np.sum(union, axis=1))

    return iou


def mask_cross_entropy(pred, target):
    return F.binary_cross_entropy_with_logits(
        pred, target, reduction='mean')[None]


def evaluate_mAP(res_file, ann_type='bbox', ann_file='person_keypoints_val2017.json', silence=True):
    """Evaluate mAP result for coco dataset.

    Parameters
    ----------
    res_file: str
        Path to result json file.
    ann_type: str
        annotation type, including: `bbox`, `segm`, `keypoints`.
    ann_file: str
        Path to groundtruth file.
    silence: bool
        True: disable running log.

    """
    class NullWriter(object):
        def write(self, arg):
            pass

    ann_file = os.path.join('./data/coco/annotations/', ann_file)

    if silence:
        nullwrite = NullWriter()
        oldstdout = sys.stdout
        sys.stdout = nullwrite  # disable output

    cocoGt = COCO(ann_file)
    cocoDt = cocoGt.loadRes(res_file)

    cocoEval = COCOeval(cocoGt, cocoDt, ann_type)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    if silence:
        sys.stdout = oldstdout  # enable output

    stats_names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)',
                   'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']
    info_str = {}
    for ind, name in enumerate(stats_names):
        info_str[name] = cocoEval.stats[ind]

    return info_str


def calc_accuracy(output, target):
    """Calculate heatmap accuracy."""
    # 3D 예측 히트맵 (e.g., [B, 576, H, W])
    preds_3d_flat = output.heatmap

    # 2D 정답 히트맵 (이전 수정)
    if 'target_hm' in target:
        labels = target['target_hm']
        labels_mask = target['target_hm_weight']
    else:
        labels = target['target_uvd']
        labels_mask = target['target_uvd_weight']

    # --- 💡 START: 3D -> 2D 변환 로직 (criterion.py에서 복사) 💡 ---
    
    # 1. 정답(labels)에서 관절 수(num_joints)를 가져옵니다.
    num_joints = labels.shape[1] # e.g., 18

    # 2. 채널이 나누어지는지 확인
    if preds_3d_flat.shape[1] % num_joints != 0:
        raise ValueError(f"Prediction channels ({preds_3d_flat.shape[1]})"
                         f" not divisible by GT channels ({num_joints})")

    # 3. 깊이(Z) 차원(depth_dim)을 계산합니다.
    depth_dim = preds_3d_flat.shape[1] // num_joints # e.g., 576 // 18 = 32

    # 4. 평탄화된 3D 예측을 [B, J, D, H, W]로 재구성합니다.
    pred_hm_3d = preds_3d_flat.reshape(-1, 
                                         num_joints, 
                                         depth_dim, 
                                         preds_3d_flat.shape[2], 
                                         preds_3d_flat.shape[3])

    # 5. 깊이(Z) 차원(dim=2)을 따라 합산(sum)하여 2D 히트맵으로 만듭니다.
    preds = torch.sum(pred_hm_3d, dim=2)
    # preds shape: [B, 18, H, W]

    # 6. (중요) 예측과 정답의 해상도가 다를 경우 리사이즈 (criterion.py에서 복사)
    if preds.shape[2:] != labels.shape[2:]:
        preds = F.interpolate(
            preds, 
            size=labels.shape[2:], # (e.g., 64x64 or 32x32)
            mode='bilinear', 
            align_corners=False
        )
    # --- 💡 END: 3D -> 2D 변환 로직 💡 ---

    # 이제 이 코드는 정상 작동합니다: [B, 18, H, W] * [B, 18, 1, 1]
    preds = preds * labels_mask
    labels = labels * labels_mask

    preds = preds.cpu().data.numpy()
    labels = labels.cpu().data.numpy()

    num_joints = preds.shape[1]

    norm = 1.0
    hm_h = preds.shape[2]
    hm_w = preds.shape[3]

    preds, _ = get_max_pred_batch(preds)
    labels, _ = get_max_pred_batch(labels)
    norm = np.ones((preds.shape[0], 2)) * np.array([hm_w, hm_h]) / 10

    dists = calc_dist(preds, labels, norm)

    acc = 0
    sum_acc = 0
    cnt = 0
    for i in range(num_joints):
        acc = dist_acc(dists[i])
        if acc >= 0:
            sum_acc += acc
            cnt += 1

    if cnt > 0:
        return sum_acc / cnt
    else:
        return 0


def calc_coord_accuracy(output, target, hm_shape, output_3d=False, num_joints=None, other_joints=None, root_idx=None):
    """Calculate integral coordinates accuracy."""
    coords = output.pred_jts.detach().cpu().numpy()
    coords = coords.astype(float)

    if output_3d:
        labels = target['target_uvd']
        label_masks = target['target_uvd_weight']
    else:
        labels = target['target_uv']
        label_masks = target['target_uv_weight']

    if num_joints is not None:
        if other_joints is not None:
            coords = coords.reshape(coords.shape[0], num_joints + other_joints, -1)
            labels = labels.reshape(labels.shape[0], num_joints + other_joints, -1)
            label_masks = label_masks.reshape(label_masks.shape[0], num_joints + other_joints, -1)
            coords = coords[:, :num_joints, :3].reshape(coords.shape[0], -1)
            labels = labels[:, :num_joints, :3].reshape(coords.shape[0], -1)
            label_masks = label_masks[:, :num_joints, :3].reshape(coords.shape[0], -1)
        else:
            coords = coords.reshape(coords.shape[0], num_joints, -1)
            labels = labels.reshape(labels.shape[0], num_joints, -1)
            label_masks = label_masks.reshape(label_masks.shape[0], num_joints, -1)
            coords = coords[:, :, :3].reshape(coords.shape[0], -1)
            labels = labels[:, :, :3].reshape(coords.shape[0], -1)
            label_masks = label_masks[:, :, :3].reshape(coords.shape[0], -1)

    if output_3d:
        hm_width, hm_height, hm_depth = hm_shape
        coords = coords.reshape((coords.shape[0], -1, 3))
    else:
        hm_width, hm_height = hm_shape[:2]
        coords = coords.reshape((coords.shape[0], -1, 2))
    coords[:, :, 0] = (coords[:, :, 0] + 0.5) * hm_width
    coords[:, :, 1] = (coords[:, :, 1] + 0.5) * hm_height

    if output_3d:
        labels = labels.cpu().data.numpy().reshape(coords.shape[0], -1, 3)
        label_masks = label_masks.cpu().data.numpy().reshape(coords.shape[0], -1, 3)

        labels[:, :, 0] = (labels[:, :, 0] + 0.5) * hm_width
        labels[:, :, 1] = (labels[:, :, 1] + 0.5) * hm_height
        if hm_depth is not None:
            labels[:, :, 2] = (labels[:, :, 2] + 0.5) * hm_depth

        coords[:, :, 2] = (coords[:, :, 2] + 0.5) * hm_depth if hm_depth is not None else coords[:, :, 2]

        if root_idx is not None:
            labels = labels - labels[:, root_idx, :][:, None, :]
            coords = coords - coords[:, root_idx, :][:, None, :]

    else:
        labels = labels.cpu().data.numpy().reshape(coords.shape[0], -1, 2)
        label_masks = label_masks.cpu().data.numpy().reshape(coords.shape[0], -1, 2)

        labels[:, :, 0] = (labels[:, :, 0] + 0.5) * hm_width
        labels[:, :, 1] = (labels[:, :, 1] + 0.5) * hm_height

    num_joints = coords.shape[1]

    coords = coords * label_masks
    labels = labels * label_masks

    if output_3d:
        norm = np.ones((coords.shape[0], 3)) * np.array([hm_width, hm_height, hm_depth if hm_depth is not None else 1]) / 10
    else:
        norm = np.ones((coords.shape[0], 2)) * np.array([hm_width, hm_height]) / 10

    dists = calc_dist(coords, labels, norm)

    acc = 0
    sum_acc = 0
    cnt = 0
    for i in range(num_joints):
        acc = dist_acc(dists[i])
        if acc >= 0:
            sum_acc += acc
            cnt += 1

    if cnt > 0:
        return sum_acc / cnt
    else:
        return 0


def calc_dist(preds, target, normalize):
    """Calculate normalized distances"""
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))

    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
            else:
                dists[c, n] = -1

    return dists


def dist_acc(dists, thr=0.5):
    """Calculate accuracy with given input distance."""
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1