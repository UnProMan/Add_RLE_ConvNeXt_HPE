import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Any, Callable, List, Optional, Type, Union
from torch import Tensor
from timm.models.layers import trunc_normal_, DropPath
import torch.quantization
from torch.quantization import QuantStub, DeQuantStub
import math

from .builder import SPPE
from collections import namedtuple
from easydict import EasyDict as Output

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        drop_path: float = 0,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.drop_path(out) ##drop_path

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        drop_path: float = 0,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.drop_path(out) ##drop_path

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNetBackbone(nn.Module):

    def __init__(self, resnet_type, drop_path_rate=0.):	
        resnet_spec = {18: (BasicBlock, [2, 2, 2, 2], [64, 64, 128, 256, 512], 'resnet18'),
		               34: (BasicBlock, [3, 4, 6, 3], [64, 64, 128, 256, 512], 'resnet34'),
		               50: (Bottleneck, [3, 4, 6, 3], [64, 256, 512, 1024, 2048], 'resnet50'),
		               101: (Bottleneck, [3, 4, 23, 3], [64, 256, 512, 1024, 2048], 'resnet101'),
		               152: (Bottleneck, [3, 8, 36, 3], [64, 256, 512, 1024, 2048], 'resnet152')}
        block, layers, channels, name = resnet_spec[resnet_type]
        
        self.name = name
        self.inplanes = 64
        super(ResNetBackbone, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(layers))] 

        self.layer1 = self._make_layer(block, 64, layers[0], dp_range=[0, dp_rates[sum(layers[0:1])-1]])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dp_range=[dp_rates[sum(layers[0:1])], dp_rates[sum(layers[0:2])-1]])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dp_range=[dp_rates[sum(layers[0:2])], dp_rates[sum(layers[0:3])-1]])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dp_range=[dp_rates[sum(layers[0:3])], dp_rates[sum(layers[0:4])-1]])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, mean=0, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dp_range=[0,0]):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        dp_rates=[x.item() for x in torch.linspace(dp_range[0], dp_range[1], blocks)] 

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, drop_path=dp_rates[0]))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, drop_path=dp_rates[i]))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def init_weights(self):
        org_resnet = torch.utils.model_zoo.load_url(model_urls[self.name])
        # drop orginal resnet fc layer, add 'None' in case of no fc layer, that will raise error
        org_resnet.pop('fc.weight', None)
        org_resnet.pop('fc.bias', None)
        self.load_state_dict(org_resnet)
        print("Initialize resnet from model zoo")

class SimCCHead(nn.Module):
    
    def __init__(self, embedding_size, joint_num, channel_per_joint, feat_height, feat_width, output_shape, depth_dim, drop_rate=0):
        self.inplanes = embedding_size
        self.outplanes = joint_num * channel_per_joint
        self.joint_num = joint_num
        self.channel_joint = channel_per_joint
        self.fh = feat_height
        self.fw = feat_width
        self.out_shape = output_shape
        self.depth_dim = depth_dim
        self.drop_rate = drop_rate

        super(SimCCHead, self).__init__()
        
        self.conv = nn.Conv2d(embedding_size, self.outplanes, kernel_size=1, stride=1, padding=0)        
        self.mlp_x = nn.Linear(self.channel_joint * self.fh * self.fw, self.out_shape[1])
        self.mlp_y = nn.Linear(self.channel_joint * self.fh * self.fw, self.out_shape[0])
        self.mlp_z = nn.Linear(self.channel_joint * self.fh * self.fw, self.depth_dim)
        self.dropout = nn.Dropout(self.drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x):
        x = self.conv(x)       
        x = x.reshape(-1, self.joint_num, self.channel_joint * self.fh * self.fw) #[b, joint_num, ch*fh*fw]
        x = self.dropout(x)
        
        # pred_xyz : [b, joint_num, 3, tokens(out_shape)]
        pred_x = self.mlp_x(x)
        pred_y = self.mlp_y(x)
        pred_z = self.mlp_z(x)
            
        # [b, joint_num, 3, tokens]
        return torch.stack([pred_x, pred_y, pred_z], dim=2)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                #nn.init.xavier_uniform_(m.weight, gain=0.01)
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

def _get_coords_and_maxvals_3d(heatmaps, joint_num, depth_dim, output_shape):
    """
        SimCCHead의 output을 기반으로 좌표와 maxvals 계산
        heatmaps: [B, J, 3, tokens]
        output:
            coord_out: [B, J, 3] (x, y, z 좌표)
            maxvals: [B, J, 1] (신뢰도 점수)
    """
    # heatmaps: [B, J, 3, tokens]
    # Softmax 적용
    heatmaps_softmax = F.softmax(heatmaps, dim=3)  # [B, J, 3, tokens]

    # 각 축별로 soft argmax 계산
    device = heatmaps.device

    # X 축 (dim=3, tokens = output_shape[1])
    accu_x = heatmaps_softmax[:, :, 0, :] * torch.arange(0, output_shape[1], device=device).float()
    accu_x = accu_x.sum(dim=2, keepdim=True)

    # Y 축 (tokens = output_shape[0])
    accu_y = heatmaps_softmax[:, :, 1, :] * torch.arange(0, output_shape[0], device=device).float()
    accu_y = accu_y.sum(dim=2, keepdim=True)

    # Z 축 (tokens = depth_dim)
    accu_z = heatmaps_softmax[:, :, 2, :] * torch.arange(0, depth_dim, device=device).float()
    accu_z = accu_z.sum(dim=2, keepdim=True)

    coord_out = torch.cat((accu_x, accu_y, accu_z), dim=2)

    # maxvals: 각 축의 최대 softmax 값의 평균 또는 최대
    maxvals, _ = heatmaps_softmax.max(dim=3)
    maxvals = maxvals.mean(dim=2, keepdim=True)  # [B, J, 1]

    return coord_out, maxvals

def _get_coords_and_maxvals_2d(heatmaps, joint_num, output_shape):
    """
    heatmaps: [B, J, 2, tokens] (X and Y only)
    output:
        coord_out: [B, J, 2] (x, y 좌표)
        maxvals: [B, J, 1] (신뢰도 점수)
    """
    # heatmaps: [B, J, 2, tokens]
    # Softmax 적용
    heatmaps_softmax = F.softmax(heatmaps, dim=3)  # [B, J, 2, tokens]

    # 각 축별로 soft argmax 계산
    device = heatmaps.device

    # X 축 (dim=3, tokens = output_shape[1])
    accu_x = heatmaps_softmax[:, :, 0, :] * torch.arange(0, output_shape[1], device=device).float()
    accu_x = accu_x.sum(dim=2, keepdim=True)

    # Y 축 (tokens = output_shape[0])
    accu_y = heatmaps_softmax[:, :, 1, :] * torch.arange(0, output_shape[0], device=device).float()
    accu_y = accu_y.sum(dim=2, keepdim=True)

    coord_out = torch.cat((accu_x, accu_y), dim=2)

    # maxvals: 각 축의 최대 softmax 값의 평균 또는 최대
    maxvals, _ = heatmaps_softmax.max(dim=3)
    maxvals = maxvals.mean(dim=2, keepdim=True)  # [B, J, 1]

    return coord_out, maxvals

@SPPE.register_module
class ResNetPoseSimCC(nn.Module):

    def __init__(self, **cfg):
        super(ResNetPoseSimCC, self).__init__()

        preset_cfg = cfg['PRESET']
        self.output_shape = preset_cfg['HEATMAP_SIZE']
        self.joint_num = preset_cfg['NUM_JOINTS']
        self.input_size = preset_cfg['IMAGE_SIZE']
        self.depth_dim = cfg.get('depth_dim', 0)
        self.is_3d = preset_cfg['OUT_3D']

        self.backbone = ResNetBackbone(resnet_type=cfg.get('resnet_type', 50), drop_path_rate=cfg.get('drop_rate', 0.))
        self.head = SimCCHead(
            embedding_size=cfg.get('embedding_size', 2048),
            joint_num=self.joint_num,
            channel_per_joint=cfg.get('channel_per_joint', 36),
            feat_height=cfg.get('feat_height', 8),
            feat_width=cfg.get('feat_width', 8),
            output_shape=self.output_shape,
            depth_dim=self.depth_dim,
            drop_rate=cfg.get('drop_rate', 0)
        )

        self.quant_flag = cfg.get('quant_flag', False)
        if self.quant_flag:
            pass

    def forward(self, x, labels=None):
        fm = self.backbone(x)
        raw_heatmap = self.head(fm)  # [B, J, 3, tokens] or [B, J, 2, tokens] if not 3D

        if self.is_3d:
            pred_jts, maxvals = _get_coords_and_maxvals_3d(
                raw_heatmap.detach(),
                self.joint_num,
                self.depth_dim,
                self.output_shape
            )
        else:
            # Assume raw_heatmap is [B, J, 2, tokens] for 2D
            pred_jts, maxvals = _get_coords_and_maxvals_2d(
                raw_heatmap.detach(),
                self.joint_num,
                self.output_shape
            )

        output = Output(
            heatmap=raw_heatmap,
            pred_jts=pred_jts,
            maxvals=maxvals
        )
        return output

    def init_weights(self):
        self.backbone.init_weights()
        self.head.init_weights()

    def _initialize(self):
        self.init_weights()