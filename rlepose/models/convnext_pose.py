import torch
import torch.nn as nn
from .layers.ConvNeXt import Block, DeConv
from timm.models.layers import trunc_normal_
from .builder import SPPE
import torch.nn.functional as F
from collections import namedtuple
from easydict import EasyDict as Output

def _get_coords_and_maxvals_3d(heatmaps, joint_num, depth_dim, output_shape):
    """
        mobilehpe의 soft_argmax 로직 기반
        heatmaps: [B, J*D, H, W]
        ouput:
            coord_out: [B, J, 3] (x, y, z 좌표)
            maxvals: [B, J, 1] (신뢰도 점수)
    """

    # Reshape 및 Softmax 적용
    heatmaps_flat = heatmaps.reshape((-1, joint_num, depth_dim * output_shape[0] * output_shape[1]))
    heatmaps_softmax = F.softmax(heatmaps_flat, 2)
    heatmaps = heatmaps_softmax.reshape((-1, joint_num, depth_dim, output_shape[0], output_shape[1]))

    accu_x = heatmaps.sum(dim=(2, 3))
    accu_y = heatmaps.sum(dim=(2, 4))
    accu_z = heatmaps.sum(dim=(3, 4))

    device = heatmaps.device

    accu_x = accu_x * torch.arange(1, output_shape[1] + 1, device=device).float()
    accu_y = accu_y * torch.arange(1, output_shape[0] + 1, device=device).float()
    accu_z = accu_z * torch.arange(1, depth_dim + 1, device=device).float()

    accu_x = accu_x.sum(dim=2, keepdim=True) - 1
    accu_y = accu_y.sum(dim=2, keepdim=True) - 1
    accu_z = accu_z.sum(dim=2, keepdim=True) - 1

    coord_out = torch.cat((accu_x, accu_y, accu_z), dim=2)

    maxvals, _ = heatmaps_softmax.max(dim=2, keepdim=True)

    return coord_out, maxvals

def _get_coords_and_maxvals_2d(heatmaps, joint_num, output_shape):
    """
    heatmaps: [B, J, H, W]
    output:
        coord_out: [B, J, 2] (x, y 좌표)
        maxvals: [B, J, 1] (신뢰도 점수)
    """
    heatmaps_flat = heatmaps.reshape((-1, joint_num, output_shape[0] * output_shape[1]))
    heatmaps_softmax = F.softmax(heatmaps_flat, 2)
    heatmaps = heatmaps_softmax.reshape((-1, joint_num, output_shape[0], output_shape[1]))

    accu_x = heatmaps.sum(dim=2)
    accu_y = heatmaps.sum(dim=3)
    
    device = heatmaps.device

    accu_x = accu_x * torch.arange(1, output_shape[1] + 1, device=device).float()
    accu_y = accu_y * torch.arange(1, output_shape[0] + 1, device=device).float()

    accu_x = accu_x.sum(dim=2, keepdim=True) - 1
    accu_y = accu_y.sum(dim=2, keepdim=True) - 1

    coord_out = torch.cat((accu_x, accu_y), dim=2) 
    maxvals, _ = heatmaps_softmax.max(dim=2, keepdim=True)

    return coord_out, maxvals

class ConvNextBackbone(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000, convnext_type=50,
                 depths=[3, 3, 9, 3], dim=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.):
        super().__init__()

        convnext_spec = {
            18: (Block, [2, 2, 6, 2], [40, 80, 160, 320], 'convnext18'),
            34: (Block, [3, 3, 9, 3], [48, 96, 192, 384], 'convnext34'),
            50: (Block, [3, 3, 9, 3], [96, 192, 384, 768], 'convnext50')
        }

        block, depths, dims, name = convnext_spec[convnext_type]

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=dims[0], kernel_size=4, stride=4),
            nn.BatchNorm2d(dims[0])
        )

        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.BatchNorm2d(dims[i]),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2)
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        for i in range(4):
            stage = nn.Sequential(
                *[block(dim=dims[i], drop_path=dp_rates[cur + j],
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.BatchNorm2d(dims[-1])

    def init_weights(self):
        for i in [self.downsample_layers, self.stages, self.norm]:
            for name, m in i.named_modules():
                if isinstance(m, nn.Conv2d):
                    trunc_normal_(m.weight, std=.02)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_stage = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i < 3: x_stage.append(x)

        return self.norm(x)
    

class ConvNextHead(nn.Module):
    def __init__(self, embedding_size, depth_dim, joint_num, head_channels = 128, use_3ups=False):
        self.inplanes = embedding_size
        self.channels = head_channels
        super(ConvNextHead, self).__init__()

        self.deconv_layers_1 = DeConv(inplances=self.inplanes, planes=self.channels, kernel_size=3)
        self.deconv_layers_2 = DeConv(inplances=self.channels, planes=self.channels, kernel_size=3)
        self.deconv_layers_3 = DeConv(inplances=self.channels, planes=self.channels, kernel_size=3, up = use_3ups)

        self.final_layer = nn.Conv2d(
            in_channels=self.channels,
            out_channels=depth_dim,  # <--- (joint_num * depth_dim 대신)
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, x):
        x = self.deconv_layers_1(x)
        x = self.deconv_layers_2(x)
        x = self.deconv_layers_3(x)
        x = self.final_layer(x)

        return x
    
    def init_weights(self):
        for i in [self.deconv_layers_1, self.deconv_layers_2, self.deconv_layers_3]:
            for name, m in i.named_modules():
                if isinstance(m, nn.Conv2d):
                    trunc_normal_(m.weight, std=.02)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        
        for j in [self.final_layer]:
            for m in j.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if hasattr(m, 'bias'):
                        if m.bias is not None: nn.init.constant_(m.bias, 0)



@SPPE.register_module
class ConvNextPose(nn.Module):
    def __init__(self, **cfg):
        super(ConvNextPose, self).__init__()

        preset_cfg = cfg['PRESET']
        self.output_shape = preset_cfg['HEATMAP_SIZE']
        self.joint_num = preset_cfg['NUM_JOINTS']
        self.input_size = preset_cfg['IMAGE_SIZE']
        self.depth_dim = cfg['depth_dim']
        self.is_3d = preset_cfg['OUT_3D']

        self.use_3ups = True

        if self.is_3d:
            self.depth_dim = cfg['depth_dim']
            final_out_channels = self.joint_num * self.depth_dim
        else:
            self.depth_dim = 0
            final_out_channels = self.joint_num # (예: 17)

        self.backbone = ConvNextBackbone(convnext_type=cfg['convnext_type'], drop_path_rate=cfg['drop_rate'])
        self.head = ConvNextHead(
            embedding_size = cfg['embedding_size'],
            depth_dim = final_out_channels,
            joint_num = self.joint_num,
            head_channels = cfg['head_channels'],
            use_3ups = self.use_3ups
        )

        self.quant_flag = cfg['quant_flag']
        if self.quant_flag:
            pass

    def forward(self, x, labels=None):
        raw_heatmap = self.head(self.backbone(x))

        if self.is_3d:
            pred_jts, maxvals = _get_coords_and_maxvals_3d(
                raw_heatmap.detach(), 
                self.joint_num, self.depth_dim, self.output_shape)
        else:
            pred_jts, maxvals = _get_coords_and_maxvals_2d(
                raw_heatmap.detach(), 
                self.joint_num, self.output_shape)

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