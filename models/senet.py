import math
from collections import OrderedDict
from typing import Sequence, Union

import megengine as mge
import megengine.functional as F
import megengine.module as M
from megengine import hub

from .utils import MaxPool2d


def _weight_init(m):
    if isinstance(m, M.Conv2d):
        M.init.msra_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, M.BatchNorm2d):
        M.init.ones_(m.weight)
        M.init.zeros_(m.bias)


class SEBlock(M.Module):

    def __init__(self, channels, reduction):
        super(SEBlock, self).__init__()
        self.gap = M.AdaptiveAvgPool2d(1)
        self.fc1 = M.Conv2d(channels, channels // reduction, kernel_size=1)
        self.relu = M.ReLU()
        self.fc2 = M.Conv2d(channels // reduction, channels, kernel_size=1)
        self.sigmoid = M.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.gap(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class BaseBottleneck(M.Module):
    """
    Base class for bottlenecks that implements `forward()` method.
    """

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        shortcut = self.downsample(x)

        out = self.se_block(out) + shortcut
        out = self.relu(out)

        return out


class SEBottleneck(BaseBottleneck):
    """
    Bottleneck for SENet154.
    """
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        groups: int,
        reduction: int,
        stride: int = 1,
        downsample: M.Module = None
    ):
        super(SEBottleneck, self).__init__()
        self.conv1 = M.Conv2d(
            in_channels=inplanes,
            out_channels=planes * 2,
            kernel_size=1,
            bias=False)
        self.bn1 = M.BatchNorm2d(planes * 2)
        self.conv2 = M.Conv2d(
            in_channels=planes * 2,
            out_channels=planes * 4,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=groups,
            bias=False)
        self.bn2 = M.BatchNorm2d(planes * 4)
        self.conv3 = M.Conv2d(
            in_channels=planes * 4,
            out_channels=planes * 4,
            kernel_size=1,
            bias=False)
        self.bn3 = M.BatchNorm2d(planes * 4)
        self.relu = M.ReLU()
        self.se_block = SEBlock(planes * 4, reduction=reduction)
        self.downsample = M.Identity() if downsample is None else downsample
        self.stride = stride


class SEResNetBottleneck(BaseBottleneck):
    """
    ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe
    implementation and uses `stride=stride` in `conv1` and not in `conv2`
    (the latter is used in the torchvision implementation of ResNet).
    """
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        groups: int,
        reduction: int,
        stride: int = 1,
        downsample: M.Module = None
    ):
        super(SEResNetBottleneck, self).__init__()
        self.conv1 = M.Conv2d(
            in_channels=inplanes,
            out_channels=planes,
            kernel_size=1,
            bias=False,
            stride=stride)
        self.bn1 = M.BatchNorm2d(planes)
        self.conv2 = M.Conv2d(
            in_channels=planes,
            out_channels=planes,
            kernel_size=3,
            padding=1,
            groups=groups,
            bias=False)
        self.bn2 = M.BatchNorm2d(planes)
        self.conv3 = M.Conv2d(
            in_channels=planes,
            out_channels=planes * 4,
            kernel_size=1,
            bias=False)
        self.bn3 = M.BatchNorm2d(planes * 4)
        self.relu = M.ReLU()
        self.se_block = SEBlock(planes * 4, reduction=reduction)
        self.downsample = M.Identity() if downsample is None else downsample
        self.stride = stride


class SEResNetBlock(M.Module):
    expansion = 1

    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None):
        super(SEResNetBlock, self).__init__()
        self.conv1 = M.Conv2d(
            in_channels=inplanes,
            out_channels=planes,
            kernel_size=3,
            padding=1,
            stride=stride,
            bias=False)
        self.bn1 = M.BatchNorm2d(planes)
        self.conv2 = M.Conv2d(
            in_channels=planes,
            out_channels=planes,
            kernel_size=3,
            padding=1,
            groups=groups,
            bias=False)
        self.bn2 = M.BatchNorm2d(planes)
        self.relu = M.ReLU()
        self.se_block = SEBlock(planes * 4, reduction=reduction)
        self.downsample = M.Identity() if downsample is None else downsample
        self.stride = stride

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        shortcut = self.downsample(x)

        out = self.se_block(out) + shortcut
        out = self.relu(out)

        return out


class SENet(M.Module):

    def __init__(
        self,
        block: Union[M.Module, BaseBottleneck],
        layers: Sequence[int],
        groups: int,
        reduction: int,
        drop_rate: float = 0.2,
        in_chans: int = 3,
        inplanes: int = 64,
        input_3x3: bool = False,
        downsample_kernel_size: int = 1,
        downsample_padding: int = 0,
        num_classes: int = 1000,
    ):
        """
        Parameters
        ----------
        block (M.Module): Bottleneck class.
            - For SENet154: SEBottleneck
            - For SE-ResNet models: SEResNetBottleneck
            - For SE-ResNeXt models:  SEResNeXtBottleneck
        layers (list of ints): Number of residual blocks for 4 layers of the
            network (layer1...layer4).
        groups (int): Number of groups for the 3x3 convolution in each
            bottleneck block.
            - For SENet154: 64
            - For SE-ResNet models: 1
            - For SE-ResNeXt models:  32
        reduction (int): Reduction ratio for Squeeze-and-Excitation modules.
            - For all models: 16
        dropout_p (float or None): Drop probability for the Dropout layer.
            If `None` the Dropout layer is not used.
            - For SENet154: 0.2
            - For SE-ResNet models: None
            - For SE-ResNeXt models: None
        inplanes (int):  Number of input channels for layer1.
            - For SENet154: 128
            - For SE-ResNet models: 64
            - For SE-ResNeXt models: 64
        input_3x3 (bool): If `True`, use three 3x3 convolutions instead of
            a single 7x7 convolution in layer0.
            - For SENet154: True
            - For SE-ResNet models: False
            - For SE-ResNeXt models: False
        downsample_kernel_size (int): Kernel size for downsampling convolutions
            in layer2, layer3 and layer4.
            - For SENet154: 3
            - For SE-ResNet models: 1
            - For SE-ResNeXt models: 1
        downsample_padding (int): Padding for downsampling convolutions in
            layer2, layer3 and layer4.
            - For SENet154: 1
            - For SE-ResNet models: 0
            - For SE-ResNeXt models: 0
        num_classes (int): Number of outputs in `last_linear` layer.
            - For all models: 1000
        """
        super(SENet, self).__init__()
        self.inplanes = inplanes
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        if input_3x3:
            layer0_modules = [
                ('conv1', M.Conv2d(in_chans, 64, 3, stride=2, padding=1, bias=False)),
                ('bn1', M.BatchNorm2d(64)),
                ('relu1', M.ReLU()),
                ('conv2', M.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)),
                ('bn2', M.BatchNorm2d(64)),
                ('relu2', M.ReLU()),
                ('conv3', M.Conv2d(64, inplanes, 3, stride=1, padding=1, bias=False)),
                ('bn3', M.BatchNorm2d(inplanes)),
                ('relu3', M.ReLU()),
            ]
        else:
            layer0_modules = [
                ('conv1', M.Conv2d(
                    in_chans, inplanes, kernel_size=7, stride=2, padding=3, bias=False)),
                ('bn1', M.BatchNorm2d(inplanes)),
                ('relu1', M.ReLU()),
            ]
        self.layer0 = M.Sequential(OrderedDict(layer0_modules))

        self.pool0 = MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )

        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )

        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )

        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )

        self.num_features = 512 * block.expansion
        self.global_pool = M.AdaptiveAvgPool2d(1)
        self.last_linear = M.Linear(self.num_features, num_classes, bias=True)

        for m in self.modules():
            _weight_init(m)

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = M.Sequential(
                M.Conv2d(
                    self.inplanes, planes * block.expansion, kernel_size=downsample_kernel_size,
                    stride=stride, padding=downsample_padding, bias=False),
                M.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, groups,
                        reduction, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))

        return M.Sequential(*layers)

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.last_linear = M.Linear(num_features, num_classes, bias=True)

    def forward_features(self, x):
        x = self.layer0(x)
        x = self.pool0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        x = self.global_pool(x)
        if self.drop_rate > 0.:
            x = F.nn.dropout(x, self.drop_rate, training=self.training)
        x = F.flatten(x, 1)
        return x if pre_logits else self.last_linear(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def seresnet18(**kwargs):
    return SENet(block=SEResNetBlock, layers=[2, 2, 2, 2], groups=1, reduction=16, **kwargs)


def seresnet34(**kwargs):
    return SENet(block=SEResNetBlock, layers=[3, 4, 6, 3], groups=1, reduction=16, **kwargs)


@hub.pretrained(
    "https://studio.brainpp.com/api/v1/activities/3/missions/79/files/151d6cce-e568-46b4-b147-66e4a2039eb4"
)
def seresnet50(**kwargs):
    return SENet(block=SEResNetBottleneck, layers=[3, 4, 6, 3], groups=1, reduction=16, **kwargs)


def seresnet101(**kwargs):
    return SENet(block=SEResNetBottleneck, layers=[3, 4, 23, 3], groups=1, reduction=16, **kwargs)


def seresnet152(**kwargs):
    return SENet(block=SEResNetBottleneck, layers=[3, 8, 36, 3], groups=1, reduction=16, **kwargs)


@hub.pretrained(
    "https://studio.brainpp.com/api/v1/activities/3/missions/79/files/219153e0-c0fe-4a7d-a2a9-f32f7d439ba5"
)
def senet154(**kwargs):
    return SENet(block=SEBottleneck, layers=[3, 8, 36, 3], groups=64, reduction=16,
                 downsample_kernel_size=3, downsample_padding=1,  inplanes=128, input_3x3=True, **kwargs)


def seresnext26_32x4d(**kwargs):
    return SENet(block=SEResNeXtBottleneck, layers=[2, 2, 2, 2], groups=32, reduction=16, **kwargs)


def seresnext50_32x4d(**kwargs):
    return SENet(block=SEResNeXtBottleneck, layers=[3, 4, 6, 3], groups=32, reduction=16, **kwargs)


def seresnext101_32x4d(**kwargs):
    return SENet(block=SEResNeXtBottleneck, layers=[3, 4, 23, 3], groups=32, reduction=16, **kwargs)
