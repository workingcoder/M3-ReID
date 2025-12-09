# ------------------------------------------------------------------------------
# File:    M3-ReID/models/backbones/resnet.py
#
# Description:
#    This module implements ResNet backbones for feature extraction in ReID.
#    It includes standard architectures (ResNet18, 34, 50, 101, 152) and utilities
#    to load pretrained weights while removing the classification layer (fc).
#
# Key Features:
# - Standard BasicBlock and Bottleneck implementations.
# - ResNet architecture with configurable strides (useful for ReID to keep feature map size).
# - Utilities to strip fully connected layers from ImageNet pretrained weights.
#
# Classes:
# - BasicBlock
# - Bottleneck
# - ResNet
#
# Main Functions:
# - resnet18
# - resnet34
# - resnet50
# - resnet101
# - resnet152
# ------------------------------------------------------------------------------

import math
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """
    Creates a 3x3 convolution layer with padding.

    Args:
        in_planes (int): Number of input channels.
        out_planes (int): Number of output channels.
        stride (int): Stride of the convolution (default 1).
        dilation (int): Dilation rate (default 1).

    Returns:
        nn.Conv2d: The convolutional layer.
    """
    # original padding is 1; original dilation is 1
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    """
    Standard ResNet Basic Block consisting of two 3x3 convolutions.
    Used primarily in ResNet18 and ResNet34.
    """

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        """
        Initialize the BasicBlock.

        Args:
            inplanes (int): Number of input channels.
            planes (int): Number of output channels.
            stride (int): Stride for the first convolution.
            downsample (nn.Module, optional): Downsampling layer for residual connection.
            dilation (int): Dilation rate.
        """

        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """
        Forward pass of the BasicBlock.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after residual connection and ReLU.
        """

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    Standard ResNet Bottleneck Block consisting of 1x1, 3x3, and 1x1 convolutions.
    Used in ResNet50, ResNet101, and ResNet152.
    """

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        """
        Initialize the Bottleneck block.

        Args:
            inplanes (int): Number of input channels.
            planes (int): Number of intermediate channels (output channels will be planes * 4).
            stride (int): Stride for the 3x3 convolution.
            downsample (nn.Module, optional): Downsampling layer for residual connection.
            dilation (int): Dilation rate.
        """

        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # original padding is 1; original dilation is 1
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=dilation, bias=False,
                               dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """
        Forward pass of the Bottleneck block.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after residual connection and ReLU.
        """

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    Main ResNet architecture definition.
    Supports configurable strides in the last layer to maintain higher spatial resolution
    for ReID tasks.
    """

    def __init__(self, block, layers, last_conv_stride=2, last_conv_dilation=1):
        """
        Initialize the ResNet model.

        Args:
            block (class): Block type (BasicBlock or Bottleneck).
            layers (list): List of integers specifying the number of blocks per layer.
            last_conv_stride (int): Stride for the last convolutional layer (default 2).
                                    Set to 1 for higher resolution feature maps.
            last_conv_dilation (int): Dilation for the last layer (default 1).
        """

        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=last_conv_stride, dilation=last_conv_dilation)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        """
        Constructs a sequence of blocks for a specific layer.

        Process:
        1. Create a downsample layer if stride != 1 or input channels != output channels.
        2. Append the first block (which handles stride and downsampling).
        3. Append subsequent blocks (which maintain channel dimensions).

        Args:
            block (class): Block type.
            planes (int): Number of intermediate channels.
            blocks (int): Number of blocks to stack.
            stride (int): Stride for the first block.
            dilation (int): Dilation rate.

        Returns:
            nn.Sequential: The constructed layer.
        """

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the ResNet backbone.

        Process:
        1. Initial convolution, BatchNorm, ReLU, and MaxPool.
        2. Pass through Layer 1, Layer 2, Layer 3, and Layer 4.

        Args:
            x (Tensor): Input image tensor [Batch, 3, H, W].

        Returns:
            Tensor: Extracted feature maps from the last layer.
        """

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def remove_fc(state_dict):
    """
    Removes the fully connected (fc) layer parameters from a state dictionary.
    Used when loading ImageNet pretrained weights for ReID (where the class count differs).

    Process:
    1. Iterate through the keys of the state_dict.
    2. Delete any key that starts with 'fc.'.

    Args:
        state_dict (dict): Model state dictionary.

    Returns:
        dict: The modified state dictionary.
    """

    for key, value in list(state_dict.items()):
        if key.startswith('fc.'):
            del state_dict[key]
    return state_dict


def resnet18(pretrained=False, **kwargs):
    """
    Constructs a ResNet-18 model.

    Process:
    1. Instantiate ResNet with BasicBlock and layer configuration [2, 2, 2, 2].
    2. If pretrained, download and load ImageNet weights (excluding fc layer).

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet.
        **kwargs: Additional arguments passed to the ResNet constructor.

    Returns:
        ResNet: The constructed model.
    """

    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(remove_fc(model_zoo.load_url(model_urls['resnet18'])))
    return model


def resnet34(pretrained=False, **kwargs):
    """
    Constructs a ResNet-34 model.

    Process:
    1. Instantiate ResNet with BasicBlock and layer configuration [3, 4, 6, 3].
    2. If pretrained, download and load ImageNet weights (excluding fc layer).

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet.
        **kwargs: Additional arguments passed to the ResNet constructor.

    Returns:
        ResNet: The constructed model.
    """

    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(remove_fc(model_zoo.load_url(model_urls['resnet34'])))
    return model


def resnet50(pretrained=False, **kwargs):
    """
    Constructs a ResNet-50 model.

    Process:
    1. Instantiate ResNet with Bottleneck and layer configuration [3, 4, 6, 3].
    2. If pretrained, download and load ImageNet weights (excluding fc layer).

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet.
        **kwargs: Additional arguments passed to the ResNet constructor.

    Returns:
        ResNet: The constructed model.
    """

    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(remove_fc(model_zoo.load_url(model_urls['resnet50'])))
    return model


def resnet101(pretrained=False, **kwargs):
    """
    Constructs a ResNet-101 model.

    Process:
    1. Instantiate ResNet with Bottleneck and layer configuration [3, 4, 23, 3].
    2. If pretrained, download and load ImageNet weights (excluding fc layer).

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet.
        **kwargs: Additional arguments passed to the ResNet constructor.

    Returns:
        ResNet: The constructed model.
    """

    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(remove_fc(model_zoo.load_url(model_urls['resnet101'])))
    return model


def resnet152(pretrained=False, **kwargs):
    """
    Constructs a ResNet-152 model.

    Process:
    1. Instantiate ResNet with Bottleneck and layer configuration [3, 8, 36, 3].
    2. If pretrained, download and load ImageNet weights (excluding fc layer).

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet.
        **kwargs: Additional arguments passed to the ResNet constructor.

    Returns:
        ResNet: The constructed model.
    """

    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(remove_fc(model_zoo.load_url(model_urls['resnet152'])))
    return model
