from copy import deepcopy

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .attention import Self_Attn2D


def normalize_imagenet(x):
    ''' Normalize input images according to ImageNet standards.

    Args:
        x (tensor): input images
    '''
    x = x.clone()
    x[:, 0] = (x[:, 0] - 0.485) / 0.229
    x[:, 1] = (x[:, 1] - 0.456) / 0.224
    x[:, 2] = (x[:, 2] - 0.406) / 0.225
    return x

def create_resnet(type):
    if type == 'resnet':
        return Resnet
    elif type =='attention-first':
        return ResnetFirst
    elif type == 'attention-last':
        return ResnetLast
    elif type == 'attention-all':
        return ResnetAll


class Resnet(nn.Module):
    r''' ResNet encoder network for image input.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
    '''

    def __init__(self, normalize=False):
        super().__init__()
        self.normalize = normalize
        self.features = models.resnet18(pretrained=True)

    def forward(self, x):
        img = deepcopy(x)
        if self.normalize:
            x = normalize_imagenet(x)

        x = self.features.conv1(x)
        x = self.features.bn1(x)
        x = self.features.relu(x)
        x = self.features.maxpool(x)

        x = self.features.layer1(x)  # 64
        x = self.features.layer2(x)  # 128
        x = self.features.layer3(x)  # 256
        x = self.features.layer4(x)  # 512

        x = self.features.avgpool(x)
        x = torch.flatten(x, 1)  # batch, 512
        return x


class ResnetFirst(nn.Module):
    r''' ResNet encoder network for image input.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
    '''

    def __init__(self, normalize=False):
        super().__init__()
        self.normalize = normalize
        self.features = models.resnet18(pretrained=True)

        self.att = Self_Attn2D(64)
        self.att2 = Self_Attn2D(128)

    def forward(self, x):
        img = deepcopy(x)
        if self.normalize:
            x = normalize_imagenet(x)

        x = self.features.conv1(x)
        x = self.features.bn1(x)
        x = self.features.relu(x)
        x = self.features.maxpool(x)

        x = self.features.layer1(x)  # 64
        x, _ = self.att(x)

        x = self.features.layer2(x)  # 128
        x, _ = self.att2(x)

        x = self.features.layer3(x)  # 256
        x = self.features.layer4(x)  # 512

        x = self.features.avgpool(x)
        x = torch.flatten(x, 1)  # batch, 512
        return x


class ResnetLast(nn.Module):
    r''' ResNet encoder network for image input.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
    '''

    def __init__(self, normalize=False):
        super().__init__()
        self.normalize = normalize
        self.features = models.resnet18(pretrained=True)

        self.att3 = Self_Attn2D(256)
        self.att4 = Self_Attn2D(512)

    def forward(self, x):
        img = deepcopy(x)
        if self.normalize:
            x = normalize_imagenet(x)

        x = self.features.conv1(x)
        x = self.features.bn1(x)
        x = self.features.relu(x)
        x = self.features.maxpool(x)

        x = self.features.layer1(x)  # 64
        x = self.features.layer2(x)  # 128

        x = self.features.layer3(x)  # 256
        x, _ = self.att3(x)

        x = self.features.layer4(x)  # 512
        x, _ = self.att4(x)

        x = self.features.avgpool(x)
        x = torch.flatten(x, 1)  # batch, 512
        return x


class ResnetAll(nn.Module):
    r''' ResNet encoder network for image input.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
    '''

    def __init__(self, normalize=False):
        super().__init__()
        self.normalize = normalize
        self.features = models.resnet18(pretrained=True)

        self.att = Self_Attn2D(64)
        self.att2 = Self_Attn2D(128)
        self.att3 = Self_Attn2D(256)
        self.att4 = Self_Attn2D(512)

    def forward(self, x):
        img = deepcopy(x)
        if self.normalize:
            x = normalize_imagenet(x)

        x = self.features.conv1(x)
        x = self.features.bn1(x)
        x = self.features.relu(x)
        x = self.features.maxpool(x)

        x = self.features.layer1(x)  # 64
        x, _ = self.att(x)

        x = self.features.layer2(x)  # 128
        x, _ = self.att2(x)

        x = self.features.layer3(x)  # 256
        x, _ = self.att3(x)

        x = self.features.layer4(x)  # 512
        x, _ = self.att4(x)

        x = self.features.avgpool(x)
        x = torch.flatten(x, 1)  # batch, 512
        return x