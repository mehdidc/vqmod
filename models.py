import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from nearest_embed import NearestEmbed


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Block(nn.Module):

    def __init__(self, inplanes, outplanes, block_type=BasicBlock, stride=1):
        super().__init__()
        self.inplane = inplanes
        self.outplane = outplanes
        self.block_type = block_type
        self.stride = stride
        self.block = make_block(inplanes, outplanes,
                                block_type, 1, stride=stride)

    def forward(self, x):
        return self.block(x)


def make_block(inplanes, outplanes, block, nb_blocks, stride=1):
    downsample = None
    if stride != 1 or inplanes != outplanes:
        downsample = nn.Sequential(
            conv1x1(inplanes, outplanes, stride),
            nn.BatchNorm2d(outplanes),
        )
    layers = []
    layers.append(block(inplanes, outplanes, stride, downsample))
    for _ in range(1, nb_blocks):
        layers.append(block(outplanes, outplanes))
    return nn.Sequential(*layers)


class FC(nn.Module):

    def __init__(self, inplane, outplane=1000):
        super().__init__()
        self.inplane = inplane
        self.outplane = outplane
        self.fc = nn.Linear(inplane, outplane)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class Controller(nn.Module):
    pass


class ModularNet(Controller):

    def __init__(self, modules, depth=1, dim_embeddings=128):
        super().__init__()
        self.depth = depth
        self.dim_embeddings = dim_embeddings
        self.inplane = modules[0].inplane
        self.outplane = modules[-1].outplane
        self.controller = nn.Sequential(
            nn.Conv2d(self.inplane, depth * dim_embeddings, kernel_size=1),
            nn.AdaptiveAvgPool2d(1)
        )
        self.components = nn.ModuleList(modules)
        self.neareat_emb = NearestEmbed(len(modules), dim_embeddings)

    def forward(self, x):
        ctl = self.controller(x)
        ctl = ctl.view(ctl.size(0), self.dim_embeddings, self.depth)
        _, out_idx = self.neareat_emb(ctl, weight_sg=True)
        ctl_nearest, _ = self.neareat_emb(ctl.detach())
        out_idx = out_idx.view(len(out_idx), self.depth)
        ys = []
        for example in range(len(ctl)):
            y = x[example:example+1]
            for depth in range(self.depth):
                mod = self.components[out_idx[example, depth].item()]
                y = mod(y)
            ys.append(y)
        y = torch.cat(ys, dim=0)
        return y, ctl, ctl_nearest
    
    @staticmethod
    def vq_loss_function(ctl, ctl_nearest):
        vq_loss = F.mse_loss(ctl_nearest, ctl.detach())
        commit_loss = F.mse_loss(ctl_nearest.detach(), ctl)
        return vq_loss + commit_loss


class Model(nn.Module):

    def __init__(self, nb_colors=3, nb_classes=10):
        super().__init__()
        self.f1 = Block(nb_colors, 64)
        self.f2 = Block(64, 64)
        self.f3 = Block(64, 64)
        self.modular = ModularNet([self.f2, self.f3], depth=3)
        self.fc = FC(64, nb_classes)
        self.apply(weight_init)

    def forward(self, x):
        x = self.f1(x)
        x, ctl, ctl_nearest = self.modular(x)
        return self.fc(x), ctl, ctl_nearest


if __name__ == '__main__':
    net = Model(nb_classes=2)
    x = torch.rand(1, 3, 32, 32, requires_grad=True)
    y, ctl, ctl_nearest = net(x)
