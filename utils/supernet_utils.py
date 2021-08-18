import torch
import random
import copy
import pickle
import torch.nn as nn
import torch.distributed as dist
import pandas as pd
import torch.nn.functional as F

from utils.distributed_utils import get_rank
from functools import partial
from itertools import compress
from .macc import *


def sort_df(df, column_idx):
    col = df.iloc[:, column_idx]
    df = df.iloc[[i[1] for i in sorted(zip(col, range(len(col))), key=lambda x: x[0])]]
    return df


# latency lookup table we used
LUT = []

for i in range(15):
    LUT += pickle.load(open(
        f'/..absolute/..path/..to/NSE/resources/mb_imagenet_timedict_v1/mb_imagenet_ops_list_time(b128_{i}).dump',
        'rb'))
LUT = pd.DataFrame(list(list(zip(*LUT))[1]),
                   columns=["hw", "c_in", "c_out", "kernel", "stride", "padding", "dilation", "group",
                            'type_code', "exec_time"])
LUT = sort_df(LUT, 9)
LUT = LUT.drop_duplicates(
    subset=["hw", "c_in", "c_out", "kernel", "stride", "padding", "dilation", "group", 'type_code'], keep='first')


def lookup(type, hw, cin, cout, k, stride, padding, dilation, group):
    if k in [(1, 1), (3, 3), (5, 5), (7, 7)]:
        k = k[0]
    time = LUT[(LUT.type_code == type) &
               (LUT.hw == hw) &
               (LUT.c_in == cin) &
               (LUT.c_out == cout) &
               (LUT.kernel == k) &
               (LUT.stride == stride) &
               (LUT.padding == padding) &
               (LUT.dilation == dilation) &
               (LUT.group == group)].exec_time.values
    if time.size == 0:
        raise NotImplementedError(
            f'type code: {[type, hw, cin, cout, k, stride, padding, dilation, group]}\tThis is not collected')
    else:
        return torch.tensor(time, dtype=torch.float).cuda()


def _make_divisible(v, divisor, min_value=None):
    """
    It ensures that all layers have a channel number that is divisible by 8
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        self.time = 0
        self.netpara = 0
        self.macs = 0

    def forward(self, input, *args):
        return input


class Swish(nn.Module):
    def __init__(self, **kwargs):
        super(Swish, self).__init__()
        self.time = 0
        self.netpara = 0
        self.macs = 0

    def forward(self, input, *args):
        return input * F.sigmoid(input)


class SEModule(nn.Module):

    def __init__(self, channels, reduction, activation):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = activation(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class Rec(nn.Module):
    def __init__(self, inplanes, outplanes, size_in, stride=1, k=3, t=6, timing=True, forward=True,
                 activation=nn.ReLU, use_se=False):
        super(Rec, self).__init__()

        padding = k // 2
        self.time = 0
        self.size_in = size_in
        self.size_out = (size_in - 1 + stride) // stride

        if forward:
            self.conv1 = nn.Conv2d(inplanes, inplanes * t, kernel_size=1, bias=False)
            self.bn1 = BN(inplanes * t)

            self.conv2_1 = nn.Conv2d(inplanes * t, inplanes * t, kernel_size=(1, k), stride=(1, stride),
                                     padding=(0, padding), bias=False)
            self.bn2_1 = BN(inplanes * t)
            self.conv2_2 = nn.Conv2d(inplanes * t, inplanes * t, kernel_size=(k, 1), stride=(stride, 1),
                                     padding=(padding, 0), bias=False)
            self.bn2_2 = BN(inplanes * t)

            self.conv3 = nn.Conv2d(inplanes * t, outplanes, kernel_size=1, bias=False)
            self.bn3 = BN(outplanes)

            self.activation = activation(inplace=True)

            self.se = SEModule(inplanes * t, 4, activation) if use_se else Identity()

        self.stride = stride
        self.t = t
        self.inplanes = inplanes
        self.outplanes = outplanes

        if timing:
            self.time += lookup(1, self.size_in, inplanes, inplanes * t, 1, 1, 0, 1, 1)

            self.time += lookup(1, self.size_in, inplanes * t, inplanes * t, (1, k), (1, stride), (0, padding), 1, 1)
            self.time += lookup(1, (self.size_in, self.size_out), inplanes * t, inplanes * t, (k, 1), (stride, 1),
                                (padding, 0), 1, 1)

            self.time += lookup(1, self.size_out, inplanes * t, outplanes, 1, 1, 0, 1, 1)
            if self.inplanes == self.outplanes and self.stride == 1:
                self.time += lookup(4, self.size_in, self.inplanes, self.outplanes, 0, 0, 0, 0, 0)

        self.netpara = sum(p.numel() for p in self.parameters()) / 1e6
        self.macs = 0
        self.macs += mac_of_convolution(self.conv1.weight,
                                        torch.ones(1, inplanes, self.size_in, self.size_in),
                                        torch.ones(1, inplanes * t, self.size_in, self.size_in))
        self.macs += mac_of_convolution(self.conv2_1.weight,
                                        torch.ones(1, inplanes * t, self.size_in, self.size_in),
                                        torch.ones(1, inplanes * t, self.size_in, self.size_out))
        self.macs += mac_of_convolution(self.conv2_2.weight,
                                        torch.ones(1, inplanes * t, self.size_in, self.size_out),
                                        torch.ones(1, inplanes * t, self.size_out, self.size_out))
        self.macs += mac_of_convolution(self.conv3.weight,
                                        torch.ones(1, inplanes * t, self.size_out, self.size_out),
                                        torch.ones(1, outplanes, self.size_out, self.size_out))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2_1(out)
        out = self.bn2_1(out)
        out = self.activation(out)

        out = self.conv2_2(out)
        out = self.bn2_2(out)
        out = self.activation(out)

        out = self.se(out)

        out = self.conv3(out)
        out = self.bn3(out)

        return out


class DualConv(nn.Module):
    def __init__(self, inplanes, outplanes, size_in, stride=1, k=3, t=6, timing=True, forward=True,
                 activation=nn.ReLU, use_se=False):
        super(DualConv, self).__init__()
        padding = k // 2
        self.time = 0
        self.size_in = size_in
        self.size_out = (size_in - 1 + stride) // stride

        if forward:
            self.conv1 = nn.Conv2d(inplanes, inplanes * t, kernel_size=1, bias=False)
            self.bn1 = BN(inplanes * t)

            self.conv2_1 = nn.Conv2d(inplanes * t, inplanes * t, kernel_size=k, stride=1, padding=padding, bias=False)
            self.bn2_1 = BN(inplanes * t)
            self.conv2_2 = nn.Conv2d(inplanes * t, inplanes * t, kernel_size=k, stride=stride, padding=padding,
                                     bias=False)
            self.bn2_2 = BN(inplanes * t)

            self.conv3 = nn.Conv2d(inplanes * t, outplanes, kernel_size=1, bias=False)
            self.bn3 = BN(outplanes)

            self.activation = activation(inplace=True)

            self.se = SEModule(inplanes * t, 4, activation) if use_se else Identity()

        self.stride = stride
        self.t = t
        self.inplanes = inplanes
        self.outplanes = outplanes

        if timing:
            self.time += lookup(1, self.size_in, inplanes, inplanes * t, 1, 1, 0, 1, 1)
            self.time += lookup(1, self.size_in, inplanes * t, inplanes * t, k, 1, padding, 1, 1)
            self.time += lookup(1, self.size_in, inplanes * t, inplanes * t, k, stride, padding, 1, 1)
            self.time += lookup(1, self.size_out, inplanes * t, outplanes, 1, 1, 0, 1, 1)
            if self.inplanes == self.outplanes and self.stride == 1:
                self.time += lookup(4, self.size_in, self.inplanes, self.outplanes, 0, 0, 0, 0, 0)

        self.netpara = sum(p.numel() for p in self.parameters()) / 1e6
        self.macs = 0
        self.macs += mac_of_convolution(self.conv1.weight,
                                        torch.ones(1, inplanes, self.size_in, self.size_in),
                                        torch.ones(1, inplanes * t, self.size_in, self.size_in))
        self.macs += mac_of_convolution(self.conv2_1.weight,
                                        torch.ones(1, inplanes * t, self.size_in, self.size_in),
                                        torch.ones(1, inplanes * t, self.size_in, self.size_in))
        self.macs += mac_of_convolution(self.conv2_2.weight,
                                        torch.ones(1, inplanes * t, self.size_in, self.size_in),
                                        torch.ones(1, inplanes * t, self.size_out, self.size_out))
        self.macs += mac_of_convolution(self.conv3.weight,
                                        torch.ones(1, inplanes * t, self.size_out, self.size_out),
                                        torch.ones(1, outplanes, self.size_out, self.size_out))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2_1(out)
        out = self.bn2_1(out)
        out = self.activation(out)

        out = self.conv2_2(out)
        out = self.bn2_2(out)
        out = self.activation(out)

        out = self.se(out)

        out = self.conv3(out)
        out = self.bn3(out)

        return out


class NormalConv(nn.Module):
    def __init__(self, inplanes, outplanes, size_in, stride=1, k=3, t=6, timing=True, forward=True,
                 activation=nn.ReLU, use_se=False):
        super(NormalConv, self).__init__()
        padding = k // 2
        self.time = 0
        self.size_in = size_in
        self.size_out = (size_in - 1 + stride) // stride

        if forward:
            self.conv1 = nn.Conv2d(inplanes, inplanes * t, kernel_size=1, bias=False)
            self.bn1 = BN(inplanes * t)

            self.conv2 = nn.Conv2d(inplanes * t, inplanes * t, kernel_size=k, stride=stride, padding=padding,
                                   bias=False)
            self.bn2 = BN(inplanes * t)

            self.conv3 = nn.Conv2d(inplanes * t, outplanes, kernel_size=1, bias=False)
            self.bn3 = BN(outplanes)

            self.activation = activation(inplace=True)

            self.se = SEModule(inplanes * t, 4, activation) if use_se else Identity()

        self.stride = stride
        self.t = t
        self.inplanes = inplanes
        self.outplanes = outplanes

        if timing:
            self.time += lookup(1, self.size_in, inplanes, inplanes * t, 1, 1, 0, 1, 1)
            self.time += lookup(1, self.size_in, inplanes * t, inplanes * t, k, stride, padding, 1, 1)
            self.time += lookup(1, self.size_out, inplanes * t, outplanes, 1, 1, 0, 1, 1)
            if self.inplanes == self.outplanes and self.stride == 1:
                self.time += lookup(4, self.size_in, self.inplanes, self.outplanes, 0, 0, 0, 0, 0)

        self.netpara = sum(p.numel() for p in self.parameters()) / 1e6
        self.macs = 0
        self.macs += mac_of_convolution(self.conv1.weight,
                                        torch.ones(1, inplanes, self.size_in, self.size_in),
                                        torch.ones(1, inplanes * t, self.size_in, self.size_in))
        self.macs += mac_of_convolution(self.conv2.weight,
                                        torch.ones(1, inplanes * t, self.size_in, self.size_in),
                                        torch.ones(1, inplanes * t, self.size_out, self.size_out))
        self.macs += mac_of_convolution(self.conv3.weight,
                                        torch.ones(1, inplanes * t, self.size_out, self.size_out),
                                        torch.ones(1, outplanes, self.size_out, self.size_out))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out = self.se(out)

        out = self.conv3(out)
        out = self.bn3(out)

        return out


class LinearBottleneck(nn.Module):
    def __init__(self, inplanes, outplanes, size_in, stride=1, k=3, t=6, timing=True, forward=True,
                 group=1, dilation=1, activation=nn.ReLU, use_se=False):
        super(LinearBottleneck, self).__init__()
        if group in [(1, 2), (2, 1)]:
            group_fore, group_post = group
        else:
            group_fore, group_post = group, group

        dk = k + (dilation - 1) * 2
        padding = dk // 2
        self.time = 0
        self.size_in = size_in
        self.size_out = (size_in - 1 + stride) // stride

        if forward:
            self.conv1 = nn.Conv2d(inplanes, inplanes * t, kernel_size=1, bias=False, groups=group_fore)
            self.bn1 = BN(inplanes * t)
            self.conv2 = nn.Conv2d(inplanes * t, inplanes * t, kernel_size=k, stride=stride, padding=padding,
                                   bias=False,
                                   groups=inplanes * t, dilation=dilation)
            self.bn2 = BN(inplanes * t)
            self.conv3 = nn.Conv2d(inplanes * t, outplanes, kernel_size=1, bias=False, groups=group_post)
            self.bn3 = BN(outplanes)

            self.activation = activation(inplace=True)

            self.se = SEModule(inplanes * t, 4, activation) if use_se else Identity()

        self.stride = stride
        self.t = t
        self.inplanes = inplanes
        self.outplanes = outplanes

        if timing:
            self.time += lookup(1, self.size_in, inplanes, inplanes * t, 1, 1, 0, 1, 1)
            self.time += lookup(1, self.size_in, inplanes * t, inplanes * t, k, stride, padding, 1, inplanes * t)
            self.time += lookup(1, self.size_out, inplanes * t, outplanes, 1, 1, 0, 1, 1)
            if self.inplanes == self.outplanes and self.stride == 1:
                self.time += lookup(4, self.size_in, self.inplanes, self.outplanes, 0, 0, 0, 0, 0)

        self.netpara = sum(p.numel() for p in self.parameters()) / 1e6
        self.macs = 0
        self.macs += mac_of_convolution(self.conv1.weight,
                                        torch.ones(1, inplanes, self.size_in, self.size_in),
                                        torch.ones(1, inplanes * t, self.size_in, self.size_in))
        self.macs += mac_of_convolution(self.conv2.weight,
                                        torch.ones(1, inplanes * t, self.size_in, self.size_in),
                                        torch.ones(1, inplanes * t, self.size_out, self.size_out))
        self.macs += mac_of_convolution(self.conv3.weight,
                                        torch.ones(1, inplanes * t, self.size_out, self.size_out),
                                        torch.ones(1, outplanes, self.size_out, self.size_out))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out = self.se(out)

        out = self.conv3(out)
        out = self.bn3(out)

        return out


class MBCell(nn.Module):
    candidate_num = 19
    indent = [3, 3, 3, 2, 2, 3, 3]

    def __init__(self, cin, size_in, stride, cout):
        super(MBCell, self).__init__()
        self.stride = stride
        self.cin = cin
        self.cout = cout

        self.size_in = size_in

        self.candidates = nn.ModuleList()

        for k in [3, 5, 7]:
            for t in [1, 3, 6]:
                self.candidates.append(LinearBottleneck(self.cin, self.cout, self.size_in, self.stride, k, t))

        for t in [1, 2]:
            self.candidates.append(NormalConv(self.cin, self.cout, self.size_in, self.stride, 3, t))

        for t in [1, 2]:
            self.candidates.append(DualConv(self.cin, self.cout, self.size_in, self.stride, 3, t))

        for k in [5, 7]:
            for t in [1, 2, 4]:
                self.candidates.append(Rec(self.cin, self.cout, self.size_in, self.stride, k, t))

    def resource(self, path):
        if len(path) == 0:
            return 0, 0, 0
        elif self.cin == self.cout and self.stride == 1:
            return sum([self.candidates[p].time for p in path]), sum(
                [self.candidates[p].netpara for p in path]), sum(
                [self.candidates[p].macs for p in path])
        else:
            return sum([self.candidates[p].time for p in path]), sum(
                [self.candidates[p].netpara for p in path]), sum(
                [self.candidates[p].macs for p in path])

    def forward(self, curr_layer, path):
        """Runs the conv cell."""
        if len(path) == 0:
            return curr_layer, 0, 0, 0
        elif self.cin == self.cout and self.stride == 1:
            out = [self.candidates[p](curr_layer) for p in path]
            out.append(curr_layer)
            return sum(out) / len(out), sum([self.candidates[p].time for p in path]), sum(
                [self.candidates[p].netpara for p in path]), sum(
                [self.candidates[p].macs for p in path])

        else:
            out = [self.candidates[p](curr_layer) for p in path]
            return sum(out) / len(out), sum([self.candidates[p].time for p in path]) + lookup(4,
                                                                                              self.size_in // self.stride,
                                                                                              self.cout, self.cout, 0,
                                                                                              0, 0, 0, 0) * (
                           len(path) - 1), sum([self.candidates[p].netpara for p in path]), sum(
                [self.candidates[p].macs for p in path])


class MACMBCell(nn.Module):
    candidate_num = 27
    indent = None

    def __init__(self, cin, size_in, stride, cout):
        super(MACMBCell, self).__init__()
        self.stride = stride
        self.cin = cin
        self.cout = cout

        self.size_in = size_in

        self.candidates = nn.ModuleList()

        for k in [3, 5, 7, 9, 11]:
            for t in [1, 3, 6]:
                if k == 3:
                    for d in [1, 2, 3]:
                        self.candidates.append(
                            LinearBottleneck(self.cin, self.cout, self.size_in, self.stride, k, t, dilation=d,
                                             timing=False))
                else:
                    self.candidates.append(
                        LinearBottleneck(self.cin, self.cout, self.size_in, self.stride, k, t, timing=False))

        for k in [5, 7]:
            for t in [1, 2, 4]:
                self.candidates.append(Rec(self.cin, self.cout, self.size_in, self.stride, k, t, timing=False))

    def resource(self, path):
        if len(path) == 0:
            return 0, 0, 0
        elif self.cin == self.cout and self.stride == 1:
            return sum([self.candidates[p].time for p in path]), sum(
                [self.candidates[p].netpara for p in path]), sum(
                [self.candidates[p].macs for p in path])
        else:
            return sum([self.candidates[p].time for p in path]), sum(
                [self.candidates[p].netpara for p in path]), sum(
                [self.candidates[p].macs for p in path])

    def forward(self, curr_layer, path):
        """Runs the conv cell."""
        if len(path) == 0:
            return curr_layer, 0, 0, 0
        elif self.cin == self.cout and self.stride == 1:
            out = [self.candidates[p](curr_layer) for p in path]
            out.append(curr_layer)
            return sum(out) / len(out), sum([self.candidates[p].time for p in path]), sum(
                [self.candidates[p].netpara for p in path]), sum(
                [self.candidates[p].macs for p in path])

        else:
            out = [self.candidates[p](curr_layer) for p in path]
            return sum(out) / len(out), sum([self.candidates[p].time for p in path]), sum(
                [self.candidates[p].netpara for p in path]), sum(
                [self.candidates[p].macs for p in path])


class GMACMBCell(nn.Module):
    candidate_num = 27 + 45
    indent = None

    def __init__(self, cin, size_in, stride, cout):
        super(GMACMBCell, self).__init__()
        self.stride = stride
        self.cin = cin
        self.cout = cout

        self.size_in = size_in

        self.candidates = nn.ModuleList()

        for k in [3, 5, 7, 9, 11]:
            for t in [1, 3, 6]:
                if k == 3:
                    for d in [1, 2, 3]:
                        self.candidates.append(
                            LinearBottleneck(self.cin, self.cout, self.size_in, self.stride, k, t, dilation=d,
                                             timing=False))
                else:
                    self.candidates.append(
                        LinearBottleneck(self.cin, self.cout, self.size_in, self.stride, k, t, timing=False))

        for k in [5, 7]:
            for t in [1, 2, 4]:
                self.candidates.append(Rec(self.cin, self.cout, self.size_in, self.stride, k, t, timing=False))

        for k in [3, 5, 7, 9, 11]:
            for t in [1, 3, 6]:
                self.candidates.append(
                    LinearBottleneck(self.cin, self.cout, self.size_in, self.stride, k, t, group=(1, 2), timing=False))
                self.candidates.append(
                    LinearBottleneck(self.cin, self.cout, self.size_in, self.stride, k, t, group=(2, 1), timing=False))
                self.candidates.append(
                    LinearBottleneck(self.cin, self.cout, self.size_in, self.stride, k, t, group=2, timing=False))

    def resource(self, path):
        if len(path) == 0:
            return 0, 0, 0
        elif self.cin == self.cout and self.stride == 1:
            return sum([self.candidates[p].time for p in path]), sum(
                [self.candidates[p].netpara for p in path]), sum(
                [self.candidates[p].macs for p in path])
        else:
            return sum([self.candidates[p].time for p in path]), sum(
                [self.candidates[p].netpara for p in path]), sum(
                [self.candidates[p].macs for p in path])

    def forward(self, curr_layer, path):
        """Runs the conv cell."""
        if len(path) == 0:
            return curr_layer, 0, 0, 0
        elif self.cin == self.cout and self.stride == 1:
            out = [self.candidates[p](curr_layer) for p in path]
            out.append(curr_layer)
            return sum(out) / len(out), sum([self.candidates[p].time for p in path]), sum(
                [self.candidates[p].netpara for p in path]), sum(
                [self.candidates[p].macs for p in path])

        else:
            out = [self.candidates[p](curr_layer) for p in path]
            return sum(out) / len(out), sum([self.candidates[p].time for p in path]), sum(
                [self.candidates[p].netpara for p in path]), sum(
                [self.candidates[p].macs for p in path])


class MBValCell(nn.Module):
    candidate_num = 19
    candidates = []
    for k in [3, 5, 7]:
        for t in [1, 3, 6]:
            candidates.append(partial(LinearBottleneck, k=k, t=t, timing=False))
    for t in [1, 2]:
        candidates.append(partial(NormalConv, k=3, t=t, timing=False))
    for t in [1, 2]:
        candidates.append(partial(DualConv, k=3, t=t, timing=False))
    for k in [5, 7]:
        for t in [1, 2, 4]:
            candidates.append(partial(Rec, k=k, t=t, timing=False))

    def __init__(self, cin, size_in, stride, cout, branches, keep_prob=-1, activation=nn.ReLU, use_se=False):
        super(MBValCell, self).__init__()
        self.stride = stride
        self.cin = cin
        self.cout = cout
        self.size_in = size_in
        self.pathes = nn.ModuleList()
        for branch in branches:
            self.pathes.append(
                MBValCell.candidates[branch](inplanes=self.cin, outplanes=self.cout, size_in=self.size_in,
                                             stride=self.stride, activation=activation, use_se=use_se))

    def forward(self, curr_layer):
        """Runs the conv cell."""
        if self.cin == self.cout and self.stride == 1:
            out = [path(curr_layer) for path in self.pathes]
            out.append(curr_layer)
            return sum(out) / len(out)
        else:
            out = [path(curr_layer) for path in self.pathes]
            return sum(out) / len(out)


class MACMBValCell(nn.Module):
    candidate_num = 27
    candidates = []

    for k in [3, 5, 7, 9, 11]:
        for t in [1, 3, 6]:
            if k == 3:
                for d in [1, 2, 3]:
                    candidates.append(partial(LinearBottleneck, k=k, t=t, dilation=d, timing=False))
            else:
                candidates.append(partial(LinearBottleneck, k=k, t=t, timing=False))

    for k in [5, 7]:
        for t in [1, 2, 4]:
            candidates.append(partial(Rec, k=k, t=t, timing=False))

    def __init__(self, cin, size_in, stride, cout, branches, keep_prob=-1, activation=nn.ReLU, use_se=False):
        super(MACMBValCell, self).__init__()
        self.stride = stride
        self.cin = cin
        self.cout = cout
        self.size_in = size_in
        self.pathes = nn.ModuleList()
        for branch in branches:
            self.pathes.append(
                MACMBValCell.candidates[branch](inplanes=self.cin, outplanes=self.cout, size_in=self.size_in,
                                                stride=self.stride, activation=activation, use_se=use_se))

    def forward(self, curr_layer):
        """Runs the conv cell."""
        if self.cin == self.cout and self.stride == 1:
            out = [path(curr_layer) for path in self.pathes]
            out.append(curr_layer)
            return sum(out) / len(out)
        else:
            out = [path(curr_layer) for path in self.pathes]
            return sum(out) / len(out)


class MACMBGValCell(nn.Module):
    candidate_num = 27 + 45
    candidates = []

    for k in [3, 5, 7, 9, 11]:
        for t in [1, 3, 6]:
            if k == 3:
                for d in [1, 2, 3]:
                    candidates.append(partial(LinearBottleneck, k=k, t=t,
                                              dilation=d, timing=False))
            else:
                candidates.append(partial(LinearBottleneck, k=k, t=t, timing=False))

    for k in [5, 7]:
        for t in [1, 2, 4]:
            candidates.append(partial(Rec, k=k, t=t, timing=False))

    for k in [3, 5, 7, 9, 11]:
        for t in [1, 3, 6]:
            candidates.append(partial(LinearBottleneck, k=k, t=t, group=(1, 2), timing=False))
            candidates.append(partial(LinearBottleneck, k=k, t=t, group=(2, 1), timing=False))
            candidates.append(partial(LinearBottleneck, k=k, t=t, group=2, timing=False))

    def __init__(self, cin, size_in, stride, cout, branches, keep_prob=-1, activation=nn.ReLU, use_se=False):
        super(MACMBGValCell, self).__init__()
        self.stride = stride
        self.cin = cin
        self.cout = cout
        self.size_in = size_in
        self.pathes = nn.ModuleList()
        for branch in branches:
            self.pathes.append(
                MACMBGValCell.candidates[branch](inplanes=self.cin, outplanes=self.cout, size_in=self.size_in,
                                                 stride=self.stride, activation=activation, use_se=use_se))

    def forward(self, curr_layer):
        """Runs the conv cell."""
        if self.cin == self.cout and self.stride == 1:
            out = [path(curr_layer) for path in self.pathes]
            out.append(curr_layer)
            return sum(out) / len(out)
        else:
            out = [path(curr_layer) for path in self.pathes]
            return sum(out) / len(out)


class ValNet(nn.Module):
    def __init__(self, scale, channel_dist, num_classes, input_size, Cell, cell_seq,
                 bn_sync_stats=True, keep_prob=-1, activation=nn.ReLU, use_se=False):
        super(ValNet, self).__init__()
        global BN

        BN = nn.BatchNorm2d

        self.activation = activation
        self._time = 0
        self.scale = scale
        self.c = list(channel_dist[:1]) + [_make_divisible(ch * self.scale, 8) for ch in channel_dist[1:]]
        self.num_classes = num_classes
        self.input_size = input_size
        cell_seq = list(cell_seq)
        self.total_blocks = len(cell_seq)

        self._set_stem()

        self.cells = nn.ModuleList()

        self.stage = 0
        for cell_idx, (c, branch) in enumerate(cell_seq):
            if c == 'N':
                stride = 1

                cout = cin = self.c[self.stage]
            elif c == 'E':
                self.stage += 1
                stride = 1

                cout = self.c[self.stage]
                cin = self.c[self.stage - 1]
            elif c == 'R':
                self.stage += 1
                stride = 2
                cout = self.c[self.stage]
                cin = self.c[self.stage - 1]
            else:
                raise NotImplementedError(f'unimplemented supcell type: {c}')
            curr_drop_path_keep_prob = self.calculate_curr_drop_path_keep_prob(cell_idx, keep_prob)

            self.cells.append(Cell(cin, self.curr_size, stride, cout, branch, keep_prob=curr_drop_path_keep_prob,
                                   activation=activation, use_se=use_se))
            self.curr_size = (self.curr_size - 1 + stride) // stride

        self._set_tail()

        self.netpara = sum(p.numel() for p in self.parameters()) / 1e6


    def calculate_curr_drop_path_keep_prob(self, cell_idx, drop_path_keep_prob):
        layer_ratio = cell_idx / float(self.total_blocks)
        return 1 - layer_ratio * (1 - drop_path_keep_prob)

    def adjust_keep_prob(self, curr_epoch, epochs):
        ratio = float(curr_epoch) / epochs
        for cell_idx, cell in enumerate(self.cells):
            for path in cell.pathes:
                path.adjust_keep_prob(ratio)

    def _set_stem(self):
        raise NotImplementedError()

    def _set_tail(self):
        raise NotImplementedError()

    def forward(self, x):
        curr_layer = self.stem(x)
        for cell_idx, cell in enumerate(self.cells):
            curr_layer = cell(curr_layer)

        curr_layer = self.last_conv(curr_layer)
        x = self.avgpool(curr_layer)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ValImageNet(ValNet):
    def __init__(self, alloc_code, scale=1.0, channel_dist=(16, 32, 64, 128, 256), num_classes=1000, input_size=224,
                 alloc_space=(1, 4, 4, 8, 4), cell_plan='super', alloc_plan='NR',
                 activation='relu', use_se=False):
        self.alloc_plan = alloc_plan
        cell_seq = {'NR': lambda x: "N" * x[0] + "R" +
                                    "N" * x[1] + "R" +
                                    "N" * x[2] + "R" +
                                    "N" * x[3] + "R" +
                                    "N" * x[4],
                    'NER': lambda x: "N" * x[0] + "R" +
                                     "N" * x[1] + "R" +
                                     "N" * x[2] + "R" +
                                     "N" * x[3] + "E" +
                                     "N" * x[4] + "R" +
                                     "N" * x[5] + "E",
                    }[alloc_plan](alloc_space)
        cell_seq = zip(cell_seq, alloc_code)
        cell = {'mb': MBValCell,
                'macmbg': MACMBGValCell,
                'macmb': MACMBValCell}[cell_plan]
        activ = {'relu': nn.ReLU,
                 'swish': Swish}[activation]

        super(ValImageNet, self).__init__(scale, channel_dist, num_classes, input_size, cell, cell_seq,
                                          activation=activ, use_se=use_se)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.SyncBatchNorm) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _set_stem(self):
        self.stem = nn.Sequential(nn.Conv2d(3, self.c[0], 3, stride=2, padding=1, bias=False),
                                  BN(self.c[0]),
                                  self.activation(inplace=True))
        self.curr_size = (self.input_size + 1) // 2

    def _set_tail(self):
        self.last_conv = nn.Sequential(nn.Conv2d(self.c[self.stage], 1024, kernel_size=1, bias=False),
                                       self.activation(inplace=True))

        self.avgpool = nn.AvgPool2d(self.curr_size)

        self.fc = nn.Linear(1024, self.num_classes)


class AMBNet(nn.Module):
    def __init__(self, scale, channel_dist, num_classes, input_size, Cell, cell_seq,
                 K=5):
        super(AMBNet, self).__init__()
        global BN

        BN = nn.BatchNorm2d

        self._time = torch.tensor([0.]).cuda()
        self._params = 0
        self._macs = 0
        self.K = K

        self.total_rounds = (Cell.candidate_num - 1) // (K - 1)
        self.scale = scale
        self.c = list(channel_dist[:1]) + [_make_divisible(ch * self.scale, 8) for ch in channel_dist[1:]]
        self.num_classes = num_classes
        self.input_size = input_size

        self._set_stem()

        self.cells = nn.ModuleList()
        _cells = cell_seq
        self.stage = 0
        self.arch_parameters = [torch.full((len(_cells), Cell.candidate_num), 0).cuda().requires_grad_()]

        for c in _cells:
            if c == 'N':
                stride = 1

                cout = cin = self.c[self.stage]
            elif c == 'E':
                self.stage += 1
                stride = 1

                cout = self.c[self.stage]
                cin = self.c[self.stage - 1]
            elif c == 'R':
                self.stage += 1
                stride = 2

                cout = self.c[self.stage]
                cin = self.c[self.stage - 1]
            else:
                raise NotImplementedError(f'unimplemented cell type: {c}')

            self.cells.append(Cell(cin, self.curr_size, stride, cout))
            self.curr_size = (self.curr_size - 1 + stride) // stride

        self._set_tail()

        self.netpara = sum(p.numel() for p in self.parameters()) / 1e6

        self.pending_space = []
        self.sub_space = []
        self.prev_best = [[] for _ in range(len(self.cells))]
        self.prev_bests = []

        for cell_idx, cell in enumerate(self.cells):
            curr_order = torch.tensor(np.random.permutation(cell.candidate_num)).cuda()
            dist.broadcast(curr_order, src=0)

            curr_order = curr_order.tolist()
            self.pending_space.append(curr_order[self.K:])
            self.sub_space.append(torch.tensor(curr_order[:self.K]).cuda())

    def _set_stem(self):
        raise NotImplementedError()

    def _set_tail(self):
        raise NotImplementedError()

    @staticmethod
    def _itemnize(index, gate_num, bin_sequnece):
        gates = bin_sequnece[gate_num]
        return [index[i].item() for i, gate in enumerate(gates) if gate == '1']

    @staticmethod
    def _itemnize2(index, x):
        if len(x) == 0:
            return []
        gates = list(map(lambda p: p.item(), list(x)))
        return [index[i].item() for i in gates]

    @staticmethod
    def _probnize(x, gate):
        if isinstance(gate, str) and gate == '1' or gate == 1.:
            return x
        return 1 - x

    @staticmethod
    def _comb_prob(prob_pool, gates):
        return torch.cumprod(torch.stack([AMBNet._probnize(prob_pool[i], gate) for i, gate in enumerate(gates)]),
                             dim=0)[-1]

    def get_resources(self, paths):
        latency, params, macs = 0, 0, 0
        for cell_idx, cell in enumerate(self.cells):
            curr_lat, curr_param, curr_mac = cell.resource(paths[cell_idx])
            latency += curr_lat
            params += curr_param
            macs += curr_mac
        return latency + self.get_back_time(), params + self.get_back_param(), macs + self.get_back_mac()

    def sampler(self, arch_update, balance, strict_prev, post_round, single_branch=False, arch_sample_balance=False,
                balance_s_rate=0.5):
        if not arch_update:
            if get_rank() == 0:
                paths = []
                for cell_idx, cell in enumerate(self.cells):
                    curr_sub_space = self.sub_space[cell_idx]
                    if len(curr_sub_space) == 0:
                        paths.append(torch.tensor([-1] * cell.candidate_num).cuda())
                    elif len(curr_sub_space) == 1 and not (cell.cin == cell.cout and cell.stride == 1):
                        paths.append(torch.tensor(list(curr_sub_space) + [-1] * (cell.candidate_num - 1)).cuda())
                    else:
                        if not balance:
                            branch_prob = torch.sigmoid(self.arch_parameters[0][cell_idx]) * 0.999 + 5e-4
                            sub_branch_space_prob = branch_prob.index_select(0, curr_sub_space)
                            if len(curr_sub_space) <= 5:  # two sample algorithms trade off by branch num
                                if cell.cin == cell.cout and cell.stride == 1:
                                    bin_sequnece = [f"{i:0{len(curr_sub_space)}b}" for i in
                                                    range(2 ** len(curr_sub_space))]
                                else:
                                    bin_sequnece = [f"{i:0{len(curr_sub_space)}b}" for i in
                                                    range(2 ** len(curr_sub_space))][1:]

                                sub_comb_probs = torch.stack(
                                    [self._comb_prob(sub_branch_space_prob, gates) for gates in bin_sequnece])
                                sample_path_combs = torch.multinomial(sub_comb_probs, 1, replacement=False)
                                path = self._itemnize(curr_sub_space, sample_path_combs, bin_sequnece)

                            else:
                                gate = sub_branch_space_prob.bernoulli()
                                while get_rank() == 0:
                                    gate = sub_branch_space_prob.bernoulli()
                                    if (cell.cin == cell.cout and cell.stride == 1) or sum(gate) != 0.:
                                        break
                                path = self._itemnize2(curr_sub_space, gate.nonzero())
                        else:
                            if single_branch:
                                demi_prob = torch.full((len(curr_sub_space) + 1,), balance_s_rate).cuda()
                                while get_rank() == 0:
                                    path = torch.multinomial(demi_prob, 1).item()
                                    path = [] if path == len(curr_sub_space) else [path]
                                    if (cell.cin == cell.cout and cell.stride == 1) or len(path) != 0:
                                        break
                            else:
                                demi_prob = torch.full((len(curr_sub_space),), balance_s_rate).cuda()
                                gate = demi_prob.bernoulli()
                                while get_rank() == 0:
                                    gate = demi_prob.bernoulli()
                                    if (cell.cin == cell.cout and cell.stride == 1) or sum(gate) != 0.:
                                        break
                                path = self._itemnize2(curr_sub_space, gate.nonzero())
                        paths.append(torch.tensor(path + [-1] * (cell.candidate_num - len(path))).cuda())

                paths = torch.stack(paths).cuda()
            else:  # dummy tensor
                paths = torch.full((len(self.cells), self.cells[0].candidate_num), -1).cuda().type(torch.int64)

            dist.broadcast(paths, src=0)
            paths = [path_set[path_set > -1].tolist() for path_set in paths]
            return paths
        else:
            if get_rank() == 0:
                strict_hit = random.randint(0, 1)
                paths = [[], []]
                for cell_idx, cell in enumerate(self.cells):
                    curr_sub_space = self.sub_space[cell_idx]

                    if len(curr_sub_space) == 0:
                        paths[0].append(torch.tensor([-1] * cell.candidate_num).cuda())
                        paths[1].append(torch.tensor([-1] * cell.candidate_num).cuda())
                        continue
                    elif len(curr_sub_space) == 1 and not (cell.cin == cell.cout and cell.stride == 1):
                        paths[0].append(torch.tensor(list(curr_sub_space) + [-1] * (cell.candidate_num - 1)).cuda())
                        paths[1].append(torch.tensor(list(curr_sub_space) + [-1] * (cell.candidate_num - 1)).cuda())
                        continue
                    if arch_sample_balance:
                        sub_branch_space_prob = torch.full((len(curr_sub_space),), 0.5).cuda()
                    else:
                        branch_prob = torch.sigmoid(self.arch_parameters[0][cell_idx]) * 0.999 + 5e-4
                        sub_branch_space_prob = branch_prob.index_select(0, curr_sub_space)

                    if len(
                            curr_sub_space) <= 5 and not arch_sample_balance:  # two sample algorithms trade off by branch num
                        if cell.cin == cell.cout and cell.stride == 1:
                            bin_sequnece = [f"{i:0{len(curr_sub_space)}b}" for i in range(2 ** len(curr_sub_space))]
                        else:
                            bin_sequnece = [f"{i:0{len(curr_sub_space)}b}" for i in
                                            range(2 ** len(curr_sub_space))][1:]

                        sub_comb_probs = torch.stack(
                            [self._comb_prob(sub_branch_space_prob, gates) for gates in bin_sequnece])
                        sample_path_combs = torch.multinomial(sub_comb_probs, 2, replacement=False)

                        if strict_prev and post_round and strict_hit:
                            path_str = '1' * len(self.prev_best[cell_idx]) + '0' * (
                                    len(curr_sub_space) - len(self.prev_best[cell_idx]))
                            assert len(path_str) == len(curr_sub_space)
                            if bin_sequnece[sample_path_combs[0]] != path_str and bin_sequnece[
                                sample_path_combs[1]] != path_str:
                                sample_path_combs[1] = bin_sequnece.index(path_str)
                                if 1:
                                    demi_sample_path_combs = sample_path_combs.clone()
                                    demi_sample_path_combs[0], demi_sample_path_combs[1] = sample_path_combs[1], \
                                                                                           sample_path_combs[0]
                                    sample_path_combs = demi_sample_path_combs
                            elif bin_sequnece[sample_path_combs[1]] == path_str:
                                demi_sample_path_combs = sample_path_combs.clone()
                                demi_sample_path_combs[0], demi_sample_path_combs[1] = sample_path_combs[1], \
                                                                                       sample_path_combs[0]
                                sample_path_combs = demi_sample_path_combs
                                print('suboptimal hit')

                        path_selected, path_unselected = (
                            self._itemnize(curr_sub_space, sample_path_combs[0], bin_sequnece),
                            self._itemnize(curr_sub_space, sample_path_combs[1], bin_sequnece))
                    else:
                        while 1:
                            gate_1 = sub_branch_space_prob.bernoulli()
                            if (cell.cin == cell.cout and cell.stride == 1) or sum(gate_1) != 0.:
                                break
                        while 1:
                            gate_2 = sub_branch_space_prob.bernoulli()
                            if ((cell.cin == cell.cout and cell.stride == 1) or sum(
                                    gate_2) != 0.) and not torch.all(gate_1 == gate_2):
                                break

                        if strict_prev and post_round and strict_hit:
                            path_str = torch.tensor([1] * len(self.prev_best[cell_idx]) + [0] * (
                                    len(curr_sub_space) - len(self.prev_best[cell_idx]))).float().cuda()
                            assert len(path_str) == len(curr_sub_space)
                            if not torch.all(gate_1 == path_str) and not torch.all(gate_2 == path_str):
                                gate_2 = path_str
                                if 1:
                                    gate_1, gate_2 = gate_2, gate_1
                            elif torch.all(gate_2 == path_str):
                                gate_1, gate_2 = gate_2, gate_1
                                print('suboptimal hit')

                        path_selected, path_unselected = (self._itemnize2(curr_sub_space, gate_1.nonzero()),
                                                          self._itemnize2(curr_sub_space, gate_2.nonzero()))
                    paths[0].append(
                        torch.tensor(path_selected + [-1] * (cell.candidate_num - len(path_selected))).cuda())
                    paths[1].append(
                        torch.tensor(path_unselected + [-1] * (cell.candidate_num - len(path_unselected))).cuda())
                paths = torch.stack([torch.stack(paths[0]).cuda(), torch.stack(paths[1]).cuda()])
            else:  # dummy tensor
                paths = torch.full((2, len(self.cells), self.cells[0].candidate_num), -1).cuda().type(torch.int64)
            dist.broadcast(paths, src=0)

            paths_selected, paths_unselected = ([path_set[path_set > -1].tolist() for path_set in paths[0]],
                                                [path_set[path_set > -1].tolist() for path_set in paths[1]])

            probs_selected, probs_unselected = [], []  # re-calculate probabilities
            for cell_idx, cell in enumerate(self.cells):
                curr_sub_space = self.sub_space[cell_idx]
                if len(curr_sub_space) == 0 or (
                        len(curr_sub_space) == 1 and not (cell.cin == cell.cout and cell.stride == 1)):
                    probs_selected.append(torch.tensor(-1.).cuda()), probs_unselected.append(torch.tensor(-1.).cuda())
                    continue
                branch_prob = torch.sigmoid(self.arch_parameters[0][cell_idx]) * 0.999 + 5e-4
                sub_branch_space_prob = branch_prob.index_select(0, curr_sub_space)
                gate_selected, gate_unselected = ([a in paths_selected[cell_idx] for a in curr_sub_space],
                                                  [a in paths_unselected[cell_idx] for a in curr_sub_space])
                prob_selected, prob_unselected = (self._comb_prob(sub_branch_space_prob, gate_selected),
                                                  self._comb_prob(sub_branch_space_prob, gate_unselected))
                probs_selected.append(prob_selected), probs_unselected.append(prob_unselected)

            return paths_selected, paths_unselected, probs_selected, probs_unselected

    def pin_paths(self):
        self.paths_pin = self.paths_batch.pop(0)
        self.paths_batch.append(self.paths_pin)

    def get_next_pin_paths(self):
        return self.paths_batch[0]

    def get_next_prev_pin_paths(self):
        return self.prev_bests[0]

    def pin_prev_paths(self):
        self.prev_paths_pin = self.prev_bests.pop(0)
        self.prev_bests.append(self.prev_paths_pin)

    def set_front(self, front):
        self.prev_best = []
        self.prev_bests = list(compress(self.paths_batch + self.prev_bests, front))
        for cell_idx, _ in enumerate(self.cells):
            self.prev_best.append(list(set([ele for prev_best in self.prev_bests for ele in prev_best[cell_idx]])))

    def batch_sampler(self, num, mean, drift, extra_num, extra_drift, latency=False, non_alpha=False,
                      single_branch=False, balance_s_rate=0.5):
        self.paths_batch = []
        for i in range(num):
            while 1:
                paths_sample = self.sampler(arch_update=False, balance=non_alpha, strict_prev=False, post_round=False,
                                            single_branch=single_branch, balance_s_rate=balance_s_rate)

                if paths_sample not in self.paths_batch:
                    lat, param, mac = self.get_resources(paths_sample)
                    if latency:
                        cres = lat
                    else:
                        cres = mac
                    if mean - drift < cres < mean + drift:
                        self.paths_batch.append(paths_sample)
                        assert paths_sample in self.paths_batch
                        if get_rank() == 0:
                            print(f'sample hit {i}/{num}, mac: {mac}, lat: {lat}')
                        break
                    else:
                        if get_rank() == 0:
                            print(f'sample hit fail, mac: {mac}, lat: {lat}')
        for i in range(extra_num):
            while 1:
                paths_sample = self.sampler(arch_update=False, balance=non_alpha, strict_prev=False, post_round=False,
                                            single_branch=single_branch, balance_s_rate=balance_s_rate)
                if paths_sample not in self.paths_batch:
                    lat, param, mac = self.get_resources(paths_sample)
                    if latency:
                        cres = lat
                    else:
                        cres = mac
                    if mean - drift - extra_drift < cres <= mean - drift:
                        self.paths_batch.append(paths_sample)
                        assert paths_sample in self.paths_batch
                        if get_rank() == 0:
                            print(f'extra sample hit {i}/{extra_num}, mac: {mac}, lat: {lat}')
                        break
                    else:
                        if get_rank() == 0:
                            print(f'extra sample hit fail, mac: {mac}, lat: {lat}')

    def forward(self, x, arch_update=False, balance=False, prev_val=False, bulk_val=False, round=0, strict_prev=False,
                arch_sample_balance=False, balance_s_rate=0.5):
        if not arch_update:
            latency, params, macs = 0, 0, 0
            curr_layer = self.stem(x)
            if bulk_val:
                paths = self.paths_pin
            elif prev_val:
                paths = self.prev_paths_pin
            else:
                paths = self.sampler(arch_update, balance, strict_prev, round > 0, balance_s_rate=balance_s_rate)

            for cell_idx, cell in enumerate(self.cells):
                if len(paths[cell_idx]) == 0:
                    continue
                curr_layer, curr_lat, curr_param, curr_mac = cell(curr_layer, paths[cell_idx])
                latency += curr_lat
                params += curr_param
                macs += curr_mac

            curr_layer = self.last_conv(curr_layer)
            x = self.avgpool(curr_layer)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

            return x, latency, params, macs

        else:
            latency, params, macs = 0, 0, 0
            curr_layer = self.stem(x)
            paths_selected, paths_unselected, probs_selected, probs_unselected = self.sampler(arch_update,
                                                                                              balance,
                                                                                              strict_prev,
                                                                                              round > 0,
                                                                                              arch_sample_balance=arch_sample_balance)
            for cell_idx, cell in enumerate(self.cells):
                path_selected, path_unselected = paths_selected[cell_idx], paths_unselected[cell_idx]
                prob_selected, prob_unselected = probs_selected[cell_idx], probs_unselected[cell_idx]
                if prob_selected < 0:
                    if len(path_selected) == 0:
                        continue
                    else:
                        curr_layer, latency_selected, param_selected, mac_selected = cell(curr_layer, path_selected)
                        latency += latency_selected
                        params += param_selected
                        macs += mac_selected
                        continue
                scale_factor = 1 / (prob_selected + prob_unselected)
                prob_selected, prob_unselected = prob_selected * scale_factor, prob_unselected * scale_factor

                selected_layer, latency_selected, param_selected, mac_selected = cell(curr_layer, path_selected)
                with torch.no_grad():
                    unselected_layer, latency_unselected, param_unselected, mac_unselected = cell(curr_layer,
                                                                                                  path_unselected)

                curr_layer = selected_layer * (prob_selected + 1 - prob_selected.detach()) + unselected_layer * (
                        prob_unselected - prob_unselected.detach())

                latency += latency_selected * prob_selected + latency_unselected * prob_unselected
                params += param_selected * prob_selected + param_unselected * prob_unselected
                macs += mac_selected * prob_selected + mac_unselected * prob_unselected

            curr_layer = self.last_conv(curr_layer)
            x = self.avgpool(curr_layer)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

            return x, latency, params, macs

    def get_back_time(self):
        return self._time

    def get_back_param(self):
        return self._params

    def get_back_mac(self):
        return self._macs

    def re_init(self):
        self.arch_parameters = [torch.full_like(self.arch_parameters[0], 0).cuda().requires_grad_()]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.SyncBatchNorm) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def update_space(self):
        for cell_idx, cell in enumerate(self.cells):
            new_sub_space = copy.deepcopy(self.prev_best[cell_idx]) if len(self.prev_best) > 0 else []
            for _ in range(min(self.K - len(self.prev_best[cell_idx]),
                               len(self.pending_space[cell_idx]))):
                new_sub_space.append(self.pending_space[cell_idx].pop(0))

            self.sub_space[cell_idx] = torch.tensor(new_sub_space).cuda()

    def clean_space(self, threshold, preserve=True):
        for cell_idx, cell in enumerate(self.cells):
            curr_sub_space = self.sub_space[cell_idx]
            if len(curr_sub_space) == 0:
                continue
            branch_alpha = self.arch_parameters[0][cell_idx]
            sub_branch_space_alpha = branch_alpha.index_select(0, curr_sub_space)

            new_sub_space = [curr_sub_space[i].item() for i in range(len(curr_sub_space))
                             if sub_branch_space_alpha[i] >= threshold or
                             (not (cell.cin == cell.cout and cell.stride == 1) and len(curr_sub_space) == 2) or
                             (len(self.prev_best) > 0 and
                              preserve and
                              len(self.prev_best[cell_idx]) > 0 and
                              curr_sub_space[i].item() in list(self.prev_best[cell_idx]))]

            self.sub_space[cell_idx] = torch.tensor(new_sub_space).cuda()


class AMBImageNet(AMBNet):
    def __init__(self, scale=1.0, channel_dist=(16, 32, 64, 128, 256), num_classes=1000, input_size=224,
                 alloc_space=(1, 4, 4, 8, 4), cell_plan='super', alloc_plan='NR', K=5):

        cell_seq = {'NR': lambda x: "N" * x[0] + "R" +
                                    "N" * x[1] + "R" +
                                    "N" * x[2] + "R" +
                                    "N" * x[3] + "R" +
                                    "N" * x[4],
                    'NER': lambda x: "N" * x[0] + "R" +
                                     "N" * x[1] + "R" +
                                     "N" * x[2] + "R" +
                                     "N" * x[3] + "E" +
                                     "N" * x[4] + "R" +
                                     "N" * x[5] + "E",
                    }[alloc_plan](alloc_space)
        cell = {'multi_branch': MBCell,
                'multi_mac': MACMBCell,
                'multi_mac_group': GMACMBCell}[cell_plan]

        super(AMBImageNet, self).__init__(scale, channel_dist, num_classes, input_size, cell, cell_seq,
                                          K=K)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.SyncBatchNorm) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _set_stem(self):
        self.stem = nn.Sequential(nn.Conv2d(3, self.c[0], 3, stride=2, padding=1, bias=False),
                                  BN(self.c[0]),
                                  nn.ReLU(inplace=True))

        self.curr_size = (self.input_size + 1) // 2
        if self.c[0] == 32:
            self._time += lookup(1, 224, 3, self.c[0], 3, 2, 1, 1, 1)
        self._params += sum(p.numel() for p in self.stem.parameters()) / 1e6
        self._macs += mac_of_convolution(self.stem[0].weight,
                                         torch.ones(1, 3, 224, 224),
                                         torch.ones(1, self.c[0], 112, 112))

    def _set_tail(self):
        self.last_conv = nn.Sequential(nn.Conv2d(self.c[self.stage], 1024, kernel_size=1, bias=False),
                                       nn.ReLU(inplace=True))

        if self.c[self.stage] == 256:
            self._time += lookup(1, self.curr_size, self.c[self.stage], 1024, 1, 1, 0, 1, 1)
        self._params += sum(p.numel() for p in self.last_conv.parameters()) / 1e6
        self._macs += mac_of_convolution(self.last_conv[0].weight,
                                         torch.ones(1, self.c[self.stage], 7, 7),
                                         torch.ones(1, 1024, 7, 7))

        self.avgpool = nn.AvgPool2d(self.curr_size)
        if self.c[self.stage] == 256:
            self._time += lookup(3, self.curr_size, 1024, 1024, self.curr_size, 1, 0, 1, 1)
        self._params += sum(p.numel() for p in self.avgpool.parameters()) / 1e6
        self._macs += mac_of_avg_pool_or_mean(torch.ones(1, 1024, 1, 1))

        self.fc = nn.Linear(1024, self.num_classes)
        if self.c[self.stage] == 256:
            self._time += lookup(5, 1, 1024, self.num_classes, 0, 0, 0, 0, 0)
        self._params += sum(p.numel() for p in self.fc.parameters()) / 1e6
        self._macs += mac_of_addmm(torch.ones(1, 1000),
                                   self.fc.weight)


def NSENet27(**kwargs):
    model = ValImageNet(
        [[0], [6], [3], [0], [16], [3], [9], [11, 17], [9, 2], [12], [1, 6], [9, 12], [9], [17, 14], [21], [3], [13],
         [19, 3, 14]],
        scale=1.04,
        channel_dist=[16, 24, 40, 80, 96, 192, 320],
        alloc_space=[1, 2, 2, 2, 2, 3],
        cell_plan='macmb',
        alloc_plan='NER',
        **kwargs)
    return model


def NSENet(**kwargs):
    model = ValImageNet(
        [[0], [34], [3], [52], [35], [38], [17, 11, 49], [41], [35], [12], [1, 10, 6], [9], [0, 9], [35, 52, 61],
         [10, 32], [3, 57], [13], [3, 19, 44]],
        scale=1.045,
        channel_dist=[16, 24, 40, 80, 96, 192, 320],
        alloc_space=[1, 1, 2, 3, 2, 3],
        cell_plan='macmbg',
        alloc_plan='NER',
        **kwargs)
    return model


def NSENet_GPU(**kwargs):
    model = ValImageNet(
        [[15], [11], [1, 17], [16, 1], [11], [11], [2, 7], [3], [1, 6], [11], [11, 6], [8, 2], [0, 1, 8, 11],
         [2, 6, 8, 12, 14]],
        scale=1.0,
        channel_dist=[16, 32, 64, 128, 256],
        alloc_space=[0, 1, 3, 4, 2],
        cell_plan='mb',
        alloc_plan='NR',
        **kwargs)
    return model


def print_debug(*args, **kw):
    return
    if get_rank() == 0:
        print(*args, **kw)
