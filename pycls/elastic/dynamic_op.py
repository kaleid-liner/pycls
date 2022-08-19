# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.nn.parameter import Parameter

from .utils import (
    sub_filter_start_end,
    get_same_padding,
    make_divisible,
)

from pycls.core.config import cfg


class DynamicSeparableConv2d(nn.Module):
    KERNEL_TRANSFORM_MODE = 1  # None or 1

    def __init__(self, max_w_in, kernel_size_list, stride=1, dilation=1):
        super(DynamicSeparableConv2d, self).__init__()

        self.max_w_in = max_w_in
        self.kernel_size_list = kernel_size_list
        self.stride = stride
        self.dilation = dilation

        self.conv = nn.Conv2d(
            self.max_w_in,
            self.max_w_in,
            max(self.kernel_size_list),
            self.stride,
            groups=self.max_w_in,
            bias=False,
        )

        self._ks_set = list(set(self.kernel_size_list))
        self._ks_set.sort()  # e.g., [3, 5, 7]
        if self.KERNEL_TRANSFORM_MODE is not None:
            # register scaling parameters
            # 7to5_matrix, 5to3_matrix
            scale_params = {}
            for i in range(len(self._ks_set) - 1):
                ks_small = self._ks_set[i]
                ks_larger = self._ks_set[i + 1]
                param_name = "%dto%d" % (ks_larger, ks_small)
                # noinspection PyArgumentList
                scale_params["%s_matrix" % param_name] = Parameter(
                    torch.eye(ks_small ** 2)
                )
            for name, param in scale_params.items():
                self.register_parameter(name, param)

        self.active_kernel_size = max(self.kernel_size_list)

    def get_active_filter(self, w_in, kernel_size):
        w_out = w_in
        max_kernel_size = max(self.kernel_size_list)

        start, end = sub_filter_start_end(max_kernel_size, kernel_size)
        filters = self.conv.weight[:w_out, :w_in, start:end, start:end]
        if self.KERNEL_TRANSFORM_MODE is not None and kernel_size < max_kernel_size:
            start_filter = self.conv.weight[
                :w_out, :w_in, :, :
            ]  # start with max kernel
            for i in range(len(self._ks_set) - 1, 0, -1):
                src_ks = self._ks_set[i]
                if src_ks <= kernel_size:
                    break
                target_ks = self._ks_set[i - 1]
                start, end = sub_filter_start_end(src_ks, target_ks)
                _input_filter = start_filter[:, :, start:end, start:end]
                _input_filter = _input_filter
                _input_filter = _input_filter.view(
                    _input_filter.size(0), _input_filter.size(1), -1
                )
                _input_filter = _input_filter.view(-1, _input_filter.size(2))
                _input_filter = F.linear(
                    _input_filter,
                    self.__getattr__("%dto%d_matrix" % (src_ks, target_ks)),
                )
                _input_filter = _input_filter.view(
                    filters.size(0), filters.size(1), target_ks ** 2
                )
                _input_filter = _input_filter.view(
                    filters.size(0), filters.size(1), target_ks, target_ks
                )
                start_filter = _input_filter
            filters = start_filter
        return filters

    def forward(self, x, kernel_size=None):
        if kernel_size is None:
            kernel_size = self.active_kernel_size
        w_in = x.size(1)

        filters = self.get_active_filter(w_in, kernel_size)

        padding = get_same_padding(kernel_size)
        y = F.conv2d(x, filters, None, self.stride, padding, self.dilation, w_in)
        return y


class DynamicConv2d(nn.Module):
    def __init__(
        self, max_w_in, max_w_out, kernel_size=1, stride=1, dilation=1
    ):
        super(DynamicConv2d, self).__init__()

        self.max_w_in = max_w_in
        self.max_w_out = max_w_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

        self.conv = nn.Conv2d(
            self.max_w_in,
            self.max_w_out,
            self.kernel_size,
            stride=self.stride,
            bias=False,
        )

        self.active_w_out = self.max_w_out

    def get_active_filter(self, w_out, w_in):
        return self.conv.weight[:w_out, :w_in, :, :]

    def forward(self, x, w_out=None):
        if w_out is None:
            w_out = self.active_w_out
        w_in = x.size(1)
        filters = self.get_active_filter(w_out, w_in)

        padding = get_same_padding(self.kernel_size)
        y = F.conv2d(x, filters, None, self.stride, padding, self.dilation, 1)
        return y


class DynamicGroupConv2d(nn.Module):
    def __init__(
        self,
        w_in,
        w_out,
        kernel_size_list,
        groups_list,
        stride=1,
        dilation=1,
    ):
        super(DynamicGroupConv2d, self).__init__()

        self.w_in = w_in
        self.w_out = w_out
        self.kernel_size_list = kernel_size_list
        self.groups_list = groups_list
        self.stride = stride
        self.dilation = dilation

        self.conv = nn.Conv2d(
            self.w_in,
            self.w_out,
            max(self.kernel_size_list),
            self.stride,
            groups=min(self.groups_list),
            bias=False,
        )

        self.active_kernel_size = max(self.kernel_size_list)
        self.active_groups = min(self.groups_list)

    def get_active_filter(self, kernel_size, groups):
        start, end = sub_filter_start_end(max(self.kernel_size_list), kernel_size)
        filters = self.conv.weight[:, :, start:end, start:end]

        sub_filters = torch.chunk(filters, groups, dim=0)
        sub_w_in = self.w_in // groups
        sub_ratio = filters.size(1) // sub_w_in

        filter_crops = []
        for i, sub_filter in enumerate(sub_filters):
            part_id = i % sub_ratio
            start = part_id * sub_w_in
            filter_crops.append(sub_filter[:, start : start + sub_w_in, :, :])
        filters = torch.cat(filter_crops, dim=0)
        return filters

    def forward(self, x, kernel_size=None, groups=None):
        if kernel_size is None:
            kernel_size = self.active_kernel_size
        if groups is None:
            groups = self.active_groups

        filters = self.get_active_filter(kernel_size, groups)
        padding = get_same_padding(kernel_size)
        y = F.conv2d(
            x,
            filters,
            None,
            self.stride,
            padding,
            self.dilation,
            groups,
        )
        return y

        
class SuperDynamicGroupConv2d(nn.Module):
    def __init__(
        self,
        max_w_in,
        max_w_out,
        max_kernel_size,
        min_groups,
        stride=1,
        dilation=1,
    ):
        super(SuperDynamicGroupConv2d, self).__init__()

        self.max_w_in = max_w_in
        self.max_w_out = max_w_out
        self.max_kernel_size = max_kernel_size
        self.min_groups = min_groups
        self.stride = stride
        self.dilation = dilation

        self.conv = nn.Conv2d(
            self.max_w_in,
            self.max_w_out,
            self.max_kernel_size,
            self.stride,
            groups=self.min_groups,
            bias=False,
        )

        self.active_w_out = max_w_out
        self.active_kernel_size = max_kernel_size
        self.active_groups = self.min_groups

    def get_active_filter(self, w_out, w_in, kernel_size, groups):
        start, end = sub_filter_start_end(self.max_kernel_size, kernel_size)
        filters = self.conv.weight[:, :, start:end, start:end]

        sub_filters = torch.chunk(filters, groups, dim=0)
        sub_w_out = w_out // groups
        sub_w_in = w_in // groups
        sub_ratio = filters.size(1) // sub_w_in

        filter_crops = []
        for i, sub_filter in enumerate(sub_filters):
            part_id = i % sub_ratio
            start = part_id * sub_w_in
            filter_crops.append(sub_filter[:sub_w_out, :sub_w_in, :, :])
        filters = torch.cat(filter_crops, dim=0)
        return filters

    def forward(self, x, w_out=None, kernel_size=None, groups=None):
        if w_out is None:
            w_out = self.active_w_out
        if kernel_size is None:
            kernel_size = self.active_kernel_size
        if groups is None:
            groups = self.active_groups

        w_in = x.size(1)
        filters = self.get_active_filter(w_out, w_in, kernel_size, groups)
        padding = get_same_padding(kernel_size)
        y = F.conv2d(
            x,
            filters,
            None,
            self.stride,
            padding,
            self.dilation,
            groups,
        )
        return y


class DynamicBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features_list):
        self.num_features_max = max(num_features_list)
        super().__init__(
            self.num_features_max, affine=True, track_running_stats=False)
        # for tracking performance during training
        self.tracked_widths = [num_features_list[-1], num_features_list[0]] + num_features_list[1:-1]
        self.bn = nn.ModuleList([nn.BatchNorm2d(w, affine=False) for w in self.tracked_widths])

    def forward(self, input):
        c = input.size(1)
        if c in self.tracked_widths:
            idx = self.tracked_widths.index(c)
            y = nn.functional.batch_norm(
                input,
                self.bn[idx].running_mean[:c],
                self.bn[idx].running_var[:c],
                self.weight[:c],
                self.bias[:c],
                self.bn[idx].training,
                self.momentum,
                self.eps)
        else:
            y = nn.functional.batch_norm(
                input,
                self.running_mean,
                self.running_var,
                self.weight[:c],
                self.bias[:c],
                self.training,
                self.momentum,
                self.eps)
        return y


class DynamicGroupNorm(nn.GroupNorm):
    def __init__(
        self, num_groups, num_channels, eps=1e-5, affine=True, channel_per_group=None
    ):
        super(DynamicGroupNorm, self).__init__(num_groups, num_channels, eps, affine)
        self.channel_per_group = channel_per_group

    def forward(self, x):
        n_channels = x.size(1)
        n_groups = n_channels // self.channel_per_group
        return F.group_norm(
            x, n_groups, self.weight[:n_channels], self.bias[:n_channels], self.eps
        )

    @property
    def bn(self):
        return self


class DynamicLinear(nn.Module):
    def __init__(self, max_in_features, max_out_features, bias=True):
        super(DynamicLinear, self).__init__()

        self.max_in_features = max_in_features
        self.max_out_features = max_out_features
        self.bias = bias

        self.linear = nn.Linear(self.max_in_features, self.max_out_features, self.bias)

        self.active_out_features = self.max_out_features

    def get_active_weight(self, out_features, in_features):
        return self.linear.weight[:out_features, :in_features]

    def get_active_bias(self, out_features):
        return self.linear.bias[:out_features] if self.bias else None

    def forward(self, x, out_features=None):
        if out_features is None:
            out_features = self.active_out_features

        in_features = x.size(1)
        weight = self.get_active_weight(out_features, in_features)
        bias = self.get_active_bias(out_features)
        y = F.linear(x, weight, bias)
        return y
