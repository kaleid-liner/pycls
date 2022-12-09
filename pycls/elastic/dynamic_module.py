import torch.nn as nn
from .dynamic_op import (
    DynamicConv2d,
    DynamicBatchNorm2d,
    SuperDynamicGroupConv2d,
    DynamicLinear,
)
from pycls.models.blocks import activation, SE, norm2d, gap2d, linear, pool2d
from pycls.models.blocks import (
    conv2d_cx,
    gap2d_cx,
    linear_cx,
    norm2d_cx,
    pool2d_cx,
)
from .ir_gen import (
    conv2d_ir,
    gap2d_ir,
    linear_ir,
    se_ir,
)
from .utils import make_divisible


class DynamicModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.active_w_out = 0

    def set_active_subnet(self, w, **kwargs):
        self.active_w_out = w
        for module in self.modules():
            if module != self and isinstance(module, DynamicModule):
                module.set_active_subnet(w, **kwargs)


class DynamicSE(DynamicModule):
    """Squeeze-and-Excitation (SE) block: AvgPool, FC, Act, FC, Sigmoid."""

    def __init__(self, max_w_in, se_r):
        super().__init__()
        self.se_r = se_r
        max_w_se = make_divisible(max_w_in * self.se_r, 8)

        self.avg_pool = gap2d(max_w_in)

        self.a = DynamicConv2d(max_w_in, max_w_se, bias=True)
        self.a_af = activation()
        self.b = DynamicConv2d(max_w_se, max_w_in, bias=True)
        self.b_af = nn.Sigmoid()
        
    def forward(self, x):
        w_in = x.size(1)
        active_w_se = int(round(int(w_in) * self.se_r))
        active_w_se = make_divisible(w_in * self.se_r, 8)

        self.a.active_w_out = active_w_se
        self.b.active_w_out = w_in

        return x * self.b_af(self.b(self.a_af(self.a(self.avg_pool(x)))))


class DynamicResBottleneckBlock(DynamicModule):

    def __init__(self, w_in_list, w_out_list, stride, params):
        super().__init__()
        self.proj, self.bn = None, None

        max_w_in = max(w_in_list)
        max_w_out = max(w_out_list)

        if (max_w_in != max_w_out) or (stride != 1):
            self.proj = DynamicConv2d(max_w_in, max_w_out, stride=stride)
            self.bn = DynamicBatchNorm2d(w_out_list)

        self.f = DynamicBottleneckTransform(w_in_list, w_out_list, stride, params)
        self.af = activation()

        self.active_w_out = max_w_out

    def forward(self, x):
        if self.proj:
            self.proj.active_w_out = self.active_w_out
        x_p = self.bn(self.proj(x)) if self.proj else x
        return self.af(x_p + self.f(x))

    @staticmethod
    def complexity(cx, w_in, w_out, stride, params):
        if (w_in != w_out) or (stride != 1):
            h, w = cx["h"], cx["w"]
            cx = conv2d_cx(cx, w_in, w_out, 1, stride=stride)
            cx = norm2d_cx(cx, w_out)
            cx["h"], cx["w"] = h, w
        cx = DynamicBottleneckTransform.complexity(cx, w_in, w_out, stride, params)
        return cx

    @staticmethod
    def gen_ir(ir, hw, w_in, w_out, stride, params):
        if (w_in != w_out) or (stride != 1):
            ir, _ = conv2d_ir(ir, hw, w_in, w_out, 1, stride, type="conv-bn-relu")
        ir, hw = DynamicBottleneckTransform.gen_ir(ir, hw, w_in, w_out, stride, params)
        return ir, hw
        

class DynamicBottleneckTransform(DynamicModule):
    """Bottleneck transformation: 1x1, 3x3 [+SE], 1x1."""

    def __init__(self, w_in_list, w_out_list, stride, params):
        super().__init__()

        max_w_in = max(w_in_list)
        max_w_out = max(w_out_list)

        bot_mul = params["bot_mul"]
        self.se_r = params["se_r"]
        w_b_list = [int(round(w_out * bot_mul)) for w_out in w_out_list]
        max_w_b = max(w_b_list)
        min_groups = max_w_b // params["group_w"]

        self.a = DynamicConv2d(max_w_in, max_w_b)
        self.a_bn = DynamicBatchNorm2d(w_b_list)
        self.a_af = activation()
        if params.get("vanilla_conv", False):
            self.b = DynamicConv2d(max_w_b, max_w_b, 3, stride=stride)
        else:
            self.b = SuperDynamicGroupConv2d(max_w_b, max_w_b, 3, stride=stride, min_groups=min_groups)
        self.b_bn = DynamicBatchNorm2d(w_b_list)
        self.b_af = activation()
        self.se = DynamicSE(max_w_b, self.se_r) if self.se_r else None
        self.c = DynamicConv2d(max_w_b, max_w_out)
        self.c_bn = DynamicBatchNorm2d(w_out_list)
        self.c_bn.final_bn = True

        self.active_w_out = max_w_out
        self.active_bot_mul = bot_mul
        self.active_group_w = params["group_w"]

    def forward(self, x):
        active_w_b = int(round(self.active_w_out * self.active_bot_mul))
        self.a.active_w_out = active_w_b
        self.b.active_w_out = active_w_b
        self.b.active_groups = active_w_b // self.active_group_w
        self.c.active_w_out = self.active_w_out

        for layer in self.children():
            x = layer(x)
        return x

    def set_active_subnet(self, w, **kwargs):
        super().set_active_subnet(w, **kwargs)
        self.active_group_w = kwargs["g"]
        self.active_bot_mul = kwargs["b"]

    @staticmethod
    def complexity(cx, w_in, w_out, stride, params):
        w_b = int(round(w_out * params["bot_mul"]))
        w_se = int(round(w_in * params["se_r"]))
        groups = w_b // params["group_w"]
        cx = conv2d_cx(cx, w_in, w_b, 1)
        cx = norm2d_cx(cx, w_b)
        cx = conv2d_cx(cx, w_b, w_b, 3, stride=stride, groups=groups)
        cx = norm2d_cx(cx, w_b)
        cx = SE.complexity(cx, w_b, w_se) if w_se else cx
        cx = conv2d_cx(cx, w_b, w_out, 1)
        cx = norm2d_cx(cx, w_out)
        return cx

    @staticmethod
    def gen_ir(ir, hw, w_in, w_out, stride, params):
        w_b = int(round(w_out * params["bot_mul"]))
        w_se = int(round(w_in * params["se_r"]))
        groups = params["groups"]
        ir, hw = conv2d_ir(ir, hw, w_in, w_b, 1, type="conv-bn-relu")
        ir, hw = conv2d_ir(ir, hw, w_b, w_b, 3, stride, groups, type="conv-bn-relu")
        ir, hw = se_ir(ir, hw, w_b, w_se) if w_se else (ir, hw)
        ir, hw = conv2d_ir(ir, hw, w_b, w_out, 1, type="conv-bn")
        return ir, hw


class DynamicSimpleStem(DynamicModule):
    """Simple stem for ImageNet: 3x3, BN, AF."""

    def __init__(self, w_in, w_out_list):
        super().__init__()
        max_w_out = max(w_out_list)
        self.conv = DynamicConv2d(w_in, max_w_out, 3, stride=2)
        self.bn = DynamicBatchNorm2d(w_out_list)
        self.af = activation()

        self.active_w_out = max_w_out

    def forward(self, x):
        self.conv.active_w_out = self.active_w_out

        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = conv2d_cx(cx, w_in, w_out, 3, stride=2)
        cx = norm2d_cx(cx, w_out)
        return cx

    @staticmethod
    def gen_ir(ir, hw, w_in, w_out):
        ir, hw = conv2d_ir(ir, hw, w_in, w_out, 3, 2, type="conv-bn-relu")
        return ir, hw


class DynamicResStem(DynamicModule):
    """ResNet stem for ImageNet: 7x7, BN, AF, MaxPool."""

    def __init__(self, w_in, w_out_list):
        super().__init__()
        max_w_out = max(w_out_list)
        self.conv = DynamicConv2d(w_in, max_w_out, 7, stride=2)
        self.bn = DynamicBatchNorm2d(w_out_list)
        self.af = activation()
        self.pool = pool2d(max_w_out, 3, stride=2)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = conv2d_cx(cx, w_in, w_out, 7, stride=2)
        cx = norm2d_cx(cx, w_out)
        cx = pool2d_cx(cx, w_out, 3, stride=2)
        return cx


class DynamicAnyHead(DynamicModule):
    """AnyNet head: optional conv, AvgPool, 1x1."""

    def __init__(self, w_in_list, head_width, num_classes):
        super().__init__()
        # omit head width
        max_w_in = max(w_in_list)
        self.avg_pool = gap2d(max_w_in)
        self.fc = DynamicLinear(max_w_in, num_classes, bias=True)

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    @staticmethod
    def complexity(cx, w_in, head_width, num_classes):
        if head_width > 0:
            cx = conv2d_cx(cx, w_in, head_width, 1)
            cx = norm2d_cx(cx, head_width)
            w_in = head_width
        cx = gap2d_cx(cx, w_in)
        cx = linear_cx(cx, w_in, num_classes, bias=True)
        return cx

    @staticmethod
    def gen_ir(ir, hw, w_in, head_width, num_classes):
        if head_width > 0:
            ir, hw = conv2d_ir(ir, hw, w_in, head_width, 1, type="conv-bn-relu")
            w_in = head_width
        ir, hw = gap2d_ir(ir, hw, w_in)
        ir, hw = linear_ir(ir, hw, w_in, num_classes)
        return ir, hw
