from pycls.core.config import cfg

import keras.layers as layers
import tensorflow as tf

MP_START = "mp_start"


def res_bottleneck_block(x, w_in, w_out, stride, name, params, mp_start=False):
    if (w_in != w_out) or (stride != 1):
        x_proj = layers.Conv2D(w_out, 1, stride, padding="same", name=name + "_proj_conv")(x)
        x_proj = layers.BatchNormalization(axis=3, momentum=cfg.BN.MOM, epsilon=cfg.BN.EPS, name=name + "_proj_bn")(x_proj)
    else:
        x_proj = x

    x = bottleneck_transform(x, w_in, w_out, stride, name, params)
    name = "{}_{}".format(name, MP_START) if mp_start else name
    x = layers.Add(name=name + "_add")([x_proj, x])
    x = layers.ReLU(name=name + "_relu")(x)
    return x


def bottleneck_transform(x, w_in, w_out, stride, name, params, mp_start=False):
    w_b = int(round(w_out * params["bot_mul"]))
    w_se = int(round(w_in * params["se_r"]))
    groups = w_b // params["group_w"]
    x = layers.Conv2D(w_b, 1, padding="same", name=name + "_a_conv")(x)
    x = layers.BatchNormalization(axis=3, momentum=cfg.BN.MOM, epsilon=cfg.BN.EPS, name=name + "_a_bn")(x)
    x = layers.ReLU(name=name + "_a_relu")(x)
    x = group_conv(x, w_b, 3, strides=stride, padding="same", groups=groups, name=name + "_b_conv")
    x = layers.BatchNormalization(axis=3, momentum=cfg.BN.MOM, epsilon=cfg.BN.EPS, name=name + "_b_bn")(x)
    x = layers.ReLU(name=name + "_b_relu")(x)
    name = "{}_{}".format(name, MP_START) if mp_start else name
    x = layers.Conv2D(w_out, 1, padding="same", name=name + "_c_conv")(x)
    x = layers.BatchNormalization(axis=3, momentum=cfg.BN.MOM, epsilon=cfg.BN.EPS, name=name + "_c_bn")(x)
    return x


def group_conv(x, filters, kernel_size, strides, padding, groups, name):
    if groups == 1:
        return layers.Conv2D(filters, 3, strides=strides, groups=groups, padding="same", name=name)(x)

    slices = tf.split(x, groups, -1, name = name + "_split")
    slices = [
        layers.Conv2D(filters // groups, kernel_size, 
            strides=strides, padding=padding, name=name + "_{}".format(i))(slice)
        for i, slice in enumerate(slices)
    ]
    return layers.Concatenate(name=name + "_concat")(slices)


def anystage(x, w_in, w_out, stride, d, block_fun, name, params):
    for i in range(d):
        x = block_fun(x, w_in, w_out, stride, name + "_b{}".format(i + 1), params)
        stride, w_in = 1, w_out

    return x
