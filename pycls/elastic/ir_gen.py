def conv2d_ir(ir, hw, w_in, w_out, ks, stride=1, groups=1, type="conv"):
    ir.append({
        "type": type,
        "in_channels": w_in,
        "out_channels": w_out,
        "kernel_size": ks,
        "stride": stride,
        "groups": groups,
        "hw": hw,
    })
    hw = (hw - 1) // stride + 1
    return ir, hw

def gap2d_ir(ir, hw, w_in):
    ir.append({
        "type": "gap",
        "in_channels": w_in,
        "hw": hw,
    })
    hw = 1
    return ir, hw


def linear_ir(ir, hw, w_in, w_out):
    ir.append({
        "type": "linear",
        "in_channels": w_in,
        "out_channels": w_out,
        "hw": hw,
    })
    return ir, hw


def se_ir(ir, hw, w_in, w_se):
    old_hw = hw
    
    ir, hw = gap2d_ir(ir, hw, w_in)
    ir, hw = conv2d_ir(ir, hw, w_in, w_se, 1, type="conv-relu")
    ir, hw = conv2d_ir(ir, hw, w_se, w_in, 1, type="conv-sigmoid")

    hw = old_hw
    return ir, hw
