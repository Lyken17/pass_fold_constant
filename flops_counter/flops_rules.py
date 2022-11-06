from tvm import relay
from functools import reduce
FLOP_OP_MAP = {}

def register_flops_rule(op_name, level=10):
    def _register_fn(fn):
        global FLOP_OP_MAP
        FLOP_OP_MAP[op_name] = fn

        def _call(*args, **kwargs):
            return fn(*args, **kwargs)

        return _call

    return _register_fn

def l_prod(in_list):
    res = 1
    for _ in in_list:
        res *= _
    return res

def l_sum(in_list):
    res = 0
    for _ in in_list:
        res += _
    return res


@register_flops_rule("nn.batch_norm")
def batch_norm_flops(self, call):
    x0 = call.checked_type.fields[0]
    x0_shape = [int(_) for _ in x0.shape]
    x0_nelements = reduce(lambda x, y:x * y, x0_shape)
    # y = (x - mean) / \sqrt{var + e} * weight + bias
    return {
        "mul_ops": x0_nelements * 2, 
        "add_ops": x0_nelements * 2, 
    }

@register_flops_rule("nn.max_pool2d")
@register_flops_rule("reshape")
@register_flops_rule("squeeze")
@register_flops_rule("nn.relu6")
@register_flops_rule("nn.relu")
def zero_flops(self, call):
    return {}

@register_flops_rule("nn.bias_add")
@register_flops_rule("subtract")
@register_flops_rule("add")
def add_flops(self, call):
    x0, x1 = call.args
    x0_shape = [int(_) for _ in x0.checked_type.shape]
    x1_shape = [int(_) for _ in x1.checked_type.shape]
    x0_nelements = reduce(lambda x, y:x * y, x0_shape)
    x1_nelements = reduce(lambda x, y:x * y, x1_shape)
    return {
        "add_ops": max(x0_nelements, x1_nelements)
    }

@register_flops_rule("nn.adaptive_avg_pool2d")
def adaptive_avg_pool2d_flops(self, call):
    n, ic, ih, iw = [int(_) for _ in call.args[0].checked_type.shape]
    n, oc, oh, ow = [int(_) for _ in call.checked_type.shape]
    
    # mul_ops = n * m * k
    return {
        "mul_ops": n * oc,
        "add_ops": n * oc * oh * ow
    }


@register_flops_rule("nn.dense")
def dense_flops(self, call):
    x0, x1 = call.args
    n, m = [int(_) for _ in x0.checked_type.shape]
    k, _m = [int(_) for _ in  x1.checked_type.shape]
    assert m == _m 
    mul_ops = n * m * k
    return {
        "mul_ops": mul_ops
    }

@register_flops_rule("nn.conv2d")
def conv2d_flops(self, call):
    _n, inc, iw, ih = [int(_) for _ in call.args[0].checked_type.shape]
    _outc, _inc, kw, kh = [int(_) for _ in call.args[1].checked_type.shape]
    n, outc, ow, oh = [int(_) for _ in call.checked_type.shape]
    groups = int(call.attrs.groups)

    out_nelement = n * outc * ow * oh 
    mul_ops = out_nelement * (inc // groups) * kw * kh
    return {
        "mul_ops": mul_ops
    }
    
@register_flops_rule("multiply")
def multiply_flops(self, call):
    x0, x1 = call.args
    n, m = [int(_) for _ in x0.checked_type.shape]
    _m, k = [int(_) for _ in  x1.checked_type.shape]
    if n == _m and m == k:
        # element-wise
        mul_ops = n * m
    else:
        # normal matmul
        mul_ops = n * m * k
    return {
        "mul_ops": mul_ops
    }