import warnings
import numpy as np

import torch
import torch.nn as nn

import tvm
from tvm import relay

from tvm.relay import ExprFunctor, ExprMutator, ExprVisitor

from collections import Counter
from flops_counter.flops_rules import FLOP_OP_MAP


class FLOPSCounter(ExprMutator):
    def __init__(self):
        super().__init__()
        self.total_flops = Counter()
        
    def visit_call(self, call):
        new_fn = self.visit(call.op)
        
        optype = str(call.op)
        if optype in FLOP_OP_MAP:
            res = FLOP_OP_MAP[optype](self, call)
            self.total_flops += Counter(res)
        else:
            warnings.warn(f"| {call.op} | The rule for counting FLOPs is not implemented yet, treat it as zero operations temporally.")
        new_args = [self.visit(arg) for arg in call.args]
        return call

def simple_net_test():
    x0 = relay.var("x0", shape=[1, 10])
    x1 = relay.var("x1", shape=[1, 10])
    x2 = relay.var("x2", shape=[1, 10])
    x3 = relay.var("x3", shape=[1, 10])

    out = x0 + x1
    out = out * x2
    out = out - x3
    fn = relay.Function([x0, x1, x2, x3], out)
    mod = tvm.IRModule.from_expr(fn)
    mod = relay.transform.InferType()(mod)
    fn = mod['main']

    print(type(fn))
    ast = FLOPSCounter()
    out_node = ast.visit(fn)

    print(ast.total_flops)

import torch
from torchvision import models

net = models.resnet18(pretrained=True).eval()
data = torch.randn(1,3,32,32)
ts = torch.jit.trace(net, data)
mod, params = relay.frontend.from_pytorch(ts, [("data", data.shape)])
mod = relay.transform.InferType()(mod)
fn = mod['main']
# print(mod)
# 
ast = FLOPSCounter()
out_node = ast.visit(fn)
print(ast.total_flops)