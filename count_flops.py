import numpy as np

import torch
import torch.nn as nn

import tvm
from tvm import relay

from tvm.relay import ExprFunctor, ExprMutator, ExprVisitor

from functools import reduce
# reduce_matmul = reduce(lambda:)

class FLOPSCounter(ExprMutator):
    def __init__(self):
        super().__init__()
        self.total_flops = 0
        self.add_ops = 0
        self.mul_ops = 0
        
    def visit_call(self, call):
        new_fn = self.visit(call.op)
        
        if str(call.op) == "add":
            x0, x1 = call.args
            x0_shape = [int(_) for _ in x0.checked_type.shape]
            x1_shape = [int(_) for _ in x1.checked_type.shape]
            x0_nelements = reduce(lambda x, y:x * y, x0_shape)
            x1_nelements = reduce(lambda x, y:x * y, x0_shape)
            self.add_ops += max(x0_nelements, x1_nelements)
        elif str(call.op) == "multiply":
            x0, x1 = call.args
            n, m = [int(_) for _ in x0.checked_type.shape]
            _m, k = [int(_) for _ in  x1.checked_type.shape]
            if n == _m and m == k:
                # element-wise
                self.mul_ops += n * m
            else:
                # normal matmul
                self.mul_ops = n * m * k
        elif str(call.op) == "subtract":
            x0, x1 = call.args
            x0_shape = [int(_) for _ in x0.checked_type.shape]
            x1_shape = [int(_) for _ in x1.checked_type.shape]
            x0_nelements = reduce(lambda x, y:x * y, x0_shape)
            x1_nelements = reduce(lambda x, y:x * y, x0_shape)
            self.add_ops += max(x0_nelements, x1_nelements)
        else:
            raise NotImplementedError(call.op)
        
        new_args = [self.visit(arg) for arg in call.args]
        return call

x0 = relay.var("x0", shape=[1, 10])
x1 = relay.var("x1", shape=[1, 10])
x2 = relay.var("x2", shape=[1, 10])
x3 = relay.var("x3", shape=[1, 10])


out = x0 + x1
out = out * x2
out = out * x3
fn = relay.Function([x0, x1, x2, x3], out)
mod = tvm.IRModule.from_expr(fn)
mod = relay.transform.InferType()(mod)
fn = mod['main']

print(type(fn))
ast = FLOPSCounter()
out_node = ast.visit(fn)

print(ast.add_ops, ast.mul_ops)