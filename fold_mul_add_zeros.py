import numpy as np

from collections import Counter

import tvm
from tvm import relay
from tvm.relay import ExprFunctor, ExprMutator, ExprVisitor
from tvm.relay.expr_functor import ExprMutator, Call
from tvm.relay.dataflow_pattern import (
    wildcard,
    is_op,
    is_constant,
    is_expr,
    rewrite,
    DFPatternCallback,
)

from utils import check_call_info, diff_print_expr

class SimplifyMulAddZero(DFPatternCallback):
    def __init__(self, rewrite_once_only=False):
        super(SimplifyMulAddZero, self).__init__()
        self.x = wildcard()
        self.y = is_constant()

        add_op = is_op("add")(self.x, self.y) | is_op("add")(self.y, self.x)
        mul_op = is_op("multiply")(self.x, self.y) | is_op("multiply")(self.y, self.x)

        self.pattern = add_op | mul_op
        

    def callback(self, pre, post, node_map):
        x = node_map[self.x][0]
        y = node_map[self.y][0]

        c = y.data.numpy()
        if (c == 0).all():
            checked_type = check_call_info(pre)
            shape = [int(_) for _ in checked_type.shape]
            dtype = checked_type.dtype
            return relay.const(np.zeros(shape, dtype=dtype) , dtype=dtype)
        return pre



def test_replace_op():
    w = relay.var("w", shape=[1, 10])
    x = relay.var("x", shape=[1, 10])
    y = relay.var("y", shape=[1, 10])
    z = relay.var("z", shape=[1, 10])
    out = x * relay.const(0.0)
    out = out + y
    expr = relay.Function([x, y], out)
 

    mod = tvm.IRModule.from_expr(expr)
    prev_expr = str(expr)
    expr = SimplifyMulAddZero(rewrite_once_only=False).rewrite(expr)
    new_expr = str(expr)
    diff_print_expr(prev_expr, new_expr)


if __name__ == "__main__":
    test_replace_op()
