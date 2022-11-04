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

class SimplifyAddTwoIdentical(DFPatternCallback):
    def __init__(self, rewrite_once_only=False):
        super(SimplifyAddTwoIdentical, self).__init__()
        self.x = wildcard()
        self.y = wildcard()

        add_op = is_op("add")(self.x, self.y) | is_op("add")(self.y, self.x)

        self.pattern = add_op
        

    def callback(self, pre, post, node_map):
        x = node_map[self.x][0]
        y = node_map[self.y][0]

        if x.handle.value == y.handle.value:
            # same objects:
            return relay.multiply(x, relay.const(2, dtype=x.type_annotation.dtype))
        return pre



def test_replace_op():
    x = relay.var("x", shape=[1, 10])
    out = x * relay.const(0.0)
    out = x + x
    expr = relay.Function([x,], out)
 
    mod = tvm.IRModule.from_expr(expr)
    prev_expr = str(expr)
    expr = SimplifyAddTwoIdentical(rewrite_once_only=False).rewrite(expr)
    new_expr = str(expr)
    diff_print_expr(prev_expr, new_expr)


if __name__ == "__main__":
    test_replace_op()
