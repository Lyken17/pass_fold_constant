import numpy as np

from collections import Counter

import tvm
from tvm import relay
from tvm.relay import ExprFunctor, ExprMutator, ExprVisitor
from tvm.relay.expr_functor import ExprMutator, Call

from termcolor import colored

def check_call_info(call):
    expr = relay.Function(relay.analysis.all_vars(call), call)
    mod = tvm.IRModule.from_expr(expr)
    mod = relay.transform.InferType()(mod)
    return mod['main'].body.checked_type

import difflib

def diff_print_expr(expr1: relay.Function, expr2: relay.Function):
    ir_str1 = str(expr1).splitlines(keepends=True)
    ir_str2 = str(expr2).splitlines(keepends=True)
    
    diff = difflib.unified_diff(ir_str1, ir_str2)
    # print(''.join(diff))
    lines = ""
    for line in diff:
        if line.startswith("-"):
            lines += colored(line, "green")
            # print("yes")
        elif line.startswith("+"):
            lines += colored(line, "red")
        else:
            lines += line
    print(lines)



def diff_print_expr_bakup(expr1: relay.Function, expr2: relay.Function):
    ir_str1 = str(expr1).split("\n")
    ir_str2 = str(expr2).split("\n")

    max_len = max(len(ir_str1), len(ir_str2))

    for idx in range(max_len):
        if  idx < len(ir_str1) and idx < len(ir_str2) and ir_str1[idx] == ir_str2[idx]:
            print(ir_str1[idx])
        else:
            if idx < len(ir_str1):
                print(colored("-" + ir_str1[idx], "red"))
            if idx < len(ir_str2):
                print(colored("+" + ir_str2[idx], "green"))
