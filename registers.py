from binaryninja import CallingConvention
from enum import IntEnum

class IntRegister(IntEnum):
    zero = 0
    ra   = 1
    sp   = 2
    gp   = 3
    tp   = 4
    t0   = 5
    t1   = 6
    t2   = 7
    s0   = 8
    s1   = 9
    a0   = 10
    a1   = 11
    a2   = 12
    a3   = 13
    a4   = 14
    a5   = 15
    a6   = 16
    a7   = 17
    s2   = 18
    s3   = 19
    s4   = 20
    s5   = 21
    s6   = 22
    s7   = 23
    s8   = 24
    s9   = 25
    s10  = 26
    s11  = 27
    t3   = 28
    t4   = 29
    t5   = 30
    t6   = 31

special_regs = ["zero", "ra", "gp", "fp", "sp", "tp"]
int_regs = [f"a{x}" for x in range(8)] + [f"s{x}" for x in range(1, 12)] + \
    [f"t{x}" for x in range(7)] + special_regs
float_regs = [f"fa{x}" for x in range(8)] + [f"fs{x}" for x in range(12)] + \
    [f"ft{x}" for x in range(12)]
all_regs = int_regs + float_regs

caller_saved = [x for x in int_regs if not "s" in x or x != "fp"] + \
    [x for x in float_regs if not "s" in x]
callee_saved = [x for x in int_regs if "s" in x or x == "fp"] + \
    [x for x in float_regs if "s" in x]

class RiscVRegisters(CallingConvention):
    name = "RiscV"
    global_pointer_reg = 'gp'
    caller_saved_regs = tuple(caller_saved)
    callee_saved_regs = tuple(callee_saved)

    int_arg_regs = tuple([x for x in int_regs if x.startswith("a")])
    int_return_reg = 'a0'
    high_int_return_reg = 'a1'

    float_arg_regs = tuple([x for x in float_regs if x.startswith("fa")])
    float_return_arg = 'fa0'
    high_float_return_arg = 'fa1'

    implicitly_defined_regs = ('tp', 'gp')