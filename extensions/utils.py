from binaryninja import InstructionTextToken, InstructionTextTokenType, \
    LowLevelILLabel, Architecture


branch_ins = {
    'beq', 'bne', 'beqz', 'bnez', 'bge', 'bgeu', 'blt', 'bltu', 'blez', 'bgez',
    'bltz', 'bgtz', 'c.beqz', 'c.bnez'
}

direct_jump_ins   = {'c.j', 'j'}
indirect_jump_ins = {'c.jr', 'jr'}
direct_call_ins   = {'c.jal', 'jal'}
ret_ins           = {'c.ret', 'ret'}
indirect_call_ins = {'c.jalr', 'jalr'}


def sign_extend(value, bits):
    sign_bit = 1 << (bits - 1)
    return (value & (sign_bit - 1)) - (value & sign_bit)


def extract_bit(value, offset, len):
    return (value >> offset) & (2**len - 1)


def inst(cont):
    return InstructionTextToken(InstructionTextTokenType.InstructionToken, cont)


def reg(cont):
    return InstructionTextToken(InstructionTextTokenType.RegisterToken, cont)


def op_sep():
    return InstructionTextToken(InstructionTextTokenType.OperandSeparatorToken, ", ")


def imm(i):
    return InstructionTextToken(InstructionTextTokenType.IntegerToken, hex(i))


def mem_start():
    return InstructionTextToken(InstructionTextTokenType.BeginMemoryOperandToken, "(")


def mem_end():
    return InstructionTextToken(InstructionTextTokenType.EndMemoryOperandToken, ")")


def possible_address(addr):
    return InstructionTextToken(InstructionTextTokenType.PossibleAddressToken, hex(addr))


def tokenize_jmp_insn(info, r, addr):
    info.append(reg(r))
    info.append(op_sep())
    info.append(possible_address(addr))


def tokenize_uppr_insn(info, r, i):
    info.append(reg(r))
    info.append(op_sep())
    info.append(imm(i))


def tokenize_cond_insn(info, reg1, reg2, addr):
    info.append(reg(reg1))
    info.append(op_sep())
    info.append(reg(reg2))
    info.append(op_sep())
    info.append(possible_address(addr))


def tokenize_imm_insn(info, dst, src, i):
    info.append(reg(dst))
    info.append(op_sep())
    info.append(reg(src))
    info.append(op_sep())
    info.append(imm(i))


def tokenize_reg_insn(info, dst, src1, src2):
    info.append(reg(dst))
    info.append(op_sep())
    info.append(reg(src1))
    info.append(op_sep())
    info.append(reg(src2))


def tokenize_mem_insn(info, op1, op2, i):
    info.append(reg(op1))
    info.append(op_sep())
    info.append(imm(i))
    info.append(mem_start())
    info.append(reg(op2))
    info.append(mem_end())

# The following functions are helper functions to lift the assembly to the LLIL


def set_reg(il, reg, val, reg_size):
    il.append(il.set_reg(reg_size // 8, reg, val))


def set_reg_const(il, reg, val, reg_size):
    il.append(il.set_reg(reg_size // 8, reg, il.const(reg_size // 8, val & 0xffffffff)))


def set_reg_reg(il, reg, reg2, reg_size):
    il.append(il.set_reg(reg_size // 8, reg, il.reg(reg_size // 8, reg2)))


def cond_branch(il, cond, imm, reg_size):
    dest = il.add(reg_size // 8, il.const(reg_size // 8, il.current_address),
                  il.const(reg_size // 8, imm))
    t = il.get_label_for_address(Architecture["riscv"], il.current_address + imm)
    f = il.get_label_for_address(Architecture["riscv"], il.current_address + 4)
    if t is None:
        t = LowLevelILLabel()
        mark_t = True
    else:
        mark_t = False
    if f is None:
        f = LowLevelILLabel()
        mark_f = True
    else:
        mark_f = False
    il.append(il.if_expr(cond, t, f))
    if mark_t:
        il.mark_label(t)
        il.append(il.jump(dest))
    if mark_f:
        il.mark_label(f)


def branch(il, cond_f, instr, reg_size, zero=False):
    if zero:
        o2 = il.const(reg_size // 8, 0)
    else:
        o2 = il.reg(reg_size // 8, instr.operands[1])
    cond = cond_f(reg_size // 8, il.reg(reg_size // 8, instr.operands[0]), o2)
    cond_branch(il, cond, instr.imm, reg_size // 8)


def load(il, instr, size, extend, reg_size):
    offset = il.add(reg_size // 8, il.reg(reg_size // 8,
                    instr.operands[1]), il.const(reg_size // 8, instr.imm))
    set_reg(il, instr.operands[0], extend(reg_size // 8, il.load(size, offset)),
            reg_size // 8)


def store(il, instr, size, reg_size):
    offset = il.add(reg_size // 8, il.reg(reg_size // 8,
                    instr.operands[1]), il.const(reg_size // 8, instr.imm))
    if instr.operands[0] == "zero":
        val = il.const(reg_size // 8, 0)
    else:
        val = il.reg(reg_size // 8, instr.operands[0])
    il.append(il.store(size, offset, val))
