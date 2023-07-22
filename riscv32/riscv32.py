from binaryninja import Architecture, Endianness, RegisterInfo, \
    InstructionTextToken, InstructionTextTokenType, BranchType, log_warn, \
    LLIL_TEMP, LowLevelILLabel
from binaryninja.architecture import InstructionInfo

from .instructions import RiscV32Instruction
from ..registers import all_regs

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


class RISCV(Architecture):
    name = "riscv"

    address_size = 4
    default_int_size = 4
    default_float_size = 16

    max_instr_length = 4

    endianness = Endianness.LittleEndian

    regs = {x: RegisterInfo(x, 4) if not x.startswith("f") or x == "fp" else RegisterInfo(x, 16) for x in all_regs}

    stack_pointer = "sp"

    def get_instruction_info(self, data, addr):
        instr = RiscV32Instruction(data, addr)
        instr.disassemble()

        result = InstructionInfo()
        result.length = instr.size

        dest = None if instr.imm is None else addr + instr.imm

        if instr.name == 'ret':
            result.add_branch(BranchType.FunctionReturn)
        elif instr.is_branch():
            result.add_branch(BranchType.TrueBranch, dest)
            result.add_branch(BranchType.FalseBranch, addr + instr.size)
        elif instr.is_direct_jump():
            result.add_branch(BranchType.UnconditionalBranch, dest)
        elif instr.is_indirect_jump():
            result.add_branch(BranchType.UnresolvedBranch)
        elif instr.is_direct_call():
            result.add_branch(BranchType.CallDestination, dest)
        elif instr.is_indirect_call():
            result.add_branch(BranchType.UnresolvedBranch)

        return result

    def _pseudo_instr_info(self, info, instr):
        match instr.name:
            case "ret" | "nop" | "fence":
                pass
            case "mv":
                info.append(reg(instr.operands[0]))
                info.append(op_sep())
                info.append(reg(instr.operands[1]))
            case "j":
                info.append(possible_address(instr.addr + instr.imm))
            case other:
                log_warn(f"Can't get info for pseudo instruction {instr.name}")

    def get_instruction_text(self, data, addr):
        instr = RiscV32Instruction(data, addr)
        instr.disassemble()

        info = [inst(instr.name.ljust(17))]

        match instr.type:
            case "r":
                info.append(reg(instr.operands[0]))
                info.append(op_sep())
                info.append(reg(instr.operands[1]))
                info.append(op_sep())
                info.append(reg(instr.operands[2]))
            case "i":
                if not instr.name.startswith("fence"):
                    info.append(reg(instr.operands[0]))
                    info.append(op_sep())
                    if instr.name.startswith("l"):
                        info.append(imm(instr.imm))
                        info.append(mem_start())
                        info.append(reg(instr.operands[1]))
                        info.append(mem_end())
                    else:
                        info.append(reg(instr.operands[1]))
                        info.append(op_sep())
                        info.append(imm(instr.imm))
            case "s":
                info.append(reg(instr.operands[0]))
                info.append(op_sep())
                info.append(imm(instr.imm))
                info.append(mem_start())
                info.append(reg(instr.operands[1]))
                info.append(mem_end())
            case "b":
                info.append(reg(instr.operands[0]))
                info.append(op_sep())
                info.append(reg(instr.operands[1]))
                info.append(op_sep())
                info.append(possible_address(addr + instr.imm))
            case "u":
                info.append(reg(instr.operands[0]))
                info.append(op_sep())
                info.append(imm(instr.imm))
            case "j":
                info.append(reg(instr.operands[0]))
                info.append(op_sep())
                info.append(possible_address(addr + instr.imm))
            case "ecall":
                pass
            case "pseudo":
                self._pseudo_instr_info(info, instr)
            case other:
                log_warn(f"Don't know {instr.type} instruction type")

        return info, 4

    def _set_reg(self, il, reg, val):
        il.append(il.set_reg(self.default_int_size, reg, val))
    def _set_reg_const(self, il, reg, val):
        il.append(il.set_reg(self.default_int_size, reg, 
                            il.const(self.default_int_size, val & 0xffffffff)))
    def _set_reg_reg(self, il, reg, reg2):
        il.append(il.set_reg(self.default_int_size, reg, 
                            il.reg(self.default_int_size, reg2)))
    def _cond_branch(self, il, cond, imm):
        dest = il.add(self.default_int_size, il.const(self.default_int_size, il.current_address),
                        il.const(self.default_int_size, imm))
        t = il.get_label_for_address(Architecture[self.name], il.current_address + imm)
        f = il.get_label_for_address(Architecture[self.name], il.current_address + 4)
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
    def _branch(self, il, cond_f, instr, zero=False):
        if zero:
            o2 = il.const(self.default_int_size, 0)
        else:
            o2 = il.reg(self.default_int_size, instr.operands[1])
        cond = cond_f(self.default_int_size, il.reg(self.default_int_size, instr.operands[0]), o2)
        self._cond_branch(il, cond, instr.imm)
    def _load(self, il, instr, size, extend):
        offset = il.add(self.default_int_size, il.reg(self.default_int_size, instr.operands[1]),
                        il.const(self.default_int_size, instr.imm))
        self._set_reg(il, instr.operands[0], extend(self.default_int_size, il.load(size, offset)))
    def _store(self, il, instr, size):
        offset = il.add(self.default_int_size, il.reg(self.default_int_size, instr.operands[1]),
                        il.const(self.default_int_size, instr.imm))
        if instr.operands[0] == "zero":
            val = il.const(self.default_int_size, 0)
        else:
            val = il.reg(self.default_int_size, instr.operands[0])
        il.append(il.store(size, offset, val))

    def get_instruction_low_level_il(self, data, addr, il):
        instr = RiscV32Instruction(data, addr)
        instr.disassemble()

        if instr.imm is not None:
            label = il.get_label_for_address(Architecture[self.name],
                                        addr + instr.imm)

        match instr.name:
            case "lui":
                self._set_reg_const(il, instr.operands[0], instr.imm << 12)
            case "auipc":
                self._set_reg_const(il, instr.operands[0], addr + (instr.imm << 12))
            case "j":
                if label is not None:
                    il.append(il.goto(label))
                else:
                    il.append(il.jump(il.const(self.default_int_size, addr + instr.imm)))
            case "jal":
                self._set_reg_const(il, instr.operands[0], addr + 4)
                if label is not None:
                    il.append(il.goto(label))
                else:
                    il.append(il.call(il.const(self.default_int_size, addr + instr.imm)))
            case "jalr":
                self._set_reg_reg(il, LLIL_TEMP(0), instr.operands[1])
                base = il.reg(self.default_int_size, LLIL_TEMP(0))
                if instr.operands[0] != "zero":
                    self._set_reg_const(il, instr.operands[0], addr + 4)
                dest = base
                if instr.imm:
                    self._set_reg(il, LLIL_TEMP(0), il.add(self.default_int_size, base, 
                                                        il.const(self.default_int_size, instr.imm)))
                    dest = il.reg(self.default_int_size, LLIL_TEMP(0))
                if instr.operands[0] == "zero":
                    il.append(il.jump(dest))
                else:
                    il.append(il.call(dest))
            case "ret":
                il.append(il.ret(il.reg(self.default_int_size, 'ra')))
            case "beq":
                self._branch(il, il.compare_equal, instr)
            case "bne":
                self._branch(il, il.compare_not_equal, instr)
            case "blt":
                self._branch(il, il.compare_signed_less_than, instr)
            case "bge":
                self._branch(il, il.compare_signed_greater_equal, instr)
            case "bltu":
                self._branch(il, il.compare_unsigned_less_than, instr)
            case "bgeu":
                self._branch(il, il.compare_unsigned_greater_equal, instr)
            case "lb":
                self._load(il, instr, 1, il.sign_extend)
            case "lbu":
                self._load(il, instr, 1, il.zero_extend)
            case "lh":
                self._load(il, instr, 2, il.sign_extend)
            case "lhu":
                self._load(il, instr, 2, il.zero_extend)
            case "lw":
                self._load(il, instr, 4, il.sign_extend)
            case "sb":
                self._store(il, instr, 1)
            case "sh":
                self._store(il, instr, 2)
            case "sw":
                self._store(il, instr, 4)
            case "add" | "addi":
                if instr.name[-1] == "i":
                    rhs = il.const(self.default_int_size, instr.imm)
                else:
                    rhs = il.reg(self.default_int_size, instr.operands[2])
                self._set_reg(il, instr.operands[0], il.add(self.default_int_size, 
                            il.reg(self.default_int_size, instr.operands[1]), rhs))
            case "sub" | "subi":
                if instr.name[-1] == "i":
                    rhs = il.const(self.default_int_size, instr.imm)
                else:
                    rhs = il.reg(self.default_int_size, instr.operands[2])
                self._set_reg(il, instr.operands[0], il.sub(self.default_int_size, 
                            il.reg(self.default_int_size, instr.operands[1]), rhs))
            case "xor" | "xori":
                if instr.name[-1] == "i":
                    rhs = il.const(self.default_int_size, instr.imm)
                else:
                    rhs = il.reg(self.default_int_size, instr.operands[2])
                self._set_reg(il, instr.operands[0], il.xor_expr(self.default_int_size, 
                            il.reg(self.default_int_size, instr.operands[1]), rhs))
            case "or" | "ori":
                if instr.name[-1] == "i":
                    rhs = il.const(self.default_int_size, instr.imm)
                else:
                    rhs = il.reg(self.default_int_size, instr.operands[2])
                self._set_reg(il, instr.operands[0], il.or_expr(self.default_int_size, 
                            il.reg(self.default_int_size, instr.operands[1]), rhs))
            case "and" | "andi":
                if instr.name[-1] == "i":
                    rhs = il.const(self.default_int_size, instr.imm)
                else:
                    rhs = il.reg(self.default_int_size, instr.operands[2])
                self._set_reg(il, instr.operands[0], il.and_expr(self.default_int_size, 
                            il.reg(self.default_int_size, instr.operands[1]), rhs))
            case "sll" | "slli":
                if instr.name[-1] == "i":
                    rhs = il.const(self.default_int_size, instr.imm)
                else:
                    rhs = il.reg(self.default_int_size, instr.operands[2])
                self._set_reg(il, instr.operands[0], il.shift_left(self.default_int_size, 
                            il.reg(self.default_int_size, instr.operands[1]), rhs))
            case "srl" | "srli":
                if instr.name[-1] == "i":
                    rhs = il.const(self.default_int_size, instr.imm)
                else:
                    rhs = il.reg(self.default_int_size, instr.operands[2])
                self._set_reg(il, instr.operands[0], il.logical_shift_right(self.default_int_size, 
                            il.reg(self.default_int_size, instr.operands[1]), rhs))
            case "sra" | "srai":
                if instr.name[-1] == "i":
                    rhs = il.const(self.default_int_size, instr.imm)
                else:
                    rhs = il.reg(self.default_int_size, instr.operands[2])
                self._set_reg(il, instr.operands[0], il.arith_shift_right(self.default_int_size, 
                            il.reg(self.default_int_size, instr.operands[1]), rhs))
            case "slt" | "slti":
                if instr.name[-1] == "i":
                    rhs = il.const(self.default_int_size, instr.imm)
                else:
                    rhs = il.reg(self.default_int_size, instr.operands[2])
                self._set_reg(il, instr.operands[0], il.compare_signed_less_than(self.default_int_size, 
                            il.reg(self.default_int_size, instr.operands[1]), rhs))
            case "sltu" | "sltiu":
                if instr.name[-2] == "i":
                    rhs = il.const(self.default_int_size, instr.imm)
                else:
                    rhs = il.reg(self.default_int_size, instr.operands[2])
                self._set_reg(il, instr.operands[0], il.compare_unsigned_less_than(self.default_int_size, 
                            il.reg(self.default_int_size, instr.operands[1]), rhs))
            case "nop" | "fence":
                il.append(il.nop())
            case "ecall":
                il.append(il.system_call())
            case "ebreak":
                il.append(il.breakpoint())
            case "mv":
                self._set_reg_reg(il, instr.operands[0], instr.operands[1])
            case other:
                log_warn(f"Can't lift instruction {instr.name}")
        return 4