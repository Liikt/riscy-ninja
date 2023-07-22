from binaryninja import LLIL_TEMP, Architecture, LowLevelILLabel, log_warn, \
    InstructionTextToken, InstructionTextTokenType, InstructionInfo, BranchType

from .registers import IntRegister
from .utils import sign_extend

from struct import unpack, error


branch_ins = {
    'beq', 'bne', 'beqz', 'bnez', 'bge', 'bgeu', 'blt', 'bltu', 'blez', 'bgez',
    'bltz', 'bgtz'
}

direct_jump_ins   = {'j'}
indirect_jump_ins = {'jr'}
direct_call_ins   = {'jal'}
indirect_call_ins = {'jalr'}

# R-Type
OPCODE_BITS = lambda x: ((x >> 0)  & 0b1111111)
RD_BITS     = lambda x: ((x >> 7)  & 0b11111  )
FUNCT3_BITS = lambda x: ((x >> 12) & 0b111    )
RS1_BITS    = lambda x: ((x >> 15) & 0b11111  )
RS2_BITS    = lambda x: ((x >> 20) & 0b11111  )
FUNCT7_BITS = lambda x: ((x >> 25) & 0b1111111)

# I-Type
IMMI_BITS = lambda x: ((x >> 20) & 0b111111111111)

# S-Type
IMMS_LO_BITS = lambda x: ((x >> 7)  & 0b11111  )
IMMS_HI_BITS = lambda x: ((x >> 25) & 0b1111111)

# B-Type
IMMB_11_BIT  = lambda x: ((x >> 7)  & 0b1     )
IMMB_LO_BITS = lambda x: ((x >> 8)  & 0b1111  )
IMMB_HI_BITS = lambda x: ((x >> 25) & 0b111111)
IMMB_12_BIT  = lambda x: ((x >> 31) & 0b1     )

# U-Type
IMMU_BITS = lambda x: ((x >> 12) & 0b11111111111111111111)

# J-Type
IMMJ_20_BIT  = lambda x: ((x >> 31) & 0b1         )
IMMJ_LO_BITS = lambda x: ((x >> 21) & 0b1111111111)
IMMJ_11_BIT  = lambda x: ((x >> 20) & 0b1         )
IMMJ_HI_BITS = lambda x: ((x >> 12) & 0b11111111  )

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

class RV32IInstruction:
    inst_size = 4

    def __init__(self, data, addr, little_endian=True):
        data = data[:4]
        try:
            if little_endian:
                self.data = unpack("<I", data)[0]
            else:
                self.data = unpack(">I", data)[0]
        except error as e:
            print(data, addr)
            raise e
        self.addr = addr
        self.imm = None
        self.name = ""
        self.operands = []
        self.fm = None
        self.pred = None
        self.succ = None
        self.size = 4
        self.type = None

    def is_branch(self):
        return self.name in branch_ins
    def is_direct_jump(self):
        return self.name in direct_jump_ins
    def is_indirect_jump(self):
        return self.name in indirect_jump_ins
    def is_direct_call(self):
        return self.name in direct_call_ins
    def is_indirect_call(self):
        return self.name in indirect_call_ins

    def r_type(self):
        self.operands.append(IntRegister(RD_BITS(self.data)).name)
        self.operands.append(IntRegister(RS1_BITS(self.data)).name)
        self.operands.append(IntRegister(RS2_BITS(self.data)).name)
        self.type = "r"

        match (FUNCT3_BITS(self.data), FUNCT7_BITS(self.data)):
            case (0b000, 0b0000000):
                self.name = "add"
            case (0b000, 0b0100000):
                self.name = "sub"
            case (0b001, _):
                self.name = "sll"
            case (0b010, _):
                self.name = "slt"
            case (0b011, _):
                self.name = "sltu"
            case (0b100, _):
                self.name = "xor"
            case (0b101, 0b0000000):
                self.name = "srl"
            case (0b101, 0b0100000):
                self.name = "sra"
            case (0b110, _):
                self.name = "or"
            case (0b111, _):
                self.name = "and"

    def i_type(self):
        def fence(data):
            possible = ["w", "r", "o", "i"]
            val = data & 0b1111
            stri = ""
            for x in range(4):
                if val & 1:
                    stri += possible[x]
                val >> 1
            return data & 0b1111, stri[::-1]

        self.operands.append(IntRegister(RD_BITS(self.data)).name)
        self.operands.append(IntRegister(RS1_BITS(self.data)).name)
        self.imm = sign_extend(IMMI_BITS(self.data), 12)
        self.type = "i"

        match (OPCODE_BITS(self.data), FUNCT3_BITS(self.data)):
            case (0b1100111, _):
                self.name = "jalr"
            case (0b0001111, _):
                self.name = "fence"
                self.imm = None
                self.succ, succ = fence(self.data >> 20)
                self.pred, pred = fence(self.data >> 24)
                self.fm = self.data >> 28
                if self.fm:
                    self.name = "fence.tso"
                else:
                    self.name = f"fence {pred},{succ}"
            case (0b0000011, 0b000):
                self.name = "lb"
            case (0b0000011, 0b100):
                self.name = "lbu"
            case (0b0000011, 0b001):
                self.name = "lh"
            case (0b0000011, 0b101):
                self.name = "lhu"
            case (0b0000011, 0b010):
                self.name = "lw"
            case (0b0010011, 0b000):
                self.name = "addi"
            case (0b0010011, 0b010):
                self.name = "slti"
            case (0b0010011, 0b011):
                self.name = "sltiu"
            case (0b0010011, 0b100):
                self.name = "xori"
            case (0b0010011, 0b110):
                self.name = "ori"
            case (0b0010011, 0b111):
                self.name = "andi"
            case (0b0010011, 0b001):
                self.name = "slli"
            case (0b0010011, 0b101):
                if FUNCT7_BITS(self.data):
                    self.imm = RS2_BITS(self.data)
                    self.name = "srai"
                else:
                    self.name = "srli"

    def s_type(self):
        self.operands.append(IntRegister(RS2_BITS(self.data)).name)
        self.operands.append(IntRegister(RS1_BITS(self.data)).name)
        self.imm = IMMS_HI_BITS(self.data) << 5 | IMMS_LO_BITS(self.data)
        self.type = "s"

        match FUNCT3_BITS(self.data):
            case 0b000:
                self.name = "sb"
            case 0b001:
                self.name = "sh"
            case 0b010:
                self.name = "sw"

    def b_type(self):
        self.operands.append(IntRegister(RS1_BITS(self.data)).name)
        self.operands.append(IntRegister(RS2_BITS(self.data)).name)
        self.imm = IMMB_12_BIT(self.data) << 12 | IMMB_11_BIT(self.data) << 11 |\
            IMMB_HI_BITS(self.data) << 5 | IMMB_LO_BITS(self.data) << 1
        self.type = "b"

        match FUNCT3_BITS(self.data):
            case 0b000:
                self.name = "beq"
            case 0b001:
                self.name = "bne"
            case 0b100:
                self.name = "blt"
            case 0b101:
                self.name = "bge"
            case 0b110:
                self.name = "bltu"
            case 0b111:
                self.name = "bgeu"

    def u_type(self):
        self.operands.append(IntRegister(RD_BITS(self.data)).name)
        self.imm = IMMU_BITS(self.data)
        self.type = "u"

        match OPCODE_BITS(self.data):
            case 0b0110111:
                self.name = "lui"
            case 0b0010111:
                self.name = "auipc"

    def j_type(self):
        self.operands.append(IntRegister(RD_BITS(self.data)).name)
        self.imm = IMMJ_20_BIT(self.data) << 20 | IMMJ_HI_BITS(self.data) << 12 |\
            IMMJ_11_BIT(self.data) << 11 | IMMJ_LO_BITS(self.data) << 1
        self.imm = sign_extend(self.imm, 12)
        self.name = "jal"
        self.type = "j"

    def ecall(self):
        self.name = "ebreak" if IMMI_BITS(self.data) else "ecall"
        self.type = "ecall"

    def pseudo_instruction(self):
        old = self.type
        self.type = "pseudo"
        if self.name == 'jalr' and self.operands[0] == 'zero' and \
                self.operands[1] == 'ra' and not self.imm:
            self.name = "ret"
        elif self.name == 'jr' and self.operands[0] == 'ra' \
                and not self.imm:
            self.name = "ret"
        elif self.name == "addi" and self.operands[0] == 'zero' and \
                self.operands[1] == 'zero' and self.imm == 0:
            self.name = "nop"
        elif self.name == "addi" and self.imm == 0:
            self.name = "mv"
        elif self.name == "addi" and self.operands[1] == "zero":
            self.name = "li"
        elif self.name == "add" and self.operands[1] == "zero":
            self.name = "mv"
            self.operands = [self.operands[0], self.operands[2]]
        elif self.name == "add" and self.operands[2] == "zero":
            self.name = "mv"
            self.operands = [self.operands[0], self.operands[1]]
        elif self.name == "jal" and self.operands[0] == 'zero':
            self.name = "j"
        elif self.name == "fence iorw,iorw":
            self.name = "fence"
        else:
            self.type = old

    def disassemble(self):
        match OPCODE_BITS(self.data):
            case 0b0110011:
                self.r_type()
            case 0b1100111 | 0b0000011 | 0b0010011 | 0b0001111:
                self.i_type()
            case 0b0100011:
                self.s_type()
            case 0b1100011:
                self.b_type()
            case 0b0110111 | 0b0010111:
                self.u_type()
            case 0b1101111:
                self.j_type()
            case 0b1110011:
                self.ecall()

        self.pseudo_instruction()

    def info(self):
        if not self.name:
            self.disassemble()

        result = InstructionInfo()
        result.length = self.size

        dest = None if self.imm is None else self.addr + self.imm

        if self.name == 'ret':
            result.add_branch(BranchType.FunctionReturn)
        elif self.is_branch():
            result.add_branch(BranchType.TrueBranch, dest)
            result.add_branch(BranchType.FalseBranch, self.addr + self.size)
        elif self.is_direct_jump():
            result.add_branch(BranchType.UnconditionalBranch, dest)
        elif self.is_indirect_jump():
            result.add_branch(BranchType.UnresolvedBranch)
        elif self.is_direct_call():
            result.add_branch(BranchType.CallDestination, dest)
        elif self.is_indirect_call():
            result.add_branch(BranchType.UnresolvedBranch)
        return result

    def _pseudo_instr_token(self, token):
        match self.name:
            case "ret" | "nop" | "fence":
                pass
            case "mv":
                token.append(reg(self.operands[0]))
                token.append(op_sep())
                token.append(reg(self.operands[1]))
            case "j":
                token.append(possible_address(self.addr + self.imm))
            case other:
                log_warn(f"Can't get token for pseudo instruction {self.name}")

    def token(self):
        if not self.name:
            self.disassemble()

        info = [inst(self.name.ljust(17))]

        match self.type:
            case "r":
                info.append(reg(self.operands[0]))
                info.append(op_sep())
                info.append(reg(self.operands[1]))
                info.append(op_sep())
                info.append(reg(self.operands[2]))
            case "i":
                if not self.name.startswith("fence"):
                    info.append(reg(self.operands[0]))
                    info.append(op_sep())
                    if self.name.startswith("l"):
                        info.append(imm(self.imm))
                        info.append(mem_start())
                        info.append(reg(self.operands[1]))
                        info.append(mem_end())
                    else:
                        info.append(reg(self.operands[1]))
                        info.append(op_sep())
                        info.append(imm(self.imm))
            case "s":
                info.append(reg(self.operands[0]))
                info.append(op_sep())
                info.append(imm(self.imm))
                info.append(mem_start())
                info.append(reg(self.operands[1]))
                info.append(mem_end())
            case "b":
                info.append(reg(self.operands[0]))
                info.append(op_sep())
                info.append(reg(self.operands[1]))
                info.append(op_sep())
                info.append(possible_address(self.addr + self.imm))
            case "u":
                info.append(reg(self.operands[0]))
                info.append(op_sep())
                info.append(imm(self.imm))
            case "j":
                info.append(reg(self.operands[0]))
                info.append(op_sep())
                info.append(possible_address(self.addr + self.imm))
            case "ecall":
                pass
            case "pseudo":
                self._pseudo_instr_token(info)
            case other:
                log_warn(f"Don't know {self.type} instruction type")
        return info

    def _set_reg(self, il, reg, val):
        il.append(il.set_reg(4, reg, val))
    def _set_reg_const(self, il, reg, val):
        il.append(il.set_reg(4, reg, il.const(4, val & 0xffffffff)))
    def _set_reg_reg(self, il, reg, reg2):
        il.append(il.set_reg(4, reg, il.reg(4, reg2)))
    def _cond_branch(self, il, cond, imm):
        dest = il.add(4, il.const(4, il.current_address), il.const(4, imm))
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
    def _branch(self, il, cond_f, instr, zero=False):
        if zero:
            o2 = il.const(4, 0)
        else:
            o2 = il.reg(4, instr.operands[1])
        cond = cond_f(4, il.reg(4, instr.operands[0]), o2)
        self._cond_branch(il, cond, instr.imm)
    def _load(self, il, instr, size, extend):
        offset = il.add(4, il.reg(4, instr.operands[1]), il.const(4, instr.imm))
        self._set_reg(il, instr.operands[0], extend(4, il.load(size, offset)))
    def _store(self, il, instr, size):
        offset = il.add(4, il.reg(4, instr.operands[1]), il.const(4, instr.imm))
        if instr.operands[0] == "zero":
            val = il.const(4, 0)
        else:
            val = il.reg(4, instr.operands[0])
        il.append(il.store(size, offset, val))

    def lift(self, il):
        if not self.name:
            self.disassemble()

        if self.imm is not None:
            label = il.get_label_for_address(Architecture["riscv"],
                                        self.addr + self.imm)

        match self.name:
            case "lui":
                self._set_reg_const(il, self.operands[0], self.imm << 12)
            case "auipc":
                self._set_reg_const(il, self.operands[0], self.addr + (self.imm << 12))
            case "j":
                if label is not None:
                    il.append(il.goto(label))
                else:
                    il.append(il.jump(il.const(4, self.addr + self.imm)))
            case "jal":
                self._set_reg_const(il, self.operands[0], self.addr + 4)
                if label is not None:
                    il.append(il.goto(label))
                else:
                    il.append(il.call(il.const(4, self.addr + self.imm)))
            case "jalr":
                self._set_reg_reg(il, LLIL_TEMP(0), self.operands[1])
                base = il.reg(4, LLIL_TEMP(0))
                if self.operands[0] != "zero":
                    self._set_reg_const(il, self.operands[0], self.addr + 4)
                dest = base
                if self.imm:
                    self._set_reg(il, LLIL_TEMP(0), il.add(4, base, il.const(4, self.imm)))
                    dest = il.reg(4, LLIL_TEMP(0))
                if self.operands[0] == "zero":
                    il.append(il.jump(dest))
                else:
                    il.append(il.call(dest))
            case "ret":
                il.append(il.ret(il.reg(4, 'ra')))
            case "beq":
                self._branch(il, il.compare_equal, self)
            case "bne":
                self._branch(il, il.compare_not_equal, self)
            case "blt":
                self._branch(il, il.compare_signed_less_than, self)
            case "bge":
                self._branch(il, il.compare_signed_greater_equal, self)
            case "bltu":
                self._branch(il, il.compare_unsigned_less_than, self)
            case "bgeu":
                self._branch(il, il.compare_unsigned_greater_equal, self)
            case "lb":
                self._load(il, self, 1, il.sign_extend)
            case "lbu":
                self._load(il, self, 1, il.zero_extend)
            case "lh":
                self._load(il, self, 2, il.sign_extend)
            case "lhu":
                self._load(il, self, 2, il.zero_extend)
            case "lw":
                self._load(il, self, 4, il.sign_extend)
            case "sb":
                self._store(il, self, 1)
            case "sh":
                self._store(il, self, 2)
            case "sw":
                self._store(il, self, 4)
            case "add" | "addi":
                if self.name[-1] == "i":
                    rhs = il.const(4, self.imm)
                else:
                    rhs = il.reg(4, self.operands[2])
                self._set_reg(il, self.operands[0], il.add(4, il.reg(4, self.operands[1]), rhs))
            case "sub" | "subi":
                if self.name[-1] == "i":
                    rhs = il.const(4, self.imm)
                else:
                    rhs = il.reg(4, self.operands[2])
                self._set_reg(il, self.operands[0], il.sub(4, il.reg(4, self.operands[1]), rhs))
            case "xor" | "xori":
                if self.name[-1] == "i":
                    rhs = il.const(4, self.imm)
                else:
                    rhs = il.reg(4, self.operands[2])
                self._set_reg(il, self.operands[0], il.xor_expr(4, il.reg(4, self.operands[1]), rhs))
            case "or" | "ori":
                if self.name[-1] == "i":
                    rhs = il.const(4, self.imm)
                else:
                    rhs = il.reg(4, self.operands[2])
                self._set_reg(il, self.operands[0], il.or_expr(4, il.reg(4, self.operands[1]), rhs))
            case "and" | "andi":
                if self.name[-1] == "i":
                    rhs = il.const(4, self.imm)
                else:
                    rhs = il.reg(4, self.operands[2])
                self._set_reg(il, self.operands[0], il.and_expr(4, il.reg(4, self.operands[1]), rhs))
            case "sll" | "slli":
                if self.name[-1] == "i":
                    rhs = il.const(4, self.imm)
                else:
                    rhs = il.reg(4, self.operands[2])
                self._set_reg(il, self.operands[0], il.shift_left(4, il.reg(4, self.operands[1]), rhs))
            case "srl" | "srli":
                if self.name[-1] == "i":
                    rhs = il.const(4, self.imm)
                else:
                    rhs = il.reg(4, self.operands[2])
                self._set_reg(il, self.operands[0], il.logical_shift_right(4, il.reg(4, self.operands[1]), rhs))
            case "sra" | "srai":
                if self.name[-1] == "i":
                    rhs = il.const(4, self.imm)
                else:
                    rhs = il.reg(4, self.operands[2])
                self._set_reg(il, self.operands[0], il.arith_shift_right(4, il.reg(4, self.operands[1]), rhs))
            case "slt" | "slti":
                if self.name[-1] == "i":
                    rhs = il.const(4, self.imm)
                else:
                    rhs = il.reg(4, self.operands[2])
                self._set_reg(il, self.operands[0], il.compare_signed_less_than(4, il.reg(4, self.operands[1]), rhs))
            case "sltu" | "sltiu":
                if self.name[-2] == "i":
                    rhs = il.const(4, self.imm)
                else:
                    rhs = il.reg(4, self.operands[2])
                self._set_reg(il, self.operands[0], il.compare_unsigned_less_than(4, il.reg(4, self.operands[1]), rhs))
            case "nop" | "fence":
                il.append(il.nop())
            case "ecall":
                il.append(il.system_call())
            case "ebreak":
                il.append(il.breakpoint())
            case "mv":
                self._set_reg_reg(il, self.operands[0], self.operands[1])
            case other:
                log_warn(f"Can't lift instruction {self.name}")