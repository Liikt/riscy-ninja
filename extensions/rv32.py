from struct import unpack

from binaryninja import LLIL_TEMP, Architecture, log_warn

from .utils import sign_extend, inst, imm, reg, op_sep, possible_address, \
    set_reg, set_reg_reg, set_reg_const, branch, load, store, tokenize_mem_insn, \
    tokenize_reg_insn, tokenize_imm_insn, tokenize_cond_insn, tokenize_uppr_insn, \
    tokenize_jmp_insn
from .register import IntRegister
from .instruction import RVInstruction


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


class RV32Instruction(RVInstruction):
    def __init__(self, addr, data, xlen, flen, little_endian=True):
        super().__init__(addr, data, xlen, flen, little_endian)
        if len(data) < 4:
            return None

        format = "<I" if little_endian else ">I"
        self.value = unpack(format, data[:4])[0]

        self.fm = None
        self.pred = None
        self.succ = None

    def r_type(self):
        self.operands.append(IntRegister(RD_BITS(self.value)).name)
        self.operands.append(IntRegister(RS1_BITS(self.value)).name)
        self.operands.append(IntRegister(RS2_BITS(self.value)).name)
        self.type = "r"

        match (FUNCT3_BITS(self.value), FUNCT7_BITS(self.value)):
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

        self.operands.append(IntRegister(RD_BITS(self.value)).name)
        self.operands.append(IntRegister(RS1_BITS(self.value)).name)
        self.imm = sign_extend(IMMI_BITS(self.value), 12)
        self.type = "i"

        match (OPCODE_BITS(self.value), FUNCT3_BITS(self.value)):
            case (0b1100111, _):
                self.name = "jalr"
            case (0b0001111, _):
                self.name = "fence"
                self.imm = None
                self.succ, succ = fence(self.value >> 20)
                self.pred, pred = fence(self.value >> 24)
                self.fm = self.value >> 28
                if self.fm:
                    self.name = "fence.tso"
                else:
                    self.name = f"fence {pred},{succ}"
            case (0b0000011, 0b000):
                self.imm = sign_extend(self.imm, 12)
                self.name = "lb"
            case (0b0000011, 0b100):
                self.imm = sign_extend(self.imm, 12)
                self.name = "lbu"
            case (0b0000011, 0b001):
                self.imm = sign_extend(self.imm, 12)
                self.name = "lh"
            case (0b0000011, 0b101):
                self.imm = sign_extend(self.imm, 12)
                self.name = "lhu"
            case (0b0000011, 0b010):
                self.imm = sign_extend(self.imm, 12)
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
                if FUNCT7_BITS(self.value):
                    self.imm = RS2_BITS(self.value)
                    self.name = "srai"
                else:
                    self.name = "srli"

    def s_type(self):
        self.operands.append(IntRegister(RS2_BITS(self.value)).name)
        self.operands.append(IntRegister(RS1_BITS(self.value)).name)
        self.imm = IMMS_HI_BITS(self.value) << 5 | IMMS_LO_BITS(self.value)
        self.imm = sign_extend(self.imm, 12)
        self.type = "s"

        match FUNCT3_BITS(self.value):
            case 0b000:
                self.name = "sb"
            case 0b001:
                self.name = "sh"
            case 0b010:
                self.name = "sw"

    def b_type(self):
        self.operands.append(IntRegister(RS1_BITS(self.value)).name)
        self.operands.append(IntRegister(RS2_BITS(self.value)).name)
        self.imm = IMMB_12_BIT(self.value) << 12 | IMMB_11_BIT(self.value) << 11 |\
            IMMB_HI_BITS(self.value) << 5 | IMMB_LO_BITS(self.value) << 1
        self.imm = sign_extend(self.imm, 11)
        self.type = "b"

        match FUNCT3_BITS(self.value):
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
        self.operands.append(IntRegister(RD_BITS(self.value)).name)
        self.imm = IMMU_BITS(self.value)
        self.type = "u"

        match OPCODE_BITS(self.value):
            case 0b0110111:
                self.name = "lui"
            case 0b0010111:
                self.name = "auipc"

    def j_type(self):
        self.operands.append(IntRegister(RD_BITS(self.value)).name)
        self.imm = IMMJ_20_BIT(self.value) << 20 | IMMJ_HI_BITS(self.value) << 12 |\
            IMMJ_11_BIT(self.value) << 11 | IMMJ_LO_BITS(self.value) << 1
        self.imm = sign_extend(self.imm, 12)
        self.name = "jal"
        self.type = "j"

    def ecall(self):
        self.name = "ebreak" if IMMI_BITS(self.value) else "ecall"
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

    def _disassemble(self):
        if self.value is None:
            return

        disassembled = True
        match OPCODE_BITS(self.value):
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
            case _:
                disassembled = False
        self.disassebled = disassembled

    def disassemble(self):
        if self.type is None:
            self._disassemble()
        self.pseudo_instruction()

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
            case "li":
                token.append(reg(self.operands[0]))
                token.append(op_sep())
                token.append(imm(self.imm))
            case _:
                log_warn(f"Can't get token for pseudo instruction {self.name}")

    def token(self):
        self.disassemble()
        if not self.disassebled:
            return None

        info = [inst(self.name.ljust(17))]

        match self.type:
            case "r":
                tokenize_reg_insn(info, self.operands[0], self.operands[1], self.operands[2])
            case "i":
                if self.name.startswith("fence"):
                    pass

                if self.name.startswith("l") or self.name == "jalr":
                    tokenize_mem_insn(info, self.operands[0], self.operands[1], self.imm)
                else:
                    tokenize_imm_insn(info, self.operands[0], self.operands[1], self.imm)
            case "s":
                tokenize_mem_insn(info, self.operands[0], self.operands[1], self.imm)
            case "b":
                tokenize_cond_insn(info, self.operands[0], self.operands[1], self.addr + self.imm)
            case "u":
                tokenize_uppr_insn(info, self.operands[0], self.imm)
            case "j":
                tokenize_jmp_insn(info, self.operands[0], self.addr + self.imm)
            case "ecall":
                pass
            case "pseudo":
                self._pseudo_instr_token(info)
            case _:
                log_warn(f"Don't know {self.type} instruction type")
        return info, self.insn_size

    def lift(self, il):
        self.disassemble()
        if not self.disassebled:
            return None

        if self.imm is not None:
            label = il.get_label_for_address(Architecture["riscv"],
                                             self.addr + self.imm)

        match self.name:
            case "lui":
                set_reg_const(il, self.operands[0], self.imm << 12, 32)
            case "auipc":
                set_reg_const(il, self.operands[0], self.addr + (self.imm << 12), 32)
            case "j":
                if label is not None:
                    il.append(il.goto(label))
                else:
                    il.append(il.jump(il.const(4, self.addr + self.imm)))
            case "jal":
                set_reg_const(il, self.operands[0], self.addr + 4, 32)
                if label is not None:
                    il.append(il.goto(label))
                else:
                    il.append(il.call(il.const(4, self.addr + self.imm)))
            case "jalr":
                set_reg_reg(il, LLIL_TEMP(0), self.operands[1], 32)
                base = il.reg(4, LLIL_TEMP(0))
                if self.operands[0] != "zero":
                    set_reg_const(il, self.operands[0], self.addr + self.insn_size, 32)
                dest = base
                if self.imm:
                    set_reg(il, LLIL_TEMP(0), il.add(4, base, il.const(4, self.imm)), 32)
                    dest = il.reg(4, LLIL_TEMP(0))
                if self.operands[0] == "zero":
                    il.append(il.jump(dest))
                else:
                    il.append(il.call(dest))
            case "ret":
                il.append(il.ret(il.reg(4, 'ra')))
            case "beq":
                branch(il, il.compare_equal, self, 32)
            case "beqz":
                branch(il, il.compare_equal, self, 32, zero=True)
            case "bne":
                branch(il, il.compare_not_equal, self, 32)
            case "bnez":
                branch(il, il.compare_not_equal, self, 32, zero=True)
            case "blt":
                branch(il, il.compare_signed_less_than, self, 32)
            case "bge":
                branch(il, il.compare_signed_greater_equal, self, 32)
            case "bltu":
                branch(il, il.compare_unsigned_less_than, self, 32)
            case "bgeu":
                branch(il, il.compare_unsigned_greater_equal, self, 32)
            case "lb":
                load(il, self, 1, il.sign_extend, 32)
            case "lbu":
                load(il, self, 1, il.zero_extend, 32)
            case "lh":
                load(il, self, 2, il.sign_extend, 32)
            case "lhu":
                load(il, self, 2, il.zero_extend, 32)
            case "lw":
                load(il, self, 4, il.sign_extend, 32)
            case "sb":
                store(il, self, 1, 32)
            case "sh":
                store(il, self, 2, 32)
            case "sw":
                store(il, self, 4, 32)
            case "add" | "addi":
                if self.name[-1] == "i":
                    rhs = il.const(4, self.imm)
                else:
                    rhs = il.reg(4, self.operands[2])
                set_reg(il, self.operands[0], il.add(4, il.reg(4, self.operands[1]), rhs), 32)
            case "sub" | "subi":
                if self.name[-1] == "i":
                    rhs = il.const(4, self.imm)
                else:
                    rhs = il.reg(4, self.operands[2])
                set_reg(il, self.operands[0], il.sub(4, il.reg(4, self.operands[1]), rhs), 32)
            case "xor" | "xori":
                if self.name[-1] == "i":
                    rhs = il.const(4, self.imm)
                else:
                    rhs = il.reg(4, self.operands[2])
                set_reg(il, self.operands[0], il.xor_expr(
                    4, il.reg(4, self.operands[1]), rhs), 32)
            case "or" | "ori":
                if self.name[-1] == "i":
                    rhs = il.const(4, self.imm)
                else:
                    rhs = il.reg(4, self.operands[2])
                set_reg(il, self.operands[0], il.or_expr(4, il.reg(4, self.operands[1]), rhs), 32)
            case "and" | "andi":
                if self.name[-1] == "i":
                    rhs = il.const(4, self.imm)
                else:
                    rhs = il.reg(4, self.operands[2])
                set_reg(il, self.operands[0], il.and_expr(
                    4, il.reg(4, self.operands[1]), rhs), 32)
            case "sll" | "slli":
                if self.name[-1] == "i":
                    rhs = il.const(4, self.imm)
                else:
                    rhs = il.reg(4, self.operands[2])
                set_reg(il, self.operands[0], il.shift_left(
                    4, il.reg(4, self.operands[1]), rhs), 32)
            case "srl" | "srli":
                if self.name[-1] == "i":
                    rhs = il.const(4, self.imm)
                else:
                    rhs = il.reg(4, self.operands[2])
                set_reg(il, self.operands[0], il.logical_shift_right(
                    4, il.reg(4, self.operands[1]), rhs), 32)
            case "sra" | "srai":
                if self.name[-1] == "i":
                    rhs = il.const(4, self.imm)
                else:
                    rhs = il.reg(4, self.operands[2])
                set_reg(il, self.operands[0], il.arith_shift_right(
                    4, il.reg(4, self.operands[1]), rhs), 32)
            case "slt" | "slti":
                if self.name[-1] == "i":
                    rhs = il.const(4, self.imm)
                else:
                    rhs = il.reg(4, self.operands[2])
                set_reg(il, self.operands[0], il.compare_signed_less_than(
                    4, il.reg(4, self.operands[1]), rhs), 32)
            case "sltu" | "sltiu":
                if self.name[-2] == "i":
                    rhs = il.const(4, self.imm)
                else:
                    rhs = il.reg(4, self.operands[2])
                set_reg(il, self.operands[0], il.compare_unsigned_less_than(
                    4, il.reg(4, self.operands[1]), rhs), 32)
            case "nop" | "fence":
                il.append(il.nop())
            case "ecall":
                il.append(il.system_call())
            case "ebreak":
                il.append(il.breakpoint())
            case "mv":
                set_reg_reg(il, self.operands[0], self.operands[1], 32)
            case "li":
                set_reg_const(il, self.operands[0], self.imm, 32)
            case _:
                log_warn(f"Can't lift instruction '{self.name}'")

        return self.insn_size
