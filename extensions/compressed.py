from struct import unpack

from binaryninja import log_debug, log_warn, Architecture, LLIL_TEMP

from .utils import sign_extend, extract_bit, inst, reg, op_sep, imm, \
    possible_address, tokenize_mem_insn, tokenize_reg_insn, tokenize_imm_insn, \
    tokenize_uppr_insn, tokenize_jmp_insn, set_reg, set_reg_const, set_reg_reg, \
    load, store, branch
from .register import IntRegister
from .instruction import RVInstruction


# CR-Type
OPCODE_BITS = lambda x: ((x >> 0)  & 0b11   )
RS_BITS     = lambda x: ((x >> 2)  & 0b11111)
RD_BITS     = lambda x: ((x >> 7)  & 0b11111)
FUNCT4_BITS = lambda x: ((x >> 12) & 0b1111 )

# CI-Type
IMMI_LO_BITS = lambda x: ((x >> 2)  & 0b11111)
IMMI_HI_BIT  = lambda x: ((x >> 12) & 0b1    )
FUNCT3_BITS = lambda x: ((x >> 13) & 0b111  )

# CSS-Type
IMMSS_BITS = lambda x: ((x >> 7) & 0b111111)

# CIW-Type
RDP_BITS  = lambda x: ((x >> 2) & 0b111)
IMMW_BITS = lambda x: ((x >> 5) & 0b11111111)

# CL-Type
IMML_LO_BITS = lambda x: ((x >> 5)  & 0b11 )
RS1P_BITS    = lambda x: ((x >> 7)  & 0b111)
IMML_HI_BITS = lambda x: ((x >> 10) & 0b111)

# CS-Type
RS2P_BITS = lambda x: ((x >> 2) & 0b111)

# CA-Type
FUNCT2_BITS = lambda x: ((x >> 5)  & 0b11    )
FUNCT6_BITS = lambda x: ((x >> 10) & 0b111111)

# CB-Type
OFFSET_LO_BITS = lambda x: ((x >> 2)  & 0b11111)
OFFSET_HI_BITS = lambda x: ((x >> 10) & 0b111  )

# CJ-Type
JUMP_TGT_BITS = lambda x: ((x >> 2) & 0b11111111111)


class CompressedInstruction(RVInstruction):
    def __init__(self, addr, data, xlen, flen, little_endian=True):
        super().__init__(addr, data, xlen, flen, little_endian)
        if len(data) < 2:
            return None

        format = "<H" if little_endian else ">H"
        self.data = unpack(format, data[:2])[0]
        self.insn_size = 2

    def _disassemble(self):
        if self.xlen != 32:
            log_warn("Compressed instructions for XLEN != 32 currently not supported")
            return
        if not self.data:
            return

        op = OPCODE_BITS(self.data)
        funct3 = FUNCT3_BITS(self.data)

        if op == 0b00 and funct3 == 0b000:
            if self.data and IMMW_BITS(self.data):
                self.operands.append(IntRegister(RDP_BITS(self.data) + 8).name)
                self.operands.append("sp")
                imm = IMMW_BITS(self.data)
                self.imm = ((imm >> 2) & 0b1111) << 6 | ((imm >> 6) & 0b11) << 3 | \
                    (imm & 0b1) << 3 | ((imm >> 1) & 0b1) << 2
                self.name = "c.addi4spn"
                self.type = "ciw"

        elif op == 0b00 and funct3 == 0b001:
            log_warn("c.fld and c.lq not yet supported")
            return

        elif op == 0b00 and funct3 == 0b010:
            self.operands.append(IntRegister(RDP_BITS(self.data) + 8).name)
            self.operands.append(IntRegister(RS1P_BITS(self.data) + 8).name)
            hi = IMML_HI_BITS(self.data)
            lo = IMML_LO_BITS(self.data)
            self.imm = (lo & 0b1) << 6 | hi << 3 | (lo >> 1) << 2
            self.name = "c.lw"
            self.type = "cl"

        elif op == 0b00 and funct3 == 0b011:
            log_warn("c.flw and c.ld not yet supported")
            return

        elif op == 0b00 and funct3 == 0b100:
            log_warn("Instruction 00 with funct3 100 is reserved")
            return

        elif op == 0b00 and funct3 == 0b101:
            log_warn("c.fsd and c.sq not yet supported")
            return

        elif op == 0b00 and funct3 == 0b110:
            self.operands.append(IntRegister(RS2P_BITS(self.data) + 8).name)
            self.operands.append(IntRegister(RS1P_BITS(self.data) + 8).name)
            hi = IMML_HI_BITS(self.data)
            lo = IMML_LO_BITS(self.data)
            self.imm = (lo & 0b1) << 6 | hi << 3 | (lo >> 1) << 2
            self.name = "c.sw"
            self.type = "cs"

        elif op == 0b00 and funct3 == 0b111:
            log_warn("c.fsw and c.sd not yet supported")
            return

        elif op == 0b01 and funct3 == 0b000:
            hi, lo = (IMMI_HI_BIT(self.data), IMMI_LO_BITS(self.data))
            if not RD_BITS(self.data) and (hi or lo):
                self.name = "c.nop"
                self.type = "pseudo"
            elif hi or lo:
                self.operands.append(IntRegister(RD_BITS(self.data)).name)
                self.operands.append(IntRegister(RD_BITS(self.data)).name)
                self.imm = sign_extend(hi << 5 | lo, 6)
                self.name = "c.addi"
                self.type = "ci"
            else:
                log_debug("Possibly not a compressed ci type")
                return
        elif op == 0b01 and funct3 == 0b001:
            if self.xlen != 32:
                log_warn("c.addiw not yet supported")
                return
            else:
                imm = JUMP_TGT_BITS(self.data)
                self.operands.append("ra")
                self.imm = extract_bit(imm, 10, 1) << 11 | extract_bit(imm, 6, 1) << 10 | \
                    extract_bit(imm, 7, 2) << 8 | extract_bit(imm, 4, 1) << 7 | \
                    extract_bit(imm, 5, 1) << 6 | extract_bit(imm, 0, 1) << 5 | \
                    extract_bit(imm, 9, 1) << 4 | extract_bit(imm, 1, 3) << 1
                self.imm = sign_extend(self.imm, 11)
                self.name = "c.jal"
                self.type = "cj"

        elif op == 0b01 and funct3 == 0b010:
            if RD_BITS(self.data):
                self.operands.append(IntRegister(RD_BITS(self.data)).name)
                self.imm = IMMI_HI_BIT(self.data) << 5 | IMMI_LO_BITS(self.data)
                self.name = "c.li"
                self.type = "pseudo"

        elif op == 0b01 and funct3 == 0b011:
            hi, lo, reg = (IMMI_HI_BIT(self.data), IMMI_LO_BITS(self.data), RD_BITS(self.data))
            if not reg or not (hi or lo):
                log_debug("0b0110000000000001 not a valid compressed instruction")
                return

            if reg == 2:
                self.imm = hi << 9 | extract_bit(lo, 1, 2) << 7 | extract_bit(lo, 3, 1) << 6 | \
                    extract_bit(lo, 0, 1) << 5 | extract_bit(lo, 4, 1) << 4
                self.imm = sign_extend(self.imm, 9)
                self.name = "c.addi16sp"
            else:
                self.imm = hi << 5 | lo
                self.name = "c.lui"
            self.operands.append(IntRegister(RD_BITS(self.data)).name)
            self.operands.append("sp")
            self.type = "cu"

        elif op == 0b01 and funct3 == 0b100:
            funct6 = FUNCT6_BITS(self.data) & 0b11
            funct2 = FUNCT2_BITS(self.data)
            hi, lo = (IMMI_HI_BIT(self.data), IMMI_LO_BITS(self.data))

            if funct6 == 0b00:
                if not (hi or lo):
                    log_warn("c.srli64 not yet supported")
                    return
                self.operands.append(IntRegister(RDP_BITS(self.data) + 8).name)
                self.operands.append(IntRegister(RS1P_BITS(self.data) + 8).name)
                self.imm = hi << 5 | lo
                self.name = "c.srli"
                self.type = "ci"

            elif funct6 == 0b01:
                if not (hi or lo):
                    log_warn("c.srai64 not yet supported")
                    return
                self.operands.append(IntRegister(RDP_BITS(self.data) + 8).name)
                self.operands.append(IntRegister(RS1P_BITS(self.data) + 8).name)
                self.imm = hi << 5 | lo
                self.name = "c.srai"
                self.type = "ci"

            elif funct6 == 0b10:
                self.operands.append(IntRegister(RDP_BITS(self.data) + 8).name)
                self.operands.append(IntRegister(RS1P_BITS(self.data) + 8).name)
                self.imm = hi << 5 | lo
                self.name = "c.andi"
                self.type = "ci"

            elif funct6 == 0b11 and funct2 == 0b00 and not hi:
                self.operands.append(IntRegister(RS1P_BITS(self.data) + 8).name)
                self.operands.append(IntRegister(RS1P_BITS(self.data) + 8).name)
                self.operands.append(IntRegister(RS2P_BITS(self.data) + 8).name)
                self.name = "c.sub"
                self.type = "ca"

            elif funct6 == 0b11 and funct2 == 0b01 and not hi:
                self.operands.append(IntRegister(RS1P_BITS(self.data) + 8).name)
                self.operands.append(IntRegister(RS1P_BITS(self.data) + 8).name)
                self.operands.append(IntRegister(RS2P_BITS(self.data) + 8).name)
                self.name = "c.xor"
                self.type = "ca"

            elif funct6 == 0b11 and funct2 == 0b10 and not hi:
                self.operands.append(IntRegister(RS1P_BITS(self.data) + 8).name)
                self.operands.append(IntRegister(RS1P_BITS(self.data) + 8).name)
                self.operands.append(IntRegister(RS2P_BITS(self.data) + 8).name)
                self.name = "c.or"
                self.type = "ca"

            elif funct6 == 0b11 and funct2 == 0b11 and not hi:
                self.operands.append(IntRegister(RS1P_BITS(self.data) + 8).name)
                self.operands.append(IntRegister(RS1P_BITS(self.data) + 8).name)
                self.operands.append(IntRegister(RS2P_BITS(self.data) + 8).name)
                self.name = "c.and"
                self.type = "ca"

            elif funct6 == 0b11 and funct2 == 0b00 and hi:
                log_warn("c.subw currently not supported")
                return
            elif funct6 == 0b11 and funct2 == 0b01 and hi:
                log_warn("c.addw currently not supported")
                return
            elif funct6 == 0b11 and funct2 == 0b10 and hi:
                log_warn("this instruction is currently reserved")
                return
            elif funct6 == 0b11 and funct2 == 0b11 and hi:
                log_warn("this instruction is currently reserved")
                return

        elif op == 0b01 and funct3 == 0b101:
            imm = JUMP_TGT_BITS(self.data)
            self.imm = extract_bit(imm, 10, 1) << 11 | extract_bit(imm, 6, 1) << 10 | \
                extract_bit(imm, 7, 2) << 8 | extract_bit(imm, 4, 1) << 7 | \
                extract_bit(imm, 5, 1) << 6 | extract_bit(imm, 0, 1) << 5 | \
                extract_bit(imm, 9, 1) << 4 | extract_bit(imm, 1, 3) << 1
            self.imm = sign_extend(self.imm, 11)
            self.name = "c.j"
            self.type = "cj"

        elif op == 0b01 and funct3 == 0b110:
            self.operands.append(IntRegister(RS1P_BITS(self.data) + 8).name)
            hi, lo = (OFFSET_HI_BITS(self.data), OFFSET_LO_BITS(self.data))
            self.imm = extract_bit(hi, 2, 1) << 8 | extract_bit(lo, 3, 2) << 6 | \
                extract_bit(lo, 0, 1) << 5 | extract_bit(hi, 0, 2) << 3 | \
                extract_bit(lo, 1, 2) << 1
            self.imm = sign_extend(self.imm, 8)
            self.name = "c.beqz"
            self.type = "cb"

        elif op == 0b01 and funct3 == 0b111:
            self.operands.append(IntRegister(RS1P_BITS(self.data) + 8).name)
            hi, lo = (OFFSET_HI_BITS(self.data), OFFSET_LO_BITS(self.data))
            self.imm = extract_bit(hi, 2, 1) << 8 | extract_bit(lo, 3, 2) << 6 | \
                extract_bit(lo, 0, 1) << 5 | extract_bit(hi, 0, 2) << 3 | \
                extract_bit(lo, 1, 2) << 1
            self.imm = sign_extend(self.imm, 8)
            self.name = "c.bnez"
            self.type = "cb"

        elif op == 0b10 and funct3 == 0b000:
            hi, lo = (IMMI_HI_BIT(self.data), IMMI_LO_BITS(self.data))
            if not (hi or lo):
                log_warn("c.srli64 not yet supported")
                return
            self.operands.append(IntRegister(RDP_BITS(self.data) + 8).name)
            self.operands.append(IntRegister(RS1P_BITS(self.data) + 8).name)
            self.imm = hi << 5 | lo
            self.name = "c.slli"
            self.type = "ci"

        elif op == 0b10 and funct3 == 0b001:
            log_warn("c.fldsp and c.lqsp not yet supported")
            return

        elif op == 0b10 and funct3 == 0b010:
            reg = RD_BITS(self.data)
            hi, lo = (IMMI_HI_BIT(self.data), IMMI_LO_BITS(self.data))
            self.operands.append(IntRegister(reg).name)
            self.operands.append("sp")
            self.imm = (lo & 0b11) << 6 | hi << 5 | (lo >> 2) << 2
            self.name = "c.lwsp"
            self.type = "css"

        elif op == 0b10 and funct3 == 0b011:
            log_warn("c.flwsp and c.ldsp not yet supported")
            return

        elif op == 0b10 and funct3 == 0b100:
            bit = IMMI_HI_BIT(self.data)
            reg1 = RD_BITS(self.data)
            reg2 = RS_BITS(self.data)

            if bit and not reg1 and not reg2:
                self.name = "c.ebreak"
                self.type = "ecall"
            elif not bit and reg1 and not reg2:
                self.operands.append("zero")
                self.operands.append(IntRegister(reg1).name)
                self.imm = 0
                self.name = "c.jr"
                self.type = "cjr"
            elif not bit and reg1 and reg2:
                self.operands.append(IntRegister(reg1).name)
                self.operands.append(IntRegister(reg2).name)
                self.name = "c.mv"
                self.type = "pseudo"
            elif bit and reg1 and not reg2:
                self.operands.append("ra")
                self.operands.append(IntRegister(reg1).name)
                self.imm = 0
                self.name = "c.jalr"
                self.type = "cjr"
            elif bit and reg1 and reg2:
                self.operands.append(IntRegister(reg1).name)
                self.operands.append(IntRegister(reg1).name)
                self.operands.append(IntRegister(reg2).name)
                self.name = "c.add"
                self.type = "ca"
            else:
                log_debug("erroneous compressed instruction")
                return

        elif op == 0b10 and funct3 == 0b101:
            log_warn("c.fsdsp and c.sqsp not yet supported")
            return

        elif op == 0b10 and funct3 == 0b110:
            imm = IMMSS_BITS(self.data)
            self.operands.append(IntRegister(RS_BITS(self.data)).name)
            self.operands.append("sp")
            self.imm = (imm & 0b11) << 6 | ((imm >> 2) & 0b1111) << 2
            self.name = "c.swsp"
            self.type = "css"

        elif op == 0b10 and funct3 == 0b111:
            log_warn("c.fswsp and c.sdsp not yet supported")
            return

        self.disassebled = True

    def pseudo_instructions(self):
        old = self.type
        self.type = "pseudo"
        if self.name == "c.jr" and self.operands[1] == "ra":
            self.name = "c.ret"
        else:
            self.type = old

    def disassemble(self):
        if self.type is None:
            self._disassemble()
        self.pseudo_instructions()

    def _pseudo_insn_token(self, info):
        match self.name:
            case "c.nop" | "c.ret":
                pass
            case "c.mv":
                info.append(reg(self.operands[0]))
                info.append(op_sep())
                info.append(reg(self.operands[1]))
            case "c.li":
                info.append(reg(self.operands[0]))
                info.append(op_sep())
                info.append(imm(self.imm))
            case other:
                print(f"Don't know compressed pseudo instruction {other}")

    def token(self):
        self.disassemble()
        if not self.disassebled:
            return None

        info = [inst(self.name.ljust(17))]

        match self.type:
            case "cb":
                tokenize_jmp_insn(info, self.operands[0], self.addr + self.imm)
            case "cu":
                tokenize_uppr_insn(info, self.operands[0], self.imm)
            case "cr":
                tokenize_reg_insn(info, self.operands[0], self.operands[0], self.operands[1])
            case "css" | "cs" | "cl":
                tokenize_mem_insn(info, self.operands[0], self.operands[1], self.imm)
            case "ciw" | "ci":
                tokenize_imm_insn(info, self.operands[0], self.operands[1], self.imm)
            case "ca":
                tokenize_reg_insn(info, self.operands[0], self.operands[1], self.operands[2])
            case "cj":
                info.append(possible_address(self.addr + self.imm))
            case "cjr":
                tokenize_mem_insn(info, self.operands[0], self.operands[1], self.imm)
            case "pseudo":
                self._pseudo_insn_token(info)
            case other:
                log_warn(f"Can't produce token for {other} compressed type")

        return info, self.insn_size

    def lift(self, il):
        self.disassemble()
        if not self.disassebled:
            return None

        if self.imm is not None:
            label = il.get_label_for_address(Architecture["riscv"],
                                             self.addr + self.imm)

        match self.name:
            case "c.addi16sp" | "c.addi4spn" | "c.addi":
                rhs = il.const(self.xlen // 8, self.imm)
                set_reg(il, self.operands[0], il.add(
                    self.xlen // 8, il.reg(self.xlen // 8, self.operands[1]), rhs), self.xlen)
            case "c.add":
                set_reg(il, self.operands[0], il.add(
                    self.xlen // 8, il.reg(self.xlen // 8, self.operands[1]),
                        il.reg(self.xlen // 8, self.operands[2])), self.xlen)
            case "c.sub":
                set_reg(il, self.operands[0], il.sub(
                    self.xlen // 8, il.reg(self.xlen // 8, self.operands[1]),
                        il.reg(self.xlen // 8, self.operands[2])), self.xlen)
            case "c.j" | "c.jal":
                if "al" in self.name:
                    set_reg_const(il, self.operands[0], self.addr + 2, self.xlen)

                if label is not None:
                    il.append(il.goto(label))
                else:
                    il.append(il.jump(il.const(self.xlen // 8, self.addr + self.imm)))
            case "c.jr" | "c.jalr":
                if "al" in self.name:
                    set_reg_reg(il, LLIL_TEMP(0), self.operands[1], self.xlen)

                base = il.reg(self.xlen // 8, LLIL_TEMP(0))
                if self.operands[0] != "zero":
                    set_reg_const(il, self.operands[0],
                                  self.addr + self.insn_size, self.xlen)
                dest = base
                if self.imm:
                    set_reg(il, LLIL_TEMP(0), il.add(self.xlen // 8, base,
                            il.const(self.xlen // 8, self.imm)), self.xlen)
                    dest = il.reg(self.xlen // 8, LLIL_TEMP(0))
                if self.operands[0] == "zero":
                    il.append(il.jump(dest))
                else:
                    il.append(il.call(dest))
            case "c.ret":
                il.append(il.ret(il.reg(self.xlen // 8, "ra")))
            case "c.or":
                set_reg(il, self.operands[0], il.or_expr(
                    self.xlen // 8, il.reg(self.xlen // 8, self.operands[1]),
                        il.reg(self.xlen // 8, self.operands[2])), self.xlen)
            case "c.xor":
                set_reg(il, self.operands[0], il.xor_expr(
                    self.xlen // 8, il.reg(self.xlen // 8, self.operands[1]),
                        il.reg(self.xlen // 8, self.operands[2])), self.xlen)
            case "c.and" | "c.andi":
                if self.name.endswith("i"):
                    rhs = il.const(self.xlen // 8, self.imm)
                else:
                    rhs = il.reg(self.xlen // 8, self.operands[2])
                set_reg(il, self.operands[0], il.and_expr(
                    self.xlen // 8, il.reg(self.xlen // 8, self.operands[1]),
                        rhs), self.xlen)
            case "c.slli":
                set_reg(il, self.operands[0], il.shift_left(
                    self.xlen // 8, il.reg(self.xlen // 8, self.operands[1]),
                    il.const(self.xlen // 8, self.imm)), self.xlen)
            case "c.lw" | "c.lwsp":
                load(il, self, 4, il.sign_extend, self.xlen)
            case "c.sw" | "c.swsp":
                store(il, self, 4, self.xlen)
            case "c.mv":
                set_reg_reg(il, self.operands[0], self.operands[1], self.xlen)
            case "c.li" | "c.lui":
                imm = self.imm
                if "u" in self.name:
                    imm = imm << 12
                set_reg_const(il, self.operands[0], imm, self.xlen)
            case "c.beqz":
                branch(il, il.compare_equal, self, self.xlen, zero=True)
            case "c.bnez":
                branch(il, il.compare_not_equal, self, self.xlen, zero=True)
            case other:
                log_warn(f"Can't lift '{other}' instruction")

        return self.insn_size


"""
            case "c.beqz" | "c.bnez":
                token.append(reg(self.operands[0]))
                token.append(op_sep())
                token.append(possible_address(self.addr + self.imm))
"""
