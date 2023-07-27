from binaryninja import InstructionTextToken, InstructionTextTokenType, \
    log_warn, log_debug

from struct import unpack, error

from .registers import IntRegister
from .utils import extract_bit

direct_jump_ins   = {'c.j'}
indirect_jump_ins = {'c.jr'}
direct_call_ins   = {'c.jal'}
indirect_call_ins = {'c.jalr'}
branch_ins        = {'c.beqz', 'c.bnez'}

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


class CompressedInstruction:
    def __init__(self, data, addr, little_endian=True):
        if len(data) == 1:
            return
        data = data[:2]
        self.size = 32
        try:
            if little_endian:
                self.data = unpack("<H", data)[0]
            else:
                self.data = unpack(">H", data)[0]
        except error as e:
            print(data, addr)
            raise e
        self.addr = addr
        self.imm = None
        self.name = ""
        self.operands = []
        self.instr_size = 2
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

    def disassemble(self):
        if self.size != 32:
            log_warn("Compressed instructions for XLEN != 32 currently not supported")
            return
        if not self.data:
            return

        op = OPCODE_BITS(self.data)
        funct3 = FUNCT3_BITS(self.data)

        if op == 0b00 and funct3 == 0b000:
            if self.data and IMMW_BITS(self.data):
                self.operands.append(IntRegister(RDP_BITS(self.data)).name)
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
            self.operands.append(IntRegister(RS1P_BITS(self.data)+8).name)
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
            self.operands.append(IntRegister(RS2P_BITS(self.data)+8).name)
            self.operands.append(IntRegister(RS1P_BITS(self.data)+8).name)
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
                self.type = "ci"
            elif hi or lo:
                self.operands.append(IntRegister(RD_BITS(self.data)).name)
                self.operands.append(IntRegister(RD_BITS(self.data)).name)
                self.imm = hi << 5 | lo
                self.name = "c.addi"
                self.type = "ci"
            else:
                log_debug("Possibly not a compressed ci type")
                return
        elif op == 0b01 and funct3 == 0b001:
            if self.size != 32:
                log_warn("c.addiw not yet supported")
                return
            else:
                imm = JUMP_TGT_BITS(self.data)
                self.operands.append("ra")
                self.imm = extract_bit(imm, 10, 1) << 11 | extract_bit(imm, 6, 1) << 10 | \
                    extract_bit(imm, 7, 2) << 8 | extract_bit(imm, 4, 1) << 7 | \
                    extract_bit(imm, 5, 1) << 6 | extract_bit(imm, 0, 1) << 5 | \
                    extract_bit(imm, 9, 1) << 4 | extract_bit(imm, 1, 3) << 1
                self.name = "c.jal"
                self.type = "cj"

        elif op == 0b01 and funct3 == 0b010:
            if RD_BITS(self.data):
                self.operands.append(IntRegister(RD_BITS(self.data)).name)
                self.imm = IMMI_HI_BIT(self.data) << 5 | IMMI_LO_BITS(self.data)
                self.name = "c.li"
                self.type = "ci"

        elif op == 0b01 and funct3 == 0b011:
            hi, lo, reg = (IMMI_HI_BIT(self.data), IMMI_LO_BITS(self.data), RD_BITS(self.data))
            if not reg or not (hi or lo):
                log_debug("0b0110000000000001 not a valid compressed instruction")
                return

            if reg == 2:
                self.imm = hi << 9 | ((lo >> 1) & 0b11) << 7 | ((lo >> 4) & 0b1) << 6 | \
                    (lo & 0b1) << 5 | (lo >> 5) << 4
                self.name = "c.addi16sp"
            else:
                self.imm = hi << 5 | lo
                self.name = "c.lui"
            self.operands.append(IntRegister(RD_BITS(self.data)).name)
            self.type = "cu"

        elif op == 0b01 and funct3 == 0b100:
            funct6 = FUNCT6_BITS(self.data) & 0b11
            funct2 = FUNCT2_BITS(self.data)
            hi, lo = (IMMI_HI_BIT(self.data), IMMI_LO_BITS(self.data))

            if funct6 == 0b00:
                if not (hi or lo):
                    log_warn("c.srli64 not yet supported")
                    return
                self.operands.append(IntRegister(RDP_BITS(self.data)+8).name)
                self.operands.append(IntRegister(RS1P_BITS(self.data)+8).name)
                self.imm = hi << 5 | lo
                self.name = "c.srli"
                self.type = "ci"

            elif funct6 == 0b01:
                if not (hi or lo):
                    log_warn("c.srai64 not yet supported")
                    return
                self.operands.append(IntRegister(RDP_BITS(self.data)+8).name)
                self.operands.append(IntRegister(RS1P_BITS(self.data)+8).name)
                self.imm = hi << 5 | lo
                self.name = "c.srai"
                self.type = "ci"

            elif funct6 == 0b10:
                self.operands.append(IntRegister(RDP_BITS(self.data)+8).name)
                self.operands.append(IntRegister(RS1P_BITS(self.data)+8).name)
                self.imm = hi << 5 | lo
                self.name = "c.andi"
                self.type = "ci"

            elif funct6 == 0b11 and funct2 == 0b00 and not hi:
                self.operands.append(IntRegister(RDP_BITS(self.data)+8).name)
                self.operands.append(IntRegister(RS1P_BITS(self.data)+8).name)
                self.operands.append(IntRegister(RS2P_BITS(self.data)+8).name)
                self.name = "c.sub"
                self.type = "ca"

            elif funct6 == 0b11 and funct2 == 0b01 and not hi:
                self.operands.append(IntRegister(RDP_BITS(self.data)+8).name)
                self.operands.append(IntRegister(RS1P_BITS(self.data)+8).name)
                self.operands.append(IntRegister(RS2P_BITS(self.data)+8).name)
                self.name = "c.xor"
                self.type = "ca"

            elif funct6 == 0b11 and funct2 == 0b10 and not hi:
                self.operands.append(IntRegister(RDP_BITS(self.data)+8).name)
                self.operands.append(IntRegister(RS1P_BITS(self.data)+8).name)
                self.operands.append(IntRegister(RS2P_BITS(self.data)+8).name)
                self.name = "c.or"
                self.type = "ca"

            elif funct6 == 0b11 and funct2 == 0b11 and not hi:
                self.operands.append(IntRegister(RDP_BITS(self.data)+8).name)
                self.operands.append(IntRegister(RS1P_BITS(self.data)+8).name)
                self.operands.append(IntRegister(RS2P_BITS(self.data)+8).name)
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
            self.name = "c.j"
            self.type = "cj"

        elif op == 0b01 and funct3 == 0b110:
            self.operands.append(IntRegister(RS1P_BITS(self.data)+8).name)
            hi, lo = (OFFSET_HI_BITS(self.data), OFFSET_LO_BITS(self.data))
            self.imm = (hi >> 3) << 8 | (lo >> 4) << 6 | (lo & 0b1) << 5 | (hi & 0b11) << 3 | \
                ((lo >> 1) & 0b11) << 1
            self.name = "c.beqz"
            self.type = "cb"

        elif op == 0b01 and funct3 == 0b111:
            self.operands.append(IntRegister(RS1P_BITS(self.data)+8).name)
            hi, lo = (OFFSET_HI_BITS(self.data), OFFSET_LO_BITS(self.data))
            self.imm = (hi >> 3) << 8 | (lo >> 4) << 6 | (lo & 0b1) << 5 | (hi & 0b11) << 3 | \
                ((lo >> 1) & 0b11) << 1
            self.name = "c.bnez"
            self.type = "cb"

        elif op == 0b10 and funct3 == 0b000:
            hi, lo = (IMMI_HI_BIT(self.data), IMMI_LO_BITS(self.data))
            if not (hi or lo):
                log_warn("c.srli64 not yet supported")
                return
            self.operands.append(IntRegister(RDP_BITS(self.data)+8).name)
            self.operands.append(IntRegister(RS1P_BITS(self.data)+8).name)
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
            self.imm = (lo & 0b11) << 6 | hi << 5 | (lo >> 2) << 4
            self.name = "c.lwsp"
            self.type = "ci"

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
                self.operands.append(IntRegister(reg1).name)
                self.name = "c.jr"
                self.type = "cj"
            elif not bit and reg1 and reg2:
                self.operands.append(IntRegister(reg1).name)
                self.operands.append(IntRegister(reg2).name)
                self.name = "c.mv"
                self.type = "cr"
            elif bit and reg1 and not reg2:
                self.operands.append("ra")
                self.operands.append(IntRegister(reg1).name)
                self.imm = 0
                self.name = "c.jalr"
                self.type = "ci"
            elif bit and reg1 and reg2:
                self.operands.append(IntRegister(reg1).name)
                self.operands.append(IntRegister(reg1).name)
                self.operands.append(IntRegister(reg2).name)
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
