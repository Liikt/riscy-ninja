from binaryninja import LLIL_TEMP, Architecture, LowLevelILLabel, LowLevelILFunction

from ..registers import IntRegister
from ..utils import sign_extend

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


class RiscV32Instruction:
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
