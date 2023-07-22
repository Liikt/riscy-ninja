from binaryninja import Architecture, Endianness, RegisterInfo

from .rv32i import RV32IInstruction
from .registers import all_regs

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
        return RV32IInstruction(data, addr).info()

    def get_instruction_text(self, data, addr):
        return RV32IInstruction(data, addr).token(), 4

    def get_instruction_low_level_il(self, data, addr, il):
        RV32IInstruction(data, addr).lift(il)
        return 4