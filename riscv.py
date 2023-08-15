from binaryninja import Architecture, Endianness, RegisterInfo, InstructionInfo

from .extensions import RiscVArch
from .extensions.register import all_regs


class RISCV(Architecture):
    name = "riscv"

    address_size = 4
    default_int_size = 4
    default_float_size = 8

    max_instr_length = 4

    endianness = Endianness.LittleEndian

    regs = {x: RegisterInfo(x, 4) if not x.startswith("f")
            else RegisterInfo(x, 8) for x in all_regs}

    stack_pointer = "sp"

    def get_instruction_info(self, data, addr):
        if addr % 2 != 0:
            res = InstructionInfo()
            res.length = 0
            return res

        try:
            return RiscVArch(addr, data, 32, 64).info()
        except AttributeError:
            result = InstructionInfo()
            result.length = 2
            return result

    def get_instruction_text(self, data, addr):
        if addr % 2 != 0:
            return []
        return RiscVArch(addr, data, 32, 64).token()

    def get_instruction_low_level_il(self, data, addr, il):
        if addr % 2 != 0:
            return 0
        return RiscVArch(addr, data, 32, 64).lift(il)
