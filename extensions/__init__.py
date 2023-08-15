from .rv32 import RV32Instruction
from .compressed import CompressedInstruction

extensions = [RV32Instruction, CompressedInstruction]


class RiscVArch:
    def __init__(self, addr, data, xlen, flen, little_endian=True):
        assert xlen in [32, 64, 128], f"[-] {xlen} is an invalid XLen"
        assert flen in [32, 64, 128], f"[-] {flen} is an invalid FLen"

#        if xlen > 32:
#            extensions.append(RV64Instruction)
#        if xlen > 64:
#            extensions.append(RV128Instruction)

        for ext in extensions:
            self.instruction = ext(addr, data, xlen, flen, little_endian)
            self.instruction.disassemble()
            if self.instruction.disassebled:
                return

        self.instruction = None

    def info(self):
        return self.instruction.info()

    def token(self):
        if self.instruction is None:
            return [], 4
        return self.instruction.token()

    def lift(self, il):
        if self.instruction is None:
            return [], 4
        return self.instruction.lift(il)
