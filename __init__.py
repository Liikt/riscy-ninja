import binaryninja
from .riscv32.riscv32 import RISCV
from .registers import RiscVRegisters

RISCV.register()

riscv32 = binaryninja.architecture.Architecture['riscv']
riscv32.register_calling_convention(RiscVRegisters(riscv32, 'default'))
riscv32.standalone_platform.default_calling_convention = \
    riscv32.calling_conventions['default']

binaryninja.binaryview.BinaryViewType['ELF'].register_arch(
    243, binaryninja.enums.Endianness.LittleEndian, riscv32
)