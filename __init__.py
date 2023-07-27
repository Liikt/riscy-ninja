import binaryninja
from .riscv import RISCV
from .registers import RiscVRegisters

RISCV.register()

riscv = binaryninja.architecture.Architecture['riscv']
riscv.register_calling_convention(RiscVRegisters(riscv, 'default'))
riscv.standalone_platform.default_calling_convention = \
    riscv.calling_conventions['default']

binaryninja.binaryview.BinaryViewType['ELF'].register_arch(
    243, binaryninja.enums.Endianness.LittleEndian, riscv
)
