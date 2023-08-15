from binaryninja import InstructionInfo, BranchType

from .utils import ret_ins, branch_ins, direct_jump_ins, indirect_jump_ins, \
    direct_call_ins, indirect_call_ins


class RVInstruction(object):
    def __init__(self, addr, data, xlen, flen, little_endian=True):
        self.addr = addr
        self.data = data
        self.xlen = xlen
        self.flen = flen
        self.little_endian = little_endian

        self.name = None
        self.value = None
        self.imm = None
        self.type = None
        self.operands = []
        self.insn_size = 4

        self.disassebled = False

    def is_ret(self):
        return self.name in ret_ins

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

    def info(self):
        self.disassemble()

        result = InstructionInfo()
        result.length = self.insn_size

        if not self.disassebled:
            return None

        dest = None if self.imm is None else self.addr + self.imm

        if self.is_ret():
            result.add_branch(BranchType.FunctionReturn)
        elif self.is_branch():
            result.add_branch(BranchType.TrueBranch, dest)
            result.add_branch(BranchType.FalseBranch, self.addr + self.insn_size)
        elif self.is_direct_jump():
            result.add_branch(BranchType.UnconditionalBranch, dest)
        elif self.is_indirect_jump():
            result.add_branch(BranchType.UnresolvedBranch)
        elif self.is_direct_call():
            result.add_branch(BranchType.CallDestination, dest)
        elif self.is_indirect_call():
            result.add_branch(BranchType.UnresolvedBranch)
        return result
