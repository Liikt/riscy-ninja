j _blub
_blub:
    add a0, a1, a2
    sub a0, a1, a2
    sll a0, a1, a2
    slt a0, a1, a2
    sltu a0, a1, a2
    xor a0, a1, a2
    srl a0, a1, a2
    sra a0, a1, a2
    or a0, a1, a2
    and a0, a1, a2
    auipc fp, 0x10
    lb a0, -10(sp)
    lbu a0, 0(sp)
    lh a0, 10(sp)
    lhu a0, 0(sp)
    lw a0, 0(sp)
    addi a1, a0, 0x10
    slti a1, a0, 0x10
    sltiu a1, a0, 0x10
    xori a1, a0, -0x10
    ori a1, a0, 0x10
    andi a1, a0, 0x10
    slli a1, a0, 0x10
    srli a1, a0, 0x10
    srai a1, a0, 0x10
    sb a1, 0x10(sp)
    sh a1, 0x10(sp)
    sw a1, 0x10(sp)
    beq a0, a1, _true
    bne a0, a1, _true
    blt a0, a1, _true
    bge a0, a1, _weird
    bltu a0, a1, _true
    bgeu a0, a1, _true
    mv a0, a1
    nop
_false:
    lui a1, 0x10
    j _false
_weird:
    fence rw,rw
    ecall
    ebreak
_true:
    jalr a0, 0(sp)
