This project came to be because all available RiscV disassemblers either
didn't support all extensions or simply disassembled the binary incorrectly.

# Design

`riscy_ninja` is a completely handwritten disassembler for the RiscV architecture.

It aims to implement all current ratified (and maybe experimental) extensions,
while still remain easily extendable.

Additionally `riscy_ninja` should be able to disassemble machine mode code, with
all hardware registers like `pmp_addr_...`.

# TODO

[x] `RV32I` instructions
[ ] `RV64I` instructions
[ ] `RV128I` instructions
[ ] `M` instructions
[ ] `A` instructions
[ ] `Zicsr` instructions
[ ] `F` instructions
[ ] `D` instructions
[ ] `Q` instructions
[x] `C` instructions
[ ] priviledged instructions
