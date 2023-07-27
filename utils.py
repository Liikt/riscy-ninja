def sign_extend(value, bits):
    sign_bit = 1 << (bits - 1)
    return (value & (sign_bit - 1)) - (value & sign_bit)


def extract_bit(value, offset, len):
    return (value >> offset) & (2**len - 1)
