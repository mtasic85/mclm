__all__ = ['pack_4x_u16_to_u64', 'unpack_u64_to_4x_u16']


def pack_4x_u16_to_u64(A: np.uint16, B: np.uint16, C: np.uint16, D: np.uint16) -> np.uint64:
    packed: np.uint64 = (np.uint64(A) << 48) | (np.uint64(B) << 32) | (np.uint64(C) << 16) | np.uint64(D)
    return packed


def unpack_u64_to_4x_u16(packed: np.uint64) -> tuple[np.uint16, np.uint16, np.uint16, np.uint16]:
    A = np.uint16((packed >> 48) & 0xFFFF)
    B = np.uint16((packed >> 32) & 0xFFFF)
    C = np.uint16((packed >> 16) & 0xFFFF)
    D = np.uint16(packed & 0xFFFF)
    return A, B, C, D
