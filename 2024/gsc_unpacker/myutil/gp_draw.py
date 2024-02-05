from typing import List


def std_unpack(src: bytes, len: int, voc: bytes) -> bytes:
    result: List[int] = []
    src_iter = iter(src)

    while True:
        ah = next(src_iter)

        for _ in reversed(range(8)):
            if (ah & 128) != 0:
                s_little = next(src_iter)
                s_big = next(src_iter)
                count = 3 + (s_big >> 4)
                offset = s_little + ((s_big & 0xF) << 8)
                for i in range(count):
                    result.append(voc[offset + i])
                len -= count
            else:
                result.append(next(src_iter))
                len -= 1

            if len <= 0:
                return bytes(result)
            ah = ah << 1


def gray_unpack(src: bytes, unpack_len: int) -> bytes:
    result: List[int] = []

    src_iter = iter(src)

    unpack_len >>= 2  # I think there should be 1, in code was 2
    while unpack_len > 0:
        al = next(src_iter)
        bl = (al & 0x0F) << 1
        bh = (al & 0xF0) >> 3
        result.append(bl)
        result.append(bh)
        unpack_len -= 1

    return bytes(result)