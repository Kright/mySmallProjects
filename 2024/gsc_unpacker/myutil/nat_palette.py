from .data_ptr import DataPtr


class NatPalette:
    nat_pal: bytes = bytes([
        0xD0,  # red
        0xD1,
        0xD2,
        0xD3,
        0xD4,  # blue
        0xD5,
        0xD6,
        0xD7,
        0xD8,  # cyan
        0xD9,
        0xDA,
        0xDB,
        0xDC,  # purple
        0xDD,
        0xDE,
        0xDF,
        0xE0,  # orange
        0xE1,
        0xE2,
        0xE3,
        0xE4,  # black
        0xE5,
        0xE6,
        0xE7,
        0xE8,  # white
        0xE9,
        0xEA,
        0xEB,
        0xEC,  # mercs
        0xED,
        0xEE,
        0xEF
    ])

    @staticmethod
    def get_ptr(nation: int):
        return DataPtr(NatPalette.nat_pal, pos=nation << 2)
