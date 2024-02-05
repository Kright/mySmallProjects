from .data_ptr import DataPtr
from .palette import Palette
from .screen import Screen
from .gp_draw import std_unpack, gray_unpack


def read_binary(name: str) -> bytes:
    with open(name, "rb") as f:
        return f.read()


def extract_int(data: bytes, offset: int = 0, signed: bool = False, size: int = 4):
    return int.from_bytes(data[offset: offset + size], byteorder="little", signed=signed)


def extracts_ints(data: bytes, *, count: int, offset: int = 0, signed: bool = False, size: int = 4):
    return [extract_int(data, offset + i * size, signed, size) for i in range(count)]


def print_bytes(data: bytes):
    for i, b in enumerate(data):
        if i % 16 == 0:
            print(f"{i}:  \t", end='')
        h = hex(b)[2:]
        if len(h) == 1:
            h = "0" + h
        if i % 16 == 15:
            print(h)
        else:
            print(h, end='_')