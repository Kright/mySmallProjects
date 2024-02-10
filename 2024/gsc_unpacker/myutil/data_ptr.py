from typing import List


class DataPtr:
    def __init__(self, data, pos: int = 0, allow_read_overflow: bool = False):
        self.data = data
        self.pos = pos
        self.allow_read_overflow = allow_read_overflow

    def get(self) -> int:
        if self.allow_read_overflow and len(self.data) <= self.pos:
            return 0
        return self.data[self.pos]

    def set(self, value: int):
        self.data[self.pos] = value

    def __getitem__(self, offset: int):
        i = self.pos + offset
        if self.allow_read_overflow and (i < 0 or i >= len(self.data)):
            return 0
        return self.data[i]

    def advance(self, shift: int = 1):
        self.pos += shift

    def consume_one(self) -> int:
        r = self.get()
        self.advance()
        return r

    def push_one(self, value: int):
        self.set(value)
        self.advance()

    def copy(self) -> 'DataPtr':
        return DataPtr(self.data, self.pos)

    def consume_many(self, count: int) -> List[int]:
        return [self.consume_one() for _ in range(count)]

    def push_many(self, values: List[int]):
        for v in values:
            self.push_one(v)

    @staticmethod
    def valid_zeros() -> 'DataPtr':
        return DataPtr([], 0, allow_read_overflow=True)
