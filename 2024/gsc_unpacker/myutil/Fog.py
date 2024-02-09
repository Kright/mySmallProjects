from pydantic.dataclasses import dataclass

class Fog:
    fog: bytes
    wfog: bytes
    yfog1: bytes
    rfog: bytes

    optional1: bytes
    optional2: bytes
    optional3: bytes

    darkfog: bytes
    yfog: bytes
    trans4: bytes
    trans8: bytes
    alpha_r: bytes
    alpha_w: bytes
    refl: bytes
    water_cost: bytes
    gray_set: bytes
    bright: bytes

    @staticmethod
    def load():
        pass