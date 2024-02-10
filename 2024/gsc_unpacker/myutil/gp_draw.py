from typing import List, Tuple, Dict, Any, Optional, Callable

import json
import numpy as np
from pydantic import BaseModel
from pydantic.dataclasses import dataclass
from PIL import Image
import os

from .screen import Screen
from .data_ptr import DataPtr
from .util import extract_int
from .palette import Palette
from .nat_palette import NatPalette
from .util import print_bytes


def std_unpack(src: bytes, length: int, voc: bytes) -> bytes:
    result: List[int] = []
    src_iter = iter(src)

    while True:
        ah = next(src_iter)

        for _ in range(8):
            if (ah & 128) != 0:
                s_little = next(src_iter)
                s_big = next(src_iter)
                count = 3 + (s_big >> 4)
                offset = s_little + ((s_big & 0xF) << 8)
                for i in range(count):
                    result.append(voc[offset + i])
                length -= count
            else:
                result.append(next(src_iter))
                length -= 1

            if length <= 0:
                return bytes(result)
            ah = ah << 1


def nat_unpack(src: DataPtr, length: int) -> bytes:
    result: List[int] = []

    for _ in range(length >> 2):
        al = src.consume_one()
        result.append(al & 0b11)
        result.append((al >> 2) & 0b11)
        result.append((al >> 4) & 0b11)
        result.append((al >> 6) & 0b11)

    return bytes(result)



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


def lz_unpack(src: DataPtr, length: int) -> bytes:
    result: List[int] = []

    while True:
        ah = src.consume_one()

        for _ in range(8):
            if (ah & 1) != 0:
                cl, ch = src.consume_many(2)
                ecx = cl | (ch << 8)
                copy_count = (ecx >> 13) + 3
                offset = len(result) - 1 - (ecx & 0x1FFF)
                for _ in range(copy_count):
                    result.append(result[offset])
                    offset += 1
                    length -= 1
            else:
                result.append(src.consume_one())
                length -= 1

            if length <= 0:
                return bytes(result)

            ah = ah >> 1


@dataclass
class GpVocabulary:
    voc_offset: int
    voc_length: int
    data: bytes

    def __repr__(self) -> str:
        return f"GpVocabulary(voc_offset = {self.voc_offset}, voc_length = {self.voc_length})"


class GpHeader(BaseModel):
    next_pict: int
    dx: int
    dy: int
    lx: int
    ly: int
    pack: int  # actually pointer, but 0xFFFFFFFF when not loaded
    options: int
    cdata: int
    offset_data: bytes
    n_lines: int
    voc: GpVocabulary
    packed_data: Optional[bytes]
    next_pict_header: Optional['GpHeader']

    @property
    def opt(self) -> int:
        return self.options & 63

    @property
    def unpack_len(self) -> int:
        r = self.cdata >> 14

        opt = self.opt

        if opt == 43: r += 262144
        if opt == 44: r += 262144 * 2

        return r

    @property
    def cdoffs(self) -> int:
        r = self.cdata & 16383

        if (self.options & 64) != 0: r += 16384
        if (self.options & 128) != 0: r += 32768

        return r

    @property
    def opt_descr(self) -> str:
        opt = self.opt
        if opt == 0: return 'standard packing'
        if opt == 1: return 'National mask'
        if opt == 2: return 'transparent 1/4'
        if opt == 3: return 'transparent 2/4'
        if opt == 4: return 'transparent 3/4'
        if opt == 5: return 'Shadow'
        if opt == 6: return 'AlphaRY'
        if opt == 7: return 'AlphaWB'
        if opt == 38: return 'wtf 38'
        if opt == 39: return 'wtf 39'
        if opt == 41: return 'wtf 41'
        if opt in range(42, 45): return f'wtf {opt}'
        return f'unknown {opt}'

    @property
    def headers_list(self) -> List['GpHeader']:
        if self.next_pict_header is None:
            return [self]
        return [self] + self.next_pict_header.headers_list

    def get_lx(self):
        return max(h.dx + h.lx for h in self.headers_list)

    def get_ly(self):
        return max(h.dy + h.ly for h in self.headers_list)

    def get_dx(self):
        return min(h.dx for h in self.headers_list)

    def get_dy(self):
        return min(h.dy for h in self.headers_list)

    def get_dx_dy_lx_ly(self) -> Tuple[int, int, int, int]:
        return self.get_dx(), self.get_dy(), self.get_lx(), self.get_ly()

    def try_unpack_data(self) -> Optional[bytes]:
        if self.opt == 0:
            return std_unpack(self.packed_data, self.unpack_len, self.voc.data)

        if self.opt == 1:
            return nat_unpack(DataPtr(self.packed_data), self.unpack_len)

        if self.opt == 38:
            return gray_unpack(self.packed_data, self.unpack_len)

        if self.opt in (42, 43, 44):
            return lz_unpack(DataPtr(self.packed_data), self.unpack_len)

        print(f'unsupported format: {self.opt}')

    def get_repr(self, field_names) -> str:
        fields = ', '.join(f"{name} = {self.__getattribute__(name)}" for name in field_names)
        return f"GpHeader({fields})"

    def __repr__(self) -> str:
        headers = self.headers_list
        fields = ['next_pict', 'dx', 'dy', 'lx', 'ly', 'options', 'opt', 'opt_descr', 'unpack_len', 'cdoffs', 'n_lines']
        return '[' + ',\n'.join(h.get_repr(fields) for h in headers) + ']'

    @staticmethod
    def get_size() -> int:
        # return 4 + 2 + 2 + 2 + 2 + 4 + 1 + 4 + 2
        return 23

    @staticmethod
    def parse(original_data: bytes, offset: int, *, parent_offset: Optional[int] = None,
              voc: GpVocabulary) -> 'GpHeader':
        data = original_data[offset:]

        r = GpHeader(
            next_pict=extract_int(data, 0, signed=True, size=4),
            dx=extract_int(data, 4, signed=True, size=2),
            dy=extract_int(data, 6, signed=True, size=2),
            lx=extract_int(data, 8, signed=True, size=2),
            ly=extract_int(data, 10, signed=True, size=2),
            pack=extract_int(data, 12, signed=False, size=4),
            options=extract_int(data, 16, signed=False, size=1),
            cdata=extract_int(data, 17, signed=False, size=4),
            n_lines=extract_int(data, 21, signed=True, size=2),
            voc=voc,
            offset_data=data[GpHeader.get_size():],
            packed_data=None,
            next_pict_header=None,
        )

        assert r.pack == 0xFF_FF_FF_FF
        r = r.copy(update={
            'packed_data': data[r.cdoffs:],
            'offset_data': r.offset_data[:r.cdoffs - 23],
        })

        if r.next_pict != -1:
            parent_offset = parent_offset or offset
            r = r.copy(
                update={'next_pict_header': GpHeader.parse(original_data, parent_offset + r.next_pict,
                                                           parent_offset=parent_offset, voc=voc)})

        return r


@dataclass
class GpGlobalHeader:
    signature: bytes
    n_pictures: int
    reserved: bytes
    voc: GpVocabulary
    headers_offsets: List[int]
    headers: List[GpHeader]

    def unpack_to_dir(self, directory: str, palette: Palette):
        os.makedirs(directory, exist_ok=True)

        pictures = []
        for i, top_header in enumerate(self.headers):
            frames = []
            dx, dy, lx, ly = top_header.get_dx_dy_lx_ly()

            screen = Screen(lx - dx, ly - dy, Palette.TRANSPARENT)

            for j, h in enumerate(top_header.headers_list):
                pic_name = f"{i:03d}_{j:03d}_{h.opt}.png"

                # try:
                if True:
                    if h.opt == 0:
                        screen.fill(Palette.TRANSPARENT)
                        gp_show_masked_pict(-dx, -dy, h, h.try_unpack_data(), screen)
                        pic = screen.as_pic(palette)
                        pic.save(f"{directory}/{pic_name}")
                    elif h.opt == 1:
                        screen.fill(Palette.TRANSPARENT)
                        nation = 0 #in 0..7
                        gp_show_masked_pal_pict(-dx, -dy, h, h.try_unpack_data(), NatPalette.get_ptr(nation), screen)
                        pic = screen.as_pic(palette)
                        pic.save(f"{directory}/{pic_name}")
                    elif h.opt == 5:
                        # just draw shadow mask, but game remaps colors instead
                        screen.fill(Palette.TRANSPARENT)
                        gp_show_masked_pict_shadow_make_mask(-dx, -dy, h, 0x00, screen)
                        pic = screen.as_pic(palette)
                        pic.save(f"{directory}/{pic_name}")
                    elif h.opt in (42, 43, 44):
                        screen.fill(Palette.TRANSPARENT)
                        gp_show_masked_pict(-dx, -dy, h, h.try_unpack_data(), screen)
                        pic = screen.as_pic(palette)
                        pic.save(f"{directory}/{pic_name}")


                # except Exception as e:
                #     print(e)


                frames.append({
                    'opt': h.opt,
                    'dx': h.dx,
                    'dy': h.dy,
                    'lx': h.lx,
                    'ly': h.ly,
                    'pic_name': pic_name,
                    'opt_descr': h.opt_descr,
                })

            pictures.append({
                'dx': dx,
                'dy': dy,
                'lx': lx,
                'ly': ly,
                'frames': frames,
            })

        data = {
            'n_pictures': self.n_pictures,
            'pictures': pictures,
        }

        with open(f"{directory}/info.json", "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def parse(data: bytes) -> 'GpGlobalHeader':
        assert extract_int(data, offset=0) == 0x4B_41_50_47

        n_pictures = extract_int(data, offset=4, signed=True, size=2)
        headers_offsets = [extract_int(data, offset=14 + i * 4, signed=False, size=4) for i in range(n_pictures)]

        voc_offset = extract_int(data, offset=8, signed=True, size=4)
        voc_length = extract_int(data, offset=12, signed=True, size=2)
        voc = GpVocabulary(
            voc_offset=voc_offset,
            voc_length=voc_length,
            data=data[voc_offset: voc_offset + voc_length]
        )

        return GpGlobalHeader(
            signature=data[0:4],
            n_pictures=n_pictures,
            reserved=data[6:8],
            voc=voc,
            headers_offsets=headers_offsets,
            headers=[GpHeader.parse(data, offset=i, voc=voc) for i in headers_offsets],
        )

    def __str__(self) -> str:
        return repr(self)

    def __repr__(self) -> str:
        headers = ',\n'.join(repr(h) for h in self.headers)

        return f"""GpGlobalHeader(
    signaure = {self.signature}
    n_pictures = {self.n_pictures}
    voc = {self.voc}
    headers_offsets = {self.headers_offsets}
    headers = \n{headers}"""

    @property
    def headers_flat(self) -> List[GpHeader]:
        result = []
        for hh in self.headers:
            result += hh.headers_list
        return result


def gp_show_common(x: int, y: int, pic: GpHeader, cdata: Optional[bytes], screen: Screen, map_colors: Callable[[int, int], int]):
    x += pic.dx
    y += pic.dy

    n_lines = pic.n_lines
    ofst = DataPtr(pic.offset_data, 0)

    # no clipping case
    assert x >= screen.x, f'Should be {x} >= {screen.x}'
    assert y >= screen.y, f'Should be {y} >= {screen.y}'
    assert x + pic.lx - 1 <= screen.x1, f'Should be {x + pic.lx - 1} <= {screen.x1}'
    assert y + n_lines - 1 <= screen.y1, f'Should be {y + n_lines - 1} <= {screen.y1}'

    if cdata is not None:
        cdpos = DataPtr(cdata, 0, allow_read_overflow=False)
    else:
        cdpos = DataPtr.valid_zeros()


    scr_offset: DataPtr = DataPtr(screen.data, x + y * screen.width)

    line_start: DataPtr = scr_offset.copy()
    line_start.advance(-screen.width)

    for line in range(n_lines):
        al = ofst.consume_one()
        line_start.advance(screen.width)
        edi = line_start.copy()

        if al == 0: continue

        if (al & 128) != 0:
            # DRAW_COMPLEX_LINE

            space_mask = 0
            pix_mask = 0
            if (al & 64) != 0: space_mask = 16
            if (al & 32) != 0: pix_mask = 16

            al = al & 31
            if al == 0:
                continue

            for _ in range(al):
                al = ofst.consume_one()
                cl = (al >> 4) | pix_mask
                al = (al & 0xF) | space_mask
                edi.advance(al)

                for i in range(cl):
                    edi.push_one(map_colors(edi.get(), cdpos.consume_one()))

            continue

        # START_SIMPLE_SEGMENT
        for _ in range(al):
            al = ofst.consume_one()
            edi.advance(al)
            cl = ofst.consume_one()

            for i in range(cl):
                edi.push_one(map_colors(edi.get(), cdpos.consume_one()))


def gp_show_masked_pict(x: int, y: int, pic: GpHeader, cdata: bytes, screen: Screen):
    gp_show_common(x, y, pic, cdata, screen, lambda old_color, pixel: pixel)


def gp_show_masked_multi_pal_pict(x: int, y: int, pic: GpHeader, cdata: bytes, screen: Screen, encoder: DataPtr):
    gp_show_common(x, y, pic, cdata, screen, lambda old_color, pixel: encoder[old_color + (pixel << 8)])


def gp_show_masked_pal_pict(x: int, y: int, pic: GpHeader, cdata: bytes, encoder: DataPtr, screen: Screen):
    gp_show_common(x, y, pic, cdata, screen, lambda old_pixel, pixel: encoder[pixel])


def gp_show_masked_pict_shadow(x: int, y: int, pic: GpHeader, encoder: DataPtr, screen: Screen):
    gp_show_common(x, y, pic, None, screen, lambda old_pixel, pixel: encoder[old_pixel])


def gp_show_masked_pict_shadow_make_mask(x: int, y: int, pic: GpHeader, mask_color: int, screen: Screen):
    gp_show_common(x, y, pic, None, screen, lambda old_pixel, pixel: mask_color)

