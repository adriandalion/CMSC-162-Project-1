#!/usr/bin/env python3
"""
pcxdecoder.py — Manual PCX RLE decoder (no Pillow)

Reads:
- Header info (manufacturer, version, encoding, etc.)
- Image pixel data
- Optional VGA palette (manually decoded)
Returns:
    pixels (list[list[str]]), header_info (dict), palette (list[tuple[int,int,int]])
"""

import struct, os
from pathlib import Path
from typing import List, Tuple, Optional

# Correct 128-byte PCX header structure (little-endian)
PCX_HEADER_FMT = "<BBBBHHHHHH48sB B H H H H 54s".replace(" ", "")

class PCXHeader:
    def __init__(self, *args):
        (
            self.manufacturer,
            self.version,
            self.encoding,
            self.bits_per_pixel,
            self.xmin,
            self.ymin,
            self.xmax,
            self.ymax,
            self.hdpi,
            self.vdpi,
            self.colormap,
            self.reserved,
            self.nplanes,
            self.bytes_per_line,
            self.palette_info,
            self.hscreen_size,
            self.vscreen_size,
            self.filler,
        ) = args

    @property
    def width(self): return self.xmax - self.xmin + 1
    @property
    def height(self): return self.ymax - self.ymin + 1


def read_pcx_header(fp) -> PCXHeader:
    data = fp.read(128)
    if len(data) != 128:
        raise ValueError("Incomplete PCX header")
    fields = struct.unpack(PCX_HEADER_FMT, data)
    return PCXHeader(*fields)


def decode_pcx_rle(fp, expected_bytes: int) -> bytes:
    """Manual PCX RLE decoding"""
    out = bytearray()
    while len(out) < expected_bytes:
        b = fp.read(1)
        if not b:
            break
        v = b[0]
        if v >= 0xC0:
            count = v & 0x3F
            data = fp.read(1)
            if not data:
                raise ValueError("Truncated RLE stream")
            out.extend(data * count)
        else:
            out.append(v)
    return bytes(out[:expected_bytes])


def read_vga_palette_manual(path: Path) -> Optional[List[Tuple[int, int, int]]]:
    """Manual read of 256-color VGA palette (769 bytes at end of file)."""
    try:
        data = Path(path).read_bytes()
        if len(data) < 769 or data[-769] != 0x0C:
            return None
        raw = data[-768:]
        return [tuple(raw[i:i+3]) for i in range(0, 768, 3)]
    except Exception:
        return None


def decode_pcx(path: Path):
    with open(path, "rb") as fp:
        header = read_pcx_header(fp)
        width, height = header.width, header.height
        row_bytes = header.nplanes * header.bytes_per_line

        decoded_rows = [decode_pcx_rle(fp, row_bytes) for _ in range(height)]

        header_info = {
            "Filename": os.path.basename(path),
            "File Size": f"{path.stat().st_size} bytes",
            "Manufacturer": f"{header.manufacturer} (ZSoft .PCX)" if header.manufacturer == 10 else str(header.manufacturer),
            "Version": header.version,
            "Encoding": header.encoding,
            "Bits per Pixel": header.bits_per_pixel,
            "Image Dimensions": f"{width} × {height}",
            "HDPI": header.hdpi,
            "VDPI": header.vdpi,
            "Color Planes": header.nplanes,
        }

        # ---- Handle 8-bit Indexed ----
        if header.bits_per_pixel == 8 and header.nplanes == 1:
            palette = read_vga_palette_manual(path)
            pixels = []
            for y in range(height):
                line = decoded_rows[y][:header.bytes_per_line]
                row = []
                for x in range(width):
                    idx = line[x]
                    if palette:
                        r, g, b = palette[idx]
                    else:
                        r = g = b = idx
                    row.append(f"#{r:02x}{g:02x}{b:02x}")
                pixels.append(row)
            if not palette:
                palette = [(i, i, i) for i in range(256)]
            return pixels, header_info, palette

        # ---- Handle 24-bit RGB ----
        if header.bits_per_pixel == 8 and header.nplanes == 3:
            pixels = []
            for y in range(height):
                line = decoded_rows[y]
                bpl = header.bytes_per_line
                rplane = line[0:bpl][:width]
                gplane = line[bpl:2*bpl][:width]
                bplane = line[2*bpl:3*bpl][:width]
                row = [f"#{rplane[x]:02x}{gplane[x]:02x}{bplane[x]:02x}" for x in range(width)]
                pixels.append(row)
            return pixels, header_info, None

        # ---- Handle 1-bit Black/White ----
        if header.bits_per_pixel == 1 and header.nplanes == 1:
            pixels = []
            for y in range(height):
                line = decoded_rows[y]
                row = []
                bit_index = 0
                for byte in line[:header.bytes_per_line]:
                    for bit in range(7, -1, -1):
                        if bit_index >= width:
                            break
                        val = (byte >> bit) & 1
                        row.append("#ffffff" if val else "#000000")
                        bit_index += 1
                pixels.append(row)
            return pixels, header_info, [(0, 0, 0), (255, 255, 255)]

        raise NotImplementedError(
            f"Unsupported PCX format: {header.bits_per_pixel}-bit, {header.nplanes}-plane"
        )


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pcxdecoder.py <file.pcx>")
    else:
        p = Path(sys.argv[1])
        pixels, info, pal = decode_pcx(p)
        for k, v in info.items():
            print(f"{k}: {v}")
        print(f"Palette entries: {len(pal) if pal else 'None'}")
