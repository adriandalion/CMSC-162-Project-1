#!/usr/bin/env python3
"""
Manual PCX RLE Decoder GUI (Tkinter) - Scrollable Version

- Fully manual PCX RLE decoding (no Pillow)
- Supports: 1-bit, 8-bit, 24-bit PCX
- Full GUI scroll support
"""

from __future__ import annotations
import struct, os, tkinter as tk
from tkinter import ttk, filedialog, messagebox
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

PCX_HEADER_FMT = "<BBBBHHHHHH48sB B H H H H 54s".replace(" ", "")

@dataclass
class PCXHeader:
    manufacturer: int
    version: int
    encoding: int
    bits_per_pixel: int
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    h_dpi: int
    v_dpi: int
    colormap: bytes
    reserved: int
    n_planes: int
    bytes_per_line: int
    palette_info: int
    h_screen_size: int
    v_screen_size: int
    filler: bytes

    @property
    def width(self): return self.x_max - self.x_min + 1
    @property
    def height(self): return self.y_max - self.y_min + 1


def read_pcx_header(fp) -> PCXHeader:
    data = fp.read(128)
    if len(data) != 128:
        raise ValueError("Incomplete PCX header")
    fields = struct.unpack(PCX_HEADER_FMT, data)
    return PCXHeader(*fields)


def decode_pcx_rle(fp, expected_bytes: int) -> bytes:
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
        row_bytes = header.n_planes * header.bytes_per_line
        decoded_rows = [decode_pcx_rle(fp, row_bytes) for _ in range(height)]

        info = {
            "Filename": os.path.basename(path),
            "File Size": f"{path.stat().st_size} bytes",
            "Manufacturer": f"ZSoft .pcx ({header.manufacturer})",
            "Version": header.version,
            "Encoding": header.encoding,
            "Bits per Pixel": header.bits_per_pixel,
            "Image Dimensions": f"{width}x{height}",
            "HDPI": header.h_dpi,
            "VDPI": header.v_dpi,
            "Color Planes": header.n_planes,
            "Bytes per Line": header.bytes_per_line,
            "Palette Info": header.palette_info,
        }

        # ---- 1-bit Black & White ----
        if header.bits_per_pixel == 1 and header.n_planes == 1:
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
            return pixels, info, [(0, 0, 0), (255, 255, 255)]

        # ---- 8-bit Indexed ----
        if header.bits_per_pixel == 8 and header.n_planes == 1:
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
            return pixels, info, palette

        # ---- 24-bit RGB ----
        if header.bits_per_pixel == 8 and header.n_planes == 3:
            pixels = []
            for y in range(height):
                line = decoded_rows[y]
                bpl = header.bytes_per_line
                rplane = line[0:bpl][:width]
                gplane = line[bpl:2*bpl][:width]
                bplane = line[2*bpl:3*bpl][:width]
                row = [f"#{rplane[x]:02x}{gplane[x]:02x}{bplane[x]:02x}" for x in range(width)]
                pixels.append(row)
            return pixels, info, None

        raise NotImplementedError(
            f"Unsupported PCX format: {header.bits_per_pixel}-bit, {header.n_planes} plane(s)"
        )

# =============================================================
# SCROLLABLE GUI
# =============================================================

class PCXViewerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Manual PCX RLE Viewer")
        self.geometry("950x1000")
        self.configure(bg="white")

        # --- Main scrollable container ---
        main_canvas = tk.Canvas(self, bg="white", highlightthickness=0)
        main_scroll = ttk.Scrollbar(self, orient="vertical", command=main_canvas.yview)
        main_frame = ttk.Frame(main_canvas)

        main_canvas.configure(yscrollcommand=main_scroll.set)
        main_canvas.pack(side="left", fill="both", expand=True)
        main_scroll.pack(side="right", fill="y")

        main_canvas.create_window((0, 0), window=main_frame, anchor="nw")

        main_frame.bind("<Configure>", lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all")))

        # --- UI elements inside the scrollable frame ---
        ttk.Button(main_frame, text="Open PCX File", command=self.load_pcx).pack(pady=10)

        ttk.Label(main_frame, text="Decoded Image", font=("Segoe UI", 10, "bold")).pack()
        self.canvas = tk.Canvas(main_frame, bg="white", width=640, height=480)
        self.canvas.pack(pady=6)

        ttk.Label(main_frame, text="Header Information", font=("Segoe UI", 10, "bold")).pack()
        self.info_text = tk.Text(main_frame, height=10, wrap="word", bg="#f7f7f7", relief="flat", font=("Consolas", 10))
        self.info_text.pack(fill="x", padx=8, pady=4)

        ttk.Label(main_frame, text="Color Palette", font=("Segoe UI", 10, "bold")).pack()
        self.palette_canvas = tk.Canvas(main_frame, bg="white", highlightthickness=1, highlightbackground="#ccc")
        self.palette_canvas.pack(pady=6)

    def load_pcx(self):
        path = filedialog.askopenfilename(filetypes=[("PCX files", "*.pcx")])
        if not path:
            return
        try:
            pixels, info, palette = decode_pcx(Path(path))
            ph, pw = len(pixels), len(pixels[0])
            img = tk.PhotoImage(width=pw, height=ph)
            for y in range(ph):
                img.put("{" + " ".join(pixels[y]) + "}", to=(0, y))
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor="nw", image=img)
            self.canvas.image = img

            self.info_text.delete(1.0, tk.END)
            for k, v in info.items():
                self.info_text.insert(tk.END, f"{k}: {v}\n")

            self.draw_palette(palette)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to decode PCX:\n{e}")

    def draw_palette(self, palette):
        self.palette_canvas.delete("all")
        PAD = 8
        cols = 16

        if not palette:
            self.palette_canvas.create_text(128, 128, text="No palette data", fill="gray")
            self.palette_canvas.config(width=256, height=64)
            return

        if len(palette) == 2:
            self.palette_canvas.config(width=256, height=64)
            for i, (r, g, b) in enumerate(palette):
                x0 = PAD + i * 128
                x1 = x0 + 120
                self.palette_canvas.create_rectangle(x0, PAD, x1, 56, fill=f"#{r:02x}{g:02x}{b:02x}", outline="")
            return

        total = min(256, len(palette))
        rows = (total + cols - 1) // cols
        cell = 16
        grid_w = cols * cell
        grid_h = rows * cell
        self.palette_canvas.config(width=grid_w + 2 * PAD, height=grid_h + 2 * PAD)

        for i, (r, g, b) in enumerate(palette[:total]):
            col = i % cols
            row = i // cols
            x = PAD + col * cell
            y = PAD + row * cell
            color = f"#{r:02x}{g:02x}{b:02x}"
            self.palette_canvas.create_rectangle(x, y, x + cell, y + cell, fill=color, outline="")


if __name__ == "__main__":
    app = PCXViewerApp()
    app.mainloop()
