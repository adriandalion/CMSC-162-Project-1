#!/usr/bin/env python3
"""
CMSC 162 – Project 1 Guide 5: Image Enhancement in Spatial Domain
Adds to the Manual PCX RLE Viewer:
 - Averaging filter (3x3)
 - Median filter (3x3)
 - High-pass Laplacian (4-neighbor kernel shown in UI)
 - Unsharp masking (user amount)
 - Highboost filtering (user amplification A>1)
 - Gradient magnitude via Sobel operator (chosen operator)
All operations are implemented in pure Python on RGB rows; median/gradient
work on grayscale for clarity, then expanded back to 3-channel preview.
"""

from __future__ import annotations
import struct, os, math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

# --------------------------- PCX DECODER (as before) ---------------------------

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
        if not b: break
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

        # 1-bit
        if header.bits_per_pixel == 1 and header.n_planes == 1:
            pixels = []
            for y in range(height):
                line = decoded_rows[y]
                row, bit_index = [], 0
                for byte in line[:header.bytes_per_line]:
                    for bit in range(7, -1, -1):
                        if bit_index >= width: break
                        val = (byte >> bit) & 1
                        row.append("#ffffff" if val else "#000000")
                        bit_index += 1
                pixels.append(row)
            return pixels, info, [(0,0,0),(255,255,255)]

        # 8-bit indexed
        if header.bits_per_pixel == 8 and header.n_planes == 1:
            palette = read_vga_palette_manual(path)
            pixels = []
            for y in range(height):
                line = decoded_rows[y][:header.bytes_per_line]
                row = []
                for x in range(width):
                    idx = line[x]
                    if palette: r,g,b = palette[idx]
                    else: r=g=b=idx
                    row.append(f"#{r:02x}{g:02x}{b:02x}")
                pixels.append(row)
            if not palette:
                palette = [(i,i,i) for i in range(256)]
            return pixels, info, palette

        # 24-bit RGB (3 planes)
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

        raise NotImplementedError(f"Unsupported PCX format: {header.bits_per_pixel}-bit, {header.n_planes} plane(s)")

# --------------------- Basic conversions & drawing helpers ---------------------

def hex_to_rgb(hexstr: str) -> Tuple[int,int,int]:
    if hexstr and hexstr.startswith("#") and len(hexstr) == 7:
        r = int(hexstr[1:3], 16)
        g = int(hexstr[3:5], 16)
        b = int(hexstr[5:7], 16)
        return r, g, b
    return 0,0,0

def rgb_rows_from_hex_rows(hex_rows: List[List[str]]) -> List[List[Tuple[int,int,int]]]:
    return [[hex_to_rgb(px) for px in row] for row in hex_rows]

def build_photoimage_from_rgb_rows(rgb_rows: List[List[Tuple[int,int,int]]]) -> tk.PhotoImage:
    h = len(rgb_rows)
    w = len(rgb_rows[0]) if h else 0
    img = tk.PhotoImage(width=w, height=h)
    for y, row in enumerate(rgb_rows):
        row_str = " ".join(f"#{r:02x}{g:02x}{b:02x}" for (r,g,b) in row)
        img.put("{" + row_str + "}", to=(0, y))
    return img

def to_gray_rows(rgb_rows: List[List[Tuple[int,int,int]]]) -> List[List[int]]:
    return [[(r+g+b)//3 for (r,g,b) in row] for row in rgb_rows]

def gray_to_rgb_rows(gray_rows: List[List[int]]) -> List[List[Tuple[int,int,int]]]:
    return [[(v,v,v) for v in row] for row in gray_rows]

def clip8(x: int) -> int:
    return 0 if x < 0 else (255 if x > 255 else x)

# --------------------------- Convolution / Median ------------------------------

def pad_replicate(gray: List[List[int]], r: int) -> List[List[int]]:
    """Replicate-pad grayscale image by r pixels."""
    h, w = len(gray), len(gray[0])
    out = [[0]*(w + 2*r) for _ in range(h + 2*r)]
    for y in range(h + 2*r):
        sy = 0 if y < r else (h-1 if y >= h+r else y - r)
        for x in range(w + 2*r):
            sx = 0 if x < r else (w-1 if x >= w+r else x - r)
            out[y][x] = gray[sy][sx]
    return out

def conv3x3_gray(gray: List[List[int]], k: List[List[int]], scale: float = 1.0, bias: float = 0.0) -> List[List[int]]:
    """Clipping variant for non-signed filters (e.g., averaging)."""
    h, w = len(gray), len(gray[0])
    p = pad_replicate(gray, 1)
    out = [[0]*w for _ in range(h)]
    for y in range(h):
        for x in range(w):
            s = (
                p[y+0][x+0]*k[0][0] + p[y+0][x+1]*k[0][1] + p[y+0][x+2]*k[0][2] +
                p[y+1][x+0]*k[1][0] + p[y+1][x+1]*k[1][1] + p[y+1][x+2]*k[1][2] +
                p[y+2][x+0]*k[2][0] + p[y+2][x+1]*k[2][1] + p[y+2][x+2]*k[2][2]
            )
            v = int(round(s*scale + bias))
            out[y][x] = clip8(v)
    return out

def conv3x3_gray_signed(gray: List[List[int]], k: List[List[int]]) -> List[List[float]]:
    """Non-clipping signed convolution (for Sobel/Laplacian)."""
    h, w = len(gray), len(gray[0])
    p = pad_replicate(gray, 1)
    out = [[0.0]*w for _ in range(h)]
    for y in range(h):
        for x in range(w):
            s = (
                p[y+0][x+0]*k[0][0] + p[y+0][x+1]*k[0][1] + p[y+0][x+2]*k[0][2] +
                p[y+1][x+0]*k[1][0] + p[y+1][x+1]*k[1][1] + p[y+1][x+2]*k[1][2] +
                p[y+2][x+0]*k[2][0] + p[y+2][x+1]*k[2][1] + p[y+2][x+2]*k[2][2]
            )
            out[y][x] = float(s)
    return out

def box_blur3_gray(gray: List[List[int]]) -> List[List[int]]:
    k = [[1,1,1],[1,1,1],[1,1,1]]
    return conv3x3_gray(gray, k, scale=1/9.0)

def median3_gray(gray: List[List[int]]) -> List[List[int]]:
    h, w = len(gray), len(gray[0])
    p = pad_replicate(gray, 1)
    out = [[0]*w for _ in range(h)]
    for y in range(h):
        for x in range(w):
            win = [
                p[y+0][x+0], p[y+0][x+1], p[y+0][x+2],
                p[y+1][x+0], p[y+1][x+1], p[y+1][x+2],
                p[y+2][x+0], p[y+2][x+1], p[y+2][x+2]
            ]
            win.sort()
            out[y][x] = win[4]
    return out

# ----------------------- Guide 5: Required Operations -------------------------

def apply_averaging(rgb_rows):
    gray = to_gray_rows(rgb_rows)
    g2 = box_blur3_gray(gray)
    return gray_to_rgb_rows(g2), "Averaging filter: 3x3 box (1/9)"

def apply_median(rgb_rows):
    gray = to_gray_rows(rgb_rows)
    g2 = median3_gray(gray)
    return gray_to_rgb_rows(g2), "Median filter: 3x3 window"

def apply_laplacian_highpass(rgb_rows):
    """Laplacian response (signed), normalized for display."""
    gray = to_gray_rows(rgb_rows)
    k = [[0,-1,0],[-1,4,-1],[0,-1,0]]
    resp = conv3x3_gray_signed(gray, k)  # keep signed!
    h, w = len(gray), len(gray[0])
    vmin = min(min(row) for row in resp)
    vmax = max(max(row) for row in resp)
    rng = vmax - vmin if vmax > vmin else 1.0
    norm = [[int(round(((resp[y][x]-vmin)/rng)*255)) for x in range(w)] for y in range(h)]
    return gray_to_rgb_rows(norm), "Laplacian High-pass (normalized signed response)"

def apply_unsharp(rgb_rows, amount: float):
    """Unsharp: g = gray + amount*(gray - blur)"""
    gray = to_gray_rows(rgb_rows)
    blur = box_blur3_gray(gray)
    h, w = len(gray), len(gray[0])
    out = [[0]*w for _ in range(h)]
    for y in range(h):
        for x in range(w):
            v = gray[y][x] + amount * (gray[y][x] - blur[y][x])
            out[y][x] = clip8(int(round(v)))
    return gray_to_rgb_rows(out), f"Unsharp masking (amount={amount})"

def apply_highboost(rgb_rows, A: float):
    """Classic highboost: g = A*gray - (A-1)*blur  (A>1)"""
    if A <= 1.0: A = 1.1
    gray = to_gray_rows(rgb_rows)
    blur = box_blur3_gray(gray)
    h, w = len(gray), len(gray[0])
    out = [[0]*w for _ in range(h)]
    for y in range(h):
        for x in range(w):
            v = A*gray[y][x] - (A-1)*blur[y][x]
            out[y][x] = clip8(int(round(v)))
    return gray_to_rgb_rows(out), f"Highboost filtering (A={A}, classic form)"

def apply_sobel_gradient(rgb_rows):
    """Sobel gradient magnitude (signed), normalized 0..255."""
    gray = to_gray_rows(rgb_rows)
    gxk = [[-1,0,1],[-2,0,2],[-1,0,1]]
    gyk = [[-1,-2,-1],[0,0,0],[1,2,1]]
    gx = conv3x3_gray_signed(gray, gxk)
    gy = conv3x3_gray_signed(gray, gyk)
    h, w = len(gray), len(gray[0])
    mag = [[0.0]*w for _ in range(h)]
    maxm = 1e-9
    for y in range(h):
        for x in range(w):
            m = (gx[y][x]**2 + gy[y][x]**2) ** 0.5
            mag[y][x] = m
            if m > maxm: maxm = m
    norm = [[int(round((mag[y][x]/maxm)*255)) for x in range(w)] for y in range(h)]
    return gray_to_rgb_rows(norm), "Gradient magnitude (Sobel, signed & normalized)"

# ------------------------------- GUI ------------------------------------------

class PCXViewerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PCX Viewer – Guide 5 Spatial Domain Enhancement (Fixed)")
        self.geometry("1080x1000")
        self.configure(bg="white")

        self._rgb_rows: Optional[List[List[Tuple[int,int,int]]]] = None
        self._hex_rows: Optional[List[List[str]]] = None
        self._img_width = 0
        self._img_height = 0

        # Layout
        main_canvas = tk.Canvas(self, bg="white", highlightthickness=0)
        main_scroll = ttk.Scrollbar(self, orient="vertical", command=main_canvas.yview)
        main_frame = ttk.Frame(main_canvas)
        main_canvas.configure(yscrollcommand=main_scroll.set)
        main_canvas.pack(side="left", fill="both", expand=True)
        main_scroll.pack(side="right", fill="y")
        main_canvas.create_window((0, 0), window=main_frame, anchor="nw")
        main_frame.bind("<Configure>", lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all")))

        top_row = ttk.Frame(main_frame)
        top_row.pack(pady=10, fill="x", padx=8)
        ttk.Button(top_row, text="Open PCX File", command=self.load_pcx).pack(side="left")

        # Spatial filters row
        sf = ttk.Frame(main_frame)
        sf.pack(pady=6, fill="x", padx=8)
        ttk.Label(sf, text="Spatial Filters:", font=("Segoe UI", 10, "bold")).pack(side="left", padx=(0,10))
        ttk.Button(sf, text="Averaging (3x3)", command=self.ui_average).pack(side="left", padx=4)
        ttk.Button(sf, text="Median (3x3)", command=self.ui_median).pack(side="left", padx=4)
        ttk.Button(sf, text="Laplacian High-pass", command=self.ui_laplacian).pack(side="left", padx=4)
        ttk.Button(sf, text="Unsharp Masking…", command=self.ui_unsharp).pack(side="left", padx=4)
        ttk.Button(sf, text="Highboost…", command=self.ui_highboost).pack(side="left", padx=4)
        ttk.Button(sf, text="Sobel Gradient", command=self.ui_sobel).pack(side="left", padx=4)

        ttk.Label(main_frame, text="Image", font=("Segoe UI", 10, "bold")).pack()
        self.canvas = tk.Canvas(main_frame, bg="white", width=640, height=480)
        self.canvas.pack(pady=6)

        self.note = ttk.Label(main_frame, text="", foreground="#444")
        self.note.pack(pady=(0,8))

        ttk.Label(main_frame, text="Header Information", font=("Segoe UI", 10, "bold")).pack()
        self.info_text = tk.Text(main_frame, height=9, wrap="word", bg="#f7f7f7", relief="flat", font=("Consolas", 10))
        self.info_text.pack(fill="x", padx=8, pady=4)


    def load_pcx(self):
        path = filedialog.askopenfilename(filetypes=[("PCX files", "*.pcx")])
        if not path: return
        try:
            pixels, info, palette = decode_pcx(Path(path))
            ph, pw = len(pixels), len(pixels[0])
            img = tk.PhotoImage(width=pw, height=ph)
            for y in range(ph):
                img.put("{" + " ".join(pixels[y]) + "}", to=(0, y))
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor="nw", image=img)
            self.canvas.image = img

            self._hex_rows = pixels
            self._rgb_rows = rgb_rows_from_hex_rows(pixels)
            self._img_width = pw
            self._img_height = ph

            self.info_text.delete(1.0, tk.END)
            for k, v in info.items():
                self.info_text.insert(tk.END, f"{k}: {v}\n")
            self.note.config(text="Loaded image.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to decode PCX:\n{e}")

    def ensure_loaded(self):
        if not self._rgb_rows:
            messagebox.showwarning("No image", "Load a PCX image first.")
            return False
        return True

    def _show(self, rgb_rows, note_text: str):
        img = build_photoimage_from_rgb_rows(rgb_rows)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=img)
        self.canvas.image = img
        self.note.config(text=note_text)

    # ---- UI handlers ----
    def ui_average(self):
        if not self.ensure_loaded(): return
        out, note = apply_averaging(self._rgb_rows)
        self._show(out, note)

    def ui_median(self):
        if not self.ensure_loaded(): return
        out, note = apply_median(self._rgb_rows)
        self._show(out, note)

    def ui_laplacian(self):
        if not self.ensure_loaded(): return
        out, note = apply_laplacian_highpass(self._rgb_rows)
        self._show(out, note)

    def ui_unsharp(self):
        if not self.ensure_loaded(): return
        a = simpledialog.askfloat("Unsharp amount", "Enter amount (e.g., 0.5 .. 2.0):", minvalue=0.0)
        if a is None: return
        out, note = apply_unsharp(self._rgb_rows, a)
        self._show(out, note)

    def ui_highboost(self):
        if not self.ensure_loaded(): return
        A = simpledialog.askfloat("Highboost A", "Enter amplification A (>1):", minvalue=1.01)
        if A is None: return
        out, note = apply_highboost(self._rgb_rows, A)
        self._show(out, note)

    def ui_sobel(self):
        if not self.ensure_loaded(): return
        out, note = apply_sobel_gradient(self._rgb_rows)
        self._show(out, note)

if __name__ == "__main__":
    app = PCXViewerApp()
    app.mainloop()
