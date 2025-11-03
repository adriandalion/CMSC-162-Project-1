#!/usr/bin/env python3
"""
CMSC 162 – Integrated Project 1 (Guides 1–5)
- Pure Tkinter, no Pillow
- One shared PCX decoder + shared utilities
- Guide 2: open PCX, header, palette, image
- Guide 3: R/G/B channels + histograms, grayscale + histogram
- Guide 4: negative, manual threshold (0..255), gamma, histogram equalization (grayscale)
- Guide 5: spatial filters: averaging, median, Laplacian (high-pass), unsharp masking, high-boost, Sobel magnitude
- UI: palette auto-hides for processed truecolor; histograms appear in pop-up windows
"""
from __future__ import annotations
import struct, os, math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

# =============================== CORE: PCX DECODER (Guide 2) ===============================

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
            if not data: raise ValueError("Truncated RLE stream")
            out.extend(data * count)
        else:
            out.append(v)
    return bytes(out[:expected_bytes])

def read_vga_palette_manual(path: Path) -> Optional[List[Tuple[int,int,int]]]:
    try:
        data = path.read_bytes()
        if len(data) < 769 or data[-769] != 0x0C:
            return None
        raw = data[-768:]
        return [tuple(raw[i:i+3]) for i in range(0, 768, 3)]
    except Exception:
        return None

def decode_pcx(path: Path):
    """
    Return (pixels_rows_hex, info_dict, palette_or_None, meta_dict)
    pixels_rows_hex: list[list[str "#rrggbb"]]
    meta: minimal fields used by UI (bits_per_pixel, n_planes)
    """
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
        meta = {"bits_per_pixel": header.bits_per_pixel, "n_planes": header.n_planes}

        # 1-bit
        if header.bits_per_pixel == 1 and header.n_planes == 1:
            pixels = []
            for y in range(height):
                line = decoded_rows[y]
                row = []
                bit_index = 0
                for byte in line[:header.bytes_per_line]:
                    for bit in range(7, -1, -1):
                        if bit_index >= width: break
                        val = (byte >> bit) & 1
                        row.append("#ffffff" if val else "#000000")
                        bit_index += 1
                pixels.append(row)
            return pixels, info, [(0,0,0),(255,255,255)], meta

        # 8-bit indexed
        if header.bits_per_pixel == 8 and header.n_planes == 1:
            palette = read_vga_palette_manual(path)
            pixels = []
            for y in range(height):
                line = decoded_rows[y][:header.bytes_per_line]
                row = []
                for x in range(width):
                    idx = line[x]
                    if palette:
                        r,g,b = palette[idx]
                    else:
                        r=g=b=idx
                    row.append(f"#{r:02x}{g:02x}{b:02x}")
                pixels.append(row)
            if not palette:
                palette = [(i,i,i) for i in range(256)]
            return pixels, info, palette, meta

        # 24-bit RGB (8bpp x 3 planes)
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
            return pixels, info, None, meta

        raise NotImplementedError(f"Unsupported PCX format: {header.bits_per_pixel}-bit, {header.n_planes} plane(s)")

# =============================== SHARED UTILITIES (Guides 3–5) ===============================

def hex_to_rgb(h: str) -> Tuple[int,int,int]:
    if h and h.startswith("#") and len(h)==7:
        return int(h[1:3],16), int(h[3:5],16), int(h[5:7],16)
    return 0,0,0

def rgb_rows_from_hex_rows(hex_rows: List[List[str]]) -> List[List[Tuple[int,int,int]]]:
    return [[hex_to_rgb(px) for px in row] for row in hex_rows]

def build_photoimage_from_rgb_rows(rgb_rows: List[List[Tuple[int,int,int]]]) -> tk.PhotoImage:
    h = len(rgb_rows)
    w = len(rgb_rows[0]) if h else 0
    img = tk.PhotoImage(width=w, height=h)
    for y,row in enumerate(rgb_rows):
        row_str = " ".join(f"#{r:02x}{g:02x}{b:02x}" for (r,g,b) in row)
        img.put("{" + row_str + "}", to=(0,y))
    return img

def split_channels(rgb_rows):
    r_rows = [[(r,0,0) for (r,_,_) in row] for row in rgb_rows]
    g_rows = [[(0,g,0) for (_,g,_) in row] for row in rgb_rows]
    b_rows = [[(0,0,b) for (_,_,b) in row] for row in rgb_rows]
    return r_rows, g_rows, b_rows

def to_gray_rows(rgb_rows):
    gray = []
    for row in rgb_rows:
        prow=[]
        for (r,g,b) in row:
            s = (r+g+b)//3
            prow.append((s,s,s))
        gray.append(prow)
    return gray

def hist_gray(gray_rows):
    h = [0]*256
    for row in gray_rows:
        for (v,_,_) in row:
            h[v]+=1
    return h

def hist_rgb(rgb_rows):
    r=[0]*256; g=[0]*256; b=[0]*256
    for row in rgb_rows:
        for (rr,gg,bb) in row:
            r[rr]+=1; g[gg]+=1; b[bb]+=1
    return r,g,b

def popup_histogram(parent, title, hist=None, rgb=None):
    top = tk.Toplevel(parent); top.title(title)
    c = tk.Canvas(top, width=800, height=240, bg="white", highlightthickness=1, highlightbackground="#ccc")
    c.pack(padx=8,pady=8)
    def draw_hist(h, color="black"):
        if not h or sum(h)==0:
            c.create_text(400,120,text="No data",fill="gray"); return
        H=200; W=760; ox=20; oy=210
        maxv = max(h) or 1
        bw = W/len(h)
        for i,count in enumerate(h):
            x0 = int(ox + i*bw)
            x1 = int(ox + (i+1)*bw)
            if x1<=x0: x1=x0+1
            hh = int((count/maxv)*H)
            c.create_rectangle(x0, oy-hh, x1, oy, outline=color, fill=color)
        # axis + ticks
        c.create_line(ox, oy, ox+W, oy, fill="black")
        for t in (0,64,128,192,255):
            tx = int(ox + (t/255.0)*W)
            c.create_line(tx, oy, tx, oy-6, fill="black")
            c.create_text(tx, oy-10, text=str(t), anchor="s")
    if hist is not None:
        draw_hist(hist, "gray")
    if rgb is not None:
        r,g,b = rgb
        draw_hist(r, "red"); draw_hist(g,"green"); draw_hist(b,"blue")

# =============================== POINT PROCESSING (Guide 4) ===============================

def img_negative(rgb_rows):
    return [[(255-r,255-g,255-b) for (r,g,b) in row] for row in rgb_rows]

def img_threshold_bw(rgb_rows, t: int):
    t = max(0,min(255,int(t)))
    out=[]
    for row in rgb_rows:
        prow=[]
        for (r,g,b) in row:
            s = (r+g+b)//3
            v = 255 if s>=t else 0
            prow.append((v,v,v))
        out.append(prow)
    return out

def img_gamma(rgb_rows, gamma: float):
    if gamma<=0: gamma=1.0
    lut = [min(255, max(0, int(round(255*((i/255.0)**gamma))))) for i in range(256)]
    return [[(lut[r], lut[g], lut[b]) for (r,g,b) in row] for row in rgb_rows]

def hist_equalize_grayscale(rgb_rows):
    g = to_gray_rows(rgb_rows)
    h = hist_gray(g); total = sum(h)
    if total==0: return g, h
    pdf = [x/total for x in h]
    cdf=[]; c=0.0
    for p in pdf:
        c+=p; cdf.append(c)
    mp = [min(255,max(0,int(round(255*cc)))) for cc in cdf]
    eq=[]
    for row in g:
        prow=[]
        for (v,_,_) in row:
            w = mp[v]
            prow.append((w,w,w))
        eq.append(prow)
    return eq, hist_gray(eq)

# =============================== SPATIAL FILTERS (Guide 5) ===============================

def pad_gray(gray_rows, pad=1):
    H=len(gray_rows); W=len(gray_rows[0]) if H else 0
    P=[[ (0,0,0) for _ in range(W+2*pad)] for _ in range(H+2*pad)]
    for y in range(H):
        for x in range(W):
            P[y+pad][x+pad]=gray_rows[y][x]
    return P

def box3x3(gray_rows):
    P = pad_gray(gray_rows,1)
    H=len(gray_rows); W=len(gray_rows[0])
    out=[]
    for y in range(H):
        prow=[]
        for x in range(W):
            s=0
            for j in (-1,0,1):
                for i in (-1,0,1):
                    s += P[y+1+j][x+1+i][0]
            v = s//9
            prow.append((v,v,v))
        out.append(prow)
    return out

def median3x3(gray_rows):
    P = pad_gray(gray_rows,1)
    H=len(gray_rows); W=len(gray_rows[0])
    out=[]
    for y in range(H):
        prow=[]
        for x in range(W):
            win=[]
            for j in (-1,0,1):
                for i in (-1,0,1):
                    win.append(P[y+1+j][x+1+i][0])
            win.sort()
            v = win[4]
            prow.append((v,v,v))
        out.append(prow)
    return out

def laplacian(gray_rows, kernel="4n"):
    # 4-neighbor: [[0,-1,0],[-1,4,-1],[0,-1,0]]
    # 8-neighbor: [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]
    if kernel=="8n":
        K=[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]
    else:
        K=[[0,-1,0],[-1,4,-1],[0,-1,0]]
    P = pad_gray(gray_rows,1)
    H=len(gray_rows); W=len(gray_rows[0])
    out=[]
    for y in range(H):
        prow=[]
        for x in range(W):
            s=0
            for j in range(3):
                for i in range(3):
                    s += K[j][i]*P[y+j][x+i][0]
            v = max(0,min(255, s+128))  # showable high-pass (shifted)
            prow.append((v,v,v))
        out.append(prow)
    return out

def unsharp_mask(gray_rows):
    blur = box3x3(gray_rows)
    H=len(gray_rows); W=len(gray_rows[0])
    out=[]
    for y in range(H):
        prow=[]
        for x in range(W):
            g = gray_rows[y][x][0]
            b = blur[y][x][0]
            s = g + (g - b)  # amount=1.0
            v = max(0,min(255,int(round(s))))
            prow.append((v,v,v))
        out.append(prow)
    return out

def highboost(gray_rows, A=1.5):
    blur = box3x3(gray_rows)
    H=len(gray_rows); W=len(gray_rows[0])
    out=[]
    for y in range(H):
        prow=[]
        for x in range(W):
            g = gray_rows[y][x][0]
            b = blur[y][x][0]
            s = A*g - (A-1)*b   # g + (A-1)*(g-b)
            v = max(0,min(255,int(round(s))))
            prow.append((v,v,v))
        out.append(prow)
    return out

def sobel_magnitude(gray_rows):
    Gx=[[-1,0,1],[-2,0,2],[-1,0,1]]
    Gy=[[-1,-2,-1],[0,0,0],[1,2,1]]
    P=pad_gray(gray_rows,1)
    H=len(gray_rows); W=len(gray_rows[0])
    out=[]
    for y in range(H):
        prow=[]
        for x in range(W):
            sx=0; sy=0
            for j in range(3):
                for i in range(3):
                    v = P[y+j][x+i][0]
                    sx += Gx[j][i]*v
                    sy += Gy[j][i]*v
            mag = int(min(255, math.hypot(sx,sy)))
            prow.append((mag,mag,mag))
        out.append(prow)
    return out

# =============================== GUI (Shared for Guides 1–5) ===============================

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CMSC 162 – Integrated Project 1 (Guides 1–5)")
        self.geometry("1180x900")
        self.configure(bg="white")

        # state
        self._orig_hex = None
        self._orig_rgb = None
        self._meta = {"bits_per_pixel": None, "n_planes": None}
        self._current_rgb = None
        self._palette = None

        # =================== MAIN SCROLLABLE CONTAINER ===================
        main_canvas = tk.Canvas(self, bg="white", highlightthickness=0)
        main_scroll = ttk.Scrollbar(self, orient="vertical", command=main_canvas.yview)
        main_canvas.configure(yscrollcommand=main_scroll.set)
        main_canvas.pack(side="left", fill="both", expand=True)
        main_scroll.pack(side="right", fill="y")

        main_frame = ttk.Frame(main_canvas)
        main_canvas.create_window((0, 0), window=main_frame, anchor="nw")
        main_frame.bind("<Configure>", lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all")))

        # =================== TOP BUTTONS ===================
        top = ttk.Frame(main_frame)
        top.pack(fill="x", padx=10, pady=8)
        ttk.Button(top, text="Open PCX (Guide 2)", command=self.open_pcx).pack(side="left")
        ttk.Button(top, text="Channels (Guide 3)", command=self.show_channels).pack(side="left", padx=6)
        ttk.Button(top, text="Histograms (pop-up)", command=self.popup_hist).pack(side="left", padx=6)

        pp = ttk.Frame(main_frame)
        pp.pack(fill="x", padx=10, pady=8)
        ttk.Label(pp, text="Point Processing (Guide 4):", font=("Segoe UI", 10, "bold")).pack(side="left", padx=(0, 10))
        ttk.Button(pp, text="Grayscale + Hist", command=self.do_gray).pack(side="left", padx=3)
        ttk.Button(pp, text="Negative", command=self.do_negative).pack(side="left", padx=3)
        ttk.Button(pp, text="Threshold (0..255)", command=self.do_thresh).pack(side="left", padx=3)
        ttk.Button(pp, text="Gamma", command=self.do_gamma).pack(side="left", padx=3)
        ttk.Button(pp, text="Hist. Equalize (gray)", command=self.do_equalize).pack(side="left", padx=3)

        sp = ttk.Frame(main_frame)
        sp.pack(fill="x", padx=10, pady=(2, 10))
        ttk.Label(sp, text="Spatial Filters (Guide 5):", font=("Segoe UI", 10, "bold")).pack(side="left", padx=(0, 10))
        ttk.Button(sp, text="Averaging (3x3)", command=self.do_avg).pack(side="left", padx=3)
        ttk.Button(sp, text="Median (3x3)", command=self.do_median).pack(side="left", padx=3)
        ttk.Button(sp, text="Laplacian (4-neigh)", command=self.do_lap4).pack(side="left", padx=3)
        ttk.Button(sp, text="Unsharp", command=self.do_unsharp).pack(side="left", padx=3)
        ttk.Button(sp, text="High-boost", command=self.do_highboost).pack(side="left", padx=3)
        ttk.Button(sp, text="Sobel magnitude", command=self.do_sobel).pack(side="left", padx=3)

        # =================== IMAGE DISPLAY AREA ===================
        ttk.Label(main_frame, text="Image View", font=("Segoe UI", 10, "bold")).pack()
        self.canvas = tk.Canvas(main_frame, bg="white", width=800, height=600)
        self.canvas.pack(pady=6)

        # =================== SCROLLABLE INFO + PALETTE AREA ===================
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill="both", expand=True, padx=8, pady=(8, 10))

        canvas = tk.Canvas(bottom_frame, bg="white", highlightthickness=0)
        scrollbar = ttk.Scrollbar(bottom_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # header info + palette inside scrollable area
        info_palette_row = ttk.Frame(scrollable_frame)
        info_palette_row.pack(fill="x", expand=True, padx=10, pady=5)

        left_info = ttk.Frame(info_palette_row)
        left_info.pack(side="left", fill="both", expand=True, padx=(0, 10))
        right_palette = ttk.Frame(info_palette_row)
        right_palette.pack(side="left", fill="y")

        ttk.Label(left_info, text="Header Info", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.info = tk.Text(left_info, height=10, wrap="word", bg="#f7f7f7",
                            relief="flat", font=("Consolas", 10))
        self.info.pack(fill="x", expand=True, pady=(2, 0))

        ttk.Label(right_palette, text="Palette (8-bit indexed only)", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.palette_frame = ttk.Frame(right_palette)
        self.palette_frame.pack()
        self.palette_canvas = tk.Canvas(self.palette_frame, bg="white",
                                        highlightthickness=1, highlightbackground="#ccc")
        self.palette_canvas.pack()

        # enable mousewheel scrolling
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(-1 * int(e.delta / 120), "units"))

    # ---------- helpers ----------
    def ensure_image(self):
        if self._orig_rgb is None:
            messagebox.showwarning("No image", "Open a PCX file first.")
            return False
        return True

    def show_image(self, rgb_rows, set_current=True):
        img = build_photoimage_from_rgb_rows(rgb_rows)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=img)
        self.canvas.image = img
        if set_current:
            self._current_rgb = rgb_rows
        # hide palette panel for processed truecolor; show only for original 8-bit indexed view
        show_palette = (self._palette is not None and self._current_rgb is self._orig_rgb and
                        self._meta.get("bits_per_pixel") == 8 and self._meta.get("n_planes") == 1)
        self.palette_frame.pack_forget()
        if show_palette:
            self.palette_frame.pack()

    def draw_palette(self, palette):
        self.palette_canvas.delete("all")
        if not palette:
            return
        PAD, cell, cols = 8, 16, 16
        total = min(256, len(palette))
        rows = (total + cols - 1) // cols
        w, h = cols * cell + 2 * PAD, rows * cell + 2 * PAD
        self.palette_canvas.config(width=w, height=h)
        for i, (r, g, b) in enumerate(palette[:total]):
            col = i % cols
            row = i // cols
            x = PAD + col * cell
            y = PAD + row * cell
            self.palette_canvas.create_rectangle(
                x, y, x + cell, y + cell,
                fill=f"#{r:02x}{g:02x}{b:02x}", outline=""
            )

    # ---------- actions ----------
    def open_pcx(self):
        path = filedialog.askopenfilename(filetypes=[("PCX files", "*.pcx")])
        if not path:
            return
        try:
            pixels, info, palette, meta = decode_pcx(Path(path))
            self._meta = meta
            self._palette = palette
            self._orig_hex = pixels
            self._orig_rgb = rgb_rows_from_hex_rows(pixels)
            # header info
            self.info.delete(1.0, tk.END)
            for k, v in info.items():
                self.info.insert(tk.END, f"{k}: {v}\n")
            # palette
            self.draw_palette(palette)
            # image
            self.show_image(self._orig_rgb, set_current=True)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to decode PCX:\n{e}")

    def show_channels(self):
        if not self.ensure_image():
            return
        rows_r, rows_g, rows_b = split_channels(self._orig_rgb)
        win = tk.Toplevel(self)
        win.title("R / G / B Channels")
        for i, rows in enumerate((rows_r, rows_g, rows_b)):
            c = tk.Canvas(win, bg="white")
            c.grid(row=0, column=i, padx=6, pady=6)
            img = build_photoimage_from_rgb_rows(rows)
            c.create_image(0, 0, anchor="nw", image=img)
            c.image = img
            ttk.Label(win, text=("Red", "Green", "Blue")[i]).grid(row=1, column=i)

    def popup_hist(self):
        if not self.ensure_image():
            return
        rh, gh, bh = hist_rgb(self._current_rgb)
        popup_histogram(self, "Histogram (RGB combined)", rgb=(rh, gh, bh))

    # Guide 4
    def do_gray(self):
        if not self.ensure_image(): return
        g = to_gray_rows(self._orig_rgb)
        self.show_image(g)
        popup_histogram(self, "Grayscale histogram", hist=hist_gray(g))

    def do_negative(self):
        if not self.ensure_image(): return
        neg = img_negative(self._orig_rgb)
        self.show_image(neg)
        popup_histogram(self, "Negative – RGB hist", rgb=hist_rgb(neg))

    def do_thresh(self):
        if not self.ensure_image(): return
        t = simpledialog.askinteger("Threshold", "Enter threshold (0..255):", minvalue=0, maxvalue=255)
        if t is None: return
        bw = img_threshold_bw(self._orig_rgb, t)
        self.show_image(bw)
        popup_histogram(self, f"B/W histogram (t={t})", hist=hist_gray(bw))

    def do_gamma(self):
        if not self.ensure_image(): return
        g = simpledialog.askfloat("Gamma", "Enter gamma (>0, e.g., 0.5, 1.0, 2.2):")
        if g is None: return
        try:
            g = float(g); 
            if g<=0: raise ValueError
        except Exception:
            messagebox.showerror("Invalid", "Gamma must be a positive number."); return
        img = img_gamma(self._orig_rgb, g)
        self.show_image(img)
        popup_histogram(self, f"Gamma={g} – RGB hist", rgb=hist_rgb(img))

    def do_equalize(self):
        if not self.ensure_image(): return
        eq, h = hist_equalize_grayscale(self._orig_rgb)
        self.show_image(eq)
        popup_histogram(self, "Histogram Equalization (gray)", hist=h)

    # Guide 5
    def do_avg(self):
        if not self.ensure_image(): return
        g = to_gray_rows(self._orig_rgb)
        out = box3x3(g)
        self.show_image(out)
        popup_histogram(self, "Averaging (3x3) – gray hist", hist=hist_gray(out))

    def do_median(self):
        if not self.ensure_image(): return
        g = to_gray_rows(self._orig_rgb)
        out = median3x3(g)
        self.show_image(out)
        popup_histogram(self, "Median (3x3) – gray hist", hist=hist_gray(out))

    def do_lap4(self):
        if not self.ensure_image(): return
        g = to_gray_rows(self._orig_rgb)
        out = laplacian(g, kernel="4n")
        self.show_image(out)
        popup_histogram(self, "Laplacian (4-neigh) – gray hist", hist=hist_gray(out))

    def do_unsharp(self):
        if not self.ensure_image(): return
        g = to_gray_rows(self._orig_rgb)
        out = unsharp_mask(g)
        self.show_image(out)
        popup_histogram(self, "Unsharp – gray hist", hist=hist_gray(out))

    def do_highboost(self):
        if not self.ensure_image(): return
        g = to_gray_rows(self._orig_rgb)
        A = simpledialog.askfloat("High-boost", "Enter amplification A (>1, e.g., 1.5):", minvalue=1.01)
        if A is None: return
        out = highboost(g, A=A)
        self.show_image(out)
        popup_histogram(self, f"High-boost (A={A}) – gray hist", hist=hist_gray(out))

    def do_sobel(self):
        if not self.ensure_image(): return
        g = to_gray_rows(self._orig_rgb)
        out = sobel_magnitude(g)
        self.show_image(out)
        popup_histogram(self, "Sobel magnitude – gray hist", hist=hist_gray(out))

if __name__ == "__main__":
    App().mainloop()
