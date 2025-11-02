#!/usr/bin/env python3
"""
CMSC 162 – Project 1 Guide 4 Extension (Improved)
Point-Processing Methods added to the Manual PCX RLE Viewer:
 - Grayscale transform s = (R+G+B)/3 (and histogram)
 - Negative of image
 - Black/White via manual threshold (with user input 0..255)
 - Power-law (gamma) transform (with user input gamma)
 - Histogram Equalization (grayscale; follows lecture steps)

Improvements:
 - Palette canvas in UI
 - draw_palette() implemented (shows 2-color or 16x16 VGA palette grid)
 - draw_palette() called on image load
"""
from __future__ import annotations
import struct, os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

# ---------------- PCX decoding ----------------

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
    """
    Returns 256*RGB palette if the PCX has the VGA marker at file end.
    For 1-bit or 24-bit images, returns None.
    """
    try:
        data = Path(path).read_bytes()
        # VGA palette at file end: 0x0C marker + 768 bytes (256 * 3)
        if len(data) < 769 or data[-769] != 0x0C:
            return None
        raw = data[-768:]
        return [tuple(raw[i:i+3]) for i in range(0, 768, 3)]
    except Exception:
        return None


def decode_pcx(path: Path):
    """
    Return (pixels_rows_hex, info_dict, palette_or_None, is_indexed8)
    pixels_rows_hex: list of rows; each row is list of "#rrggbb" strings
    is_indexed8: True only for 8-bit indexed images with VGA palette marker
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

        # ---- 1-bit Black & White ----
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
            # Not considered “8-bit indexed” for palette view purposes.
            return pixels, info, [(0, 0, 0), (255, 255, 255)], False

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
            # is_indexed8 = True only when a VGA palette is really present
            return pixels, info, (palette if palette else [(i, i, i) for i in range(256)]), (palette is not None)

        # ---- 24-bit RGB (3 planes, 8 bits per pixel per plane) ----
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
            return pixels, info, None, False

        raise NotImplementedError(
            f"Unsupported PCX format: {header.bits_per_pixel}-bit, {header.n_planes} plane(s)"
        )

# ---------------- Helpers: conversions, histograms, drawing ----------------

def hex_to_rgb(hexstr: str) -> Tuple[int,int,int]:
    if hexstr and hexstr.startswith("#") and len(hexstr) == 7:
        r = int(hexstr[1:3], 16)
        g = int(hexstr[3:5], 16)
        b = int(hexstr[5:7], 16)
        return r, g, b
    return 0, 0, 0

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

def compute_rgb_histograms(rgb_rows: List[List[Tuple[int,int,int]]]):
    rhist = [0]*256; ghist = [0]*256; bhist = [0]*256
    for row in rgb_rows:
        for (r,g,b) in row:
            rhist[r] += 1; ghist[g] += 1; bhist[b] += 1
    return rhist, ghist, bhist

def compute_grayscale_rows_and_hist(rgb_rows: List[List[Tuple[int,int,int]]]):
    gray_rows = []; ghist = [0]*256
    for row in rgb_rows:
        prow = []
        for (r,g,b) in row:
            s = (r + g + b) // 3
            prow.append((s,s,s))
            ghist[s] += 1
        gray_rows.append(prow)
    return gray_rows, ghist

def draw_histogram_on_canvas(canvas: tk.Canvas, hist: List[int],
                             width: int = 800, height: int = 220,
                             color: str = "black", draw_grid: bool = True):
    canvas.delete("all")
    canvas.config(width=width, height=height)
    if not hist or sum(hist) == 0:
        canvas.create_text(width//2, height//2, text="No data", fill="gray"); return
    max_count = max(hist) if max(hist) > 0 else 1
    bins = len(hist); bin_w = float(width) / bins
    if draw_grid:
        levels = 5
        for i in range(1, levels):
            y = int(height - 1 - (i / levels) * (height - 6))
            canvas.create_line(0, y, width, y, fill="#e6e6e6")
    for i, count in enumerate(hist):
        x0 = int(round(i * bin_w))
        x1 = int(round((i + 1) * bin_w))
        if x1 <= x0: x1 = x0 + 1
        h = int((count / max_count) * (height - 6))
        y0 = height - 1 - h; y1 = height - 1
        canvas.create_rectangle(x0, y0, x1, y1, outline=color, fill=color)
    canvas.create_line(0, height-1, width, height-1, fill="black")
    for tick in (0, 64, 128, 192, 255):
        tx = int(round((tick / 255.0) * (width - 1)))
        canvas.create_line(tx, height-1, tx, height-5, fill="black")
        canvas.create_text(tx, height-12, text=str(tick), anchor="s", font=("TkDefaultFont", 8))

def draw_combined_histogram(canvas: tk.Canvas, rhist: List[int], ghist: List[int], bhist: List[int],
                            width: int = 800, height: int = 220, draw_grid: bool = True):
    canvas.delete("all")
    canvas.config(width=width, height=height)
    if not (rhist and ghist and bhist) or (sum(rhist)+sum(ghist)+sum(bhist) == 0):
        canvas.create_text(width//2, height//2, text="No data", fill="gray"); return
    max_count = max(max(rhist), max(ghist), max(bhist), 1)
    bins = len(rhist); bin_w = float(width) / bins; sub_w = bin_w / 3.0
    if draw_grid:
        levels = 5
        for i in range(1, levels):
            y = int(height - 1 - (i / levels) * (height - 6))
            canvas.create_line(0, y, width, y, fill="#e6e6e6")
    colors = ("red", "green", "blue")
    for i in range(bins):
        base_x0 = i * bin_w
        for j, hist in enumerate((rhist, ghist, bhist)):
            x0 = int(round(base_x0 + j * sub_w))
            x1 = int(round(base_x0 + (j + 1) * sub_w))
            if x1 <= x0: x1 = x0 + 1
            count = hist[i]
            h = int((count / max_count) * (height - 6))
            y0 = height - 1 - h; y1 = height - 1
            canvas.create_rectangle(x0, y0, x1, y1, outline=colors[j], fill=colors[j])
    canvas.create_line(0, height-1, width, height-1, fill="black")
    for tick in (0, 64, 128, 192, 255):
        tx = int(round((tick / 255.0) * (width - 1)))
        canvas.create_line(tx, height-1, tx, height-5, fill="black")
        canvas.create_text(tx, height-12, text=str(tick), anchor="s", font=("TkDefaultFont", 8))

# ---------------- Point-processing transforms ----------------

def apply_negative(rgb_rows: List[List[Tuple[int,int,int]]]):
    return [[(255-r, 255-g, 255-b) for (r,g,b) in row] for row in rgb_rows]

def apply_threshold_bw(rgb_rows: List[List[Tuple[int,int,int]]], t: int):
    """Threshold on grayscale avg; output pure black/white."""
    t = max(0, min(255, int(t)))
    out = []
    for row in rgb_rows:
        prow = []
        for (r,g,b) in row:
            s = (r+g+b)//3
            v = 255 if s >= t else 0
            prow.append((v, v, v))
        out.append(prow)
    return out

def apply_gamma(rgb_rows: List[List[Tuple[int,int,int]]], gamma: float):
    """Power-law per channel: s = 255 * (r/255)^gamma (and same for g,b)."""
    if gamma <= 0: gamma = 1.0
    lut = [min(255, max(0, int(round(255 * ((i/255.0) ** gamma))))) for i in range(256)]
    out = []
    for row in rgb_rows:
        prow = []
        for (r,g,b) in row:
            prow.append((lut[r], lut[g], lut[b]))
        out.append(prow)
    return out

def histogram_equalization_grayscale(rgb_rows: List[List[Tuple[int,int,int]]]):
    """Equalize the grayscale version (classic steps)."""
    gray_rows, hist = compute_grayscale_rows_and_hist(rgb_rows)
    total = sum(hist)
    if total == 0:  # degenerate
        return gray_rows, hist
    # PDF + CDF
    cdf, csum = [], 0.0
    for h in hist:
        csum += (h/total)
        cdf.append(csum)
    # mapping
    mp = [min(255, max(0, int(round(255 * c)))) for c in cdf]
    eq_rows = []
    for row in gray_rows:
        prow = []
        for (s,_,_) in row:  # s==g==b
            v = mp[s]
            prow.append((v, v, v))
        eq_rows.append(prow)
    _, eq_hist = compute_grayscale_rows_and_hist(eq_rows)
    return eq_rows, eq_hist

# ---------------- GUI ----------------

class PCXViewerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PCX Viewer – Guide 4 (Pop-ups + Auto-Hiding Palette)")
        self.geometry("1120x960")
        self.configure(bg="white")

        self._rgb_rows: Optional[List[List[Tuple[int,int,int]]]] = None
        self._hex_rows: Optional[List[List[str]]] = None
        self._img_width = 0
        self._img_height = 0
        self._is_indexed8 = False  # true only when original is 8-bit indexed with VGA palette
        self._last_palette = None

        # Main scrollable container
        main_canvas = tk.Canvas(self, bg="white", highlightthickness=0)
        main_scroll = ttk.Scrollbar(self, orient="vertical", command=main_canvas.yview)
        main_frame = ttk.Frame(main_canvas)
        main_canvas.configure(yscrollcommand=main_scroll.set)
        main_canvas.pack(side="left", fill="both", expand=True)
        main_scroll.pack(side="right", fill="y")
        main_canvas.create_window((0, 0), window=main_frame, anchor="nw")
        main_frame.bind("<Configure>", lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all")))

        # Top row (open + actions)
        top_row = ttk.Frame(main_frame)
        top_row.pack(pady=10, fill="x", padx=8)
        ttk.Button(top_row, text="Open PCX File", command=self.load_pcx).pack(side="left", padx=(0,8))
        ttk.Label(top_row, text="Point Processing:", font=("Segoe UI", 10, "bold")).pack(side="left", padx=(8,10))
        ttk.Button(top_row, text="Grayscale + Hist", command=self.ui_grayscale_popup).pack(side="left", padx=4)
        ttk.Button(top_row, text="Negative", command=self.ui_negative_popup).pack(side="left", padx=4)
        ttk.Button(top_row, text="Threshold (B/W)", command=self.ui_threshold_popup).pack(side="left", padx=4)
        ttk.Button(top_row, text="Power-Law (Gamma)", command=self.ui_gamma_popup).pack(side="left", padx=4)
        ttk.Button(top_row, text="Hist. Equalization", command=self.ui_equalize_popup).pack(side="left", padx=4)

        # Image
        ttk.Label(main_frame, text="Original Image", font=("Segoe UI", 10, "bold")).pack()
        self.canvas = tk.Canvas(main_frame, bg="white", width=640, height=480)
        self.canvas.pack(pady=6)

        # Header info
        ttk.Label(main_frame, text="Header Information", font=("Segoe UI", 10, "bold")).pack()
        self.info_text = tk.Text(main_frame, height=9, wrap="word", bg="#f7f7f7", relief="flat", font=("Consolas", 10))
        self.info_text.pack(fill="x", padx=8, pady=4)

        # Palette section (wrapped in a frame so we can show/hide together)
        self.palette_section = ttk.Frame(main_frame)
        ttk.Label(self.palette_section, text="Color Palette (Source)", font=("Segoe UI", 10, "bold")).pack(pady=(10,0))
        self.palette_canvas = tk.Canvas(self.palette_section, bg="white", highlightthickness=1, highlightbackground="#ccc")
        self.palette_canvas.pack(pady=6)
        # Initially hidden until we know the file type
        self.palette_section.pack_forget()

    # ---------- Palette helpers ----------
    def show_palette_section(self):
        try:
            self.palette_section.pack(pady=(8, 4))
        except Exception:
            pass

    def hide_palette_section(self):
        try:
            self.palette_section.pack_forget()
        except Exception:
            pass

    def draw_palette(self, palette: Optional[List[Tuple[int,int,int]]]):
        """Draws the source palette (only for 8-bit indexed with VGA palette)."""
        c = self.palette_canvas
        c.delete("all")

        PAD = 8
        if not (self._is_indexed8 and palette):
            # no palette to show
            self.hide_palette_section()
            return

        # 16xN grid up to 256 entries
        cols = 16
        cell = 16
        total = min(256, len(palette))
        rows = (total + cols - 1) // cols
        grid_w = cols * cell
        grid_h = rows * cell
        c.config(width=grid_w + 2*PAD, height=grid_h + 2*PAD)
        for i, (r,g,b) in enumerate(palette[:total]):
            col = i % cols
            row = i // cols
            x = PAD + col * cell
            y = PAD + row * cell
            color = f"#{r:02x}{g:02x}{b:02x}"
            c.create_rectangle(x, y, x + cell, y + cell, fill=color, outline="")
        c.create_rectangle(PAD-1, PAD-1, PAD+grid_w+1, PAD+grid_h+1, outline="#ccc")
        self.show_palette_section()

    # ---------- Load file ----------
    def load_pcx(self):
        path = filedialog.askopenfilename(filetypes=[("PCX files", "*.pcx")])
        if not path: return
        try:
            pixels, info, palette, is_indexed8 = decode_pcx(Path(path))
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
            self._is_indexed8 = bool(is_indexed8)
            self._last_palette = palette if is_indexed8 else None

            # header info
            self.info_text.delete(1.0, tk.END)
            for k, v in info.items():
                self.info_text.insert(tk.END, f"{k}: {v}\n")

            # Palette visibility
            if self._is_indexed8:
                self.draw_palette(self._last_palette)
            else:
                self.hide_palette_section()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to decode PCX:\n{e}")

    def ensure_image_loaded(self) -> bool:
        if not self._rgb_rows:
            messagebox.showwarning("No image", "Load a PCX image first.")
            return False
        return True

    # ---------- Pop-up result windows ----------
    def _popup_image_with_hist(self, rgb_rows, title: str, is_truecolor: bool,
                               hist_mode: str = "combined", hist=None):
        """
        Create a Toplevel showing processed image + histogram.
        hist_mode: "combined" (RGB) or "gray"
        hist: for combined, pass (rhist, ghist, bhist); for gray, pass ghist
        is_truecolor: if True, palette panel is not shown (there is none).
        """
        win = tk.Toplevel(self)
        win.title(title)

        # image
        img = build_photoimage_from_rgb_rows(rgb_rows)
        h = len(rgb_rows)
        w = len(rgb_rows[0]) if h else 0

        ttk.Label(win, text=title, font=("Segoe UI", 10, "bold")).pack(pady=(8,2))
        can = tk.Canvas(win, bg="white", width=w, height=h)
        can.pack(padx=8, pady=(0,8))
        can.create_image(0, 0, anchor="nw", image=img)
        can.image = img

        # histogram below
        ttk.Label(win, text="Histogram", font=("Segoe UI", 10, "bold")).pack()
        hcan = tk.Canvas(win, bg="white", highlightthickness=1, highlightbackground="#ccc")
        hcan.pack(padx=8, pady=(2,8))

        if hist_mode == "combined":
            rh, gh, bh = hist if hist else compute_rgb_histograms(rgb_rows)
            draw_combined_histogram(hcan, rh, gh, bh, width=800, height=220, draw_grid=True)
        else:
            ghist = hist if hist else compute_grayscale_rows_and_hist(rgb_rows)[1]
            draw_histogram_on_canvas(hcan, ghist, width=800, height=220, color="gray", draw_grid=True)

        # Palette handling for popup (processed views are truecolor => no palette)
        # If ever you decide to preview *original* indexed image in a popup, you can draw its palette similarly.

    # ---------- Actions (each opens a pop-up) ----------
    def ui_grayscale_popup(self):
        if not self.ensure_image_loaded(): return
        gray_rows, ghist = compute_grayscale_rows_and_hist(self._rgb_rows)
        self._popup_image_with_hist(gray_rows, "Grayscale (avg) + Histogram",
                                    is_truecolor=True, hist_mode="gray", hist=ghist)

    def ui_negative_popup(self):
        if not self.ensure_image_loaded(): return
        neg = apply_negative(self._rgb_rows)
        rh, gh, bh = compute_rgb_histograms(neg)
        self._popup_image_with_hist(neg, "Negative (RGB) + Histogram",
                                    is_truecolor=True, hist_mode="combined", hist=(rh, gh, bh))

    def ui_threshold_popup(self):
        if not self.ensure_image_loaded(): return
        t = simpledialog.askinteger("Threshold", "Enter threshold (0..255):", minvalue=0, maxvalue=255)
        if t is None: return
        bw = apply_threshold_bw(self._rgb_rows, t)
        _, ghist = compute_grayscale_rows_and_hist(bw)
        self._popup_image_with_hist(bw, f"Threshold B/W (t={t}) + Histogram",
                                    is_truecolor=True, hist_mode="gray", hist=ghist)

    def ui_gamma_popup(self):
        if not self.ensure_image_loaded(): return
        g = simpledialog.askfloat("Gamma", "Enter gamma (>0, e.g., 0.5, 1.0, 2.2):")
        if g is None: return
        try:
            g = float(g)
            if g <= 0: raise ValueError
        except Exception:
            messagebox.showerror("Invalid", "Gamma must be a positive number."); return
        gamma_img = apply_gamma(self._rgb_rows, g)
        rh, gh, bh = compute_rgb_histograms(gamma_img)
        self._popup_image_with_hist(gamma_img, f"Power-Law (gamma={g}) + Histogram",
                                    is_truecolor=True, hist_mode="combined", hist=(rh, gh, bh))

    def ui_equalize_popup(self):
        if not self.ensure_image_loaded(): return
        eq_rows, eq_hist = histogram_equalization_grayscale(self._rgb_rows)
        self._popup_image_with_hist(eq_rows, "Histogram Equalization (grayscale) + Histogram",
                                    is_truecolor=True, hist_mode="gray", hist=eq_hist)

if __name__ == "__main__":
    app = PCXViewerApp()
    app.mainloop()

