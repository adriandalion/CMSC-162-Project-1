#!/usr/bin/env python3
"""
Manual PCX RLE Decoder GUI (Tkinter) - Scrollable Version
Complete file with:
 - PCX decoding (1-bit, 8-bit, 24-bit) — no Pillow
 - Scrollable UI
 - Split into R, G, B channels (display)
 - Histograms for each channel (fixed drawing so bars show)
 - Grayscale transform s = (R+G+B)/3 and its histogram
 - Channel histogram window supports Combined (RGB) view and single-channel views
"""
from __future__ import annotations
import struct
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

# Keep the header format consistent with original code
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
        # VGA palette appended at end of PCX: 0x0C marker then 768 bytes
        if len(data) < 769 or data[-769] != 0x0C:
            return None
        raw = data[-768:]
        return [tuple(raw[i:i+3]) for i in range(0, 768, 3)]
    except Exception:
        return None


def decode_pcx(path: Path):
    """Return (pixels_rows_hex, info_dict, palette_or_None)
    pixels_rows_hex: list of rows; each row is list of "#rrggbb" strings
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
                # each plane has bytes_per_line bytes; take first 'width' from each plane
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
# Helper functions for channels/histograms/grayscale
# =============================================================

def hex_to_rgb(hexstr: str) -> Tuple[int, int, int]:
    """Convert '#rrggbb' to (r,g,b)."""
    if hexstr and hexstr.startswith("#") and len(hexstr) == 7:
        r = int(hexstr[1:3], 16)
        g = int(hexstr[3:5], 16)
        b = int(hexstr[5:7], 16)
        return r, g, b
    return 0, 0, 0

def rgb_rows_from_hex_rows(hex_rows: List[List[str]]) -> List[List[Tuple[int,int,int]]]:
    """Convert rows of '#rrggbb' into rows of (r,g,b)."""
    return [[hex_to_rgb(px) for px in row] for row in hex_rows]

def build_photoimage_from_rgb_rows(rgb_rows: List[List[Tuple[int,int,int]]]) -> tk.PhotoImage:
    """Create a Tk PhotoImage from rgb rows."""
    height = len(rgb_rows)
    width = len(rgb_rows[0]) if height else 0
    img = tk.PhotoImage(width=width, height=height)
    for y, row in enumerate(rgb_rows):
        # Compose one row string; PhotoImage.put accepts row string expression
        row_str = " ".join(f"#{r:02x}{g:02x}{b:02x}" for (r,g,b) in row)
        img.put("{" + row_str + "}", to=(0, y))
    return img

def split_channels_rgb_rows(rgb_rows: List[List[Tuple[int,int,int]]]):
    """Return three rgb_rows arrays for R, G, B channels (other channels zeroed)."""
    rows_r = [[(r,0,0) for (r,_,_) in row] for row in rgb_rows]
    rows_g = [[(0,g,0) for (_,g,_) in row] for row in rgb_rows]
    rows_b = [[(0,0,b) for (_,_,b) in row] for row in rgb_rows]
    return rows_r, rows_g, rows_b

def compute_rgb_histograms(rgb_rows: List[List[Tuple[int,int,int]]]):
    """Compute histograms arrays for R, G, B channels (length 256 each)."""
    rhist = [0]*256
    ghist = [0]*256
    bhist = [0]*256
    for row in rgb_rows:
        for (r,g,b) in row:
            rhist[r] += 1
            ghist[g] += 1
            bhist[b] += 1
    return rhist, ghist, bhist

def compute_grayscale_rows_and_hist(rgb_rows: List[List[Tuple[int,int,int]]]):
    """Compute grayscale rows and histogram using average (R+G+B)/3)."""
    gray_rows = []
    ghist = [0]*256
    for row in rgb_rows:
        prow = []
        for (r,g,b) in row:
            s = int((r + g + b) / 3)
            prow.append((s,s,s))
            ghist[s] += 1
        gray_rows.append(prow)
    return gray_rows, ghist

def draw_histogram_on_canvas(canvas: tk.Canvas, hist: List[int],
                             width: int = 512, height: int = 200,
                             color: str = "black", draw_grid: bool = True):
    """
    Draw a visible histogram on the canvas (resizes canvas to width x height).
    - hist: list of counts length typically 256
    - color: fill color for bars (e.g. 'red','green','blue','gray' or '#rrggbb')
    - draw_grid: whether to draw light gridlines
    """
    canvas.delete("all")
    canvas.config(width=width, height=height)
    if not hist or sum(hist) == 0:
        canvas.create_text(width//2, height//2, text="No data", fill="gray")
        return

    max_count = max(hist) if max(hist) > 0 else 1
    bins = len(hist)
    bin_w = float(width) / bins

    # optionally draw horizontal grid lines
    if draw_grid:
        levels = 5
        for i in range(1, levels):
            y = int(height - 1 - (i / levels) * (height - 6))
            canvas.create_line(0, y, width, y, fill="#e6e6e6")

    for i, count in enumerate(hist):
        x0 = int(round(i * bin_w))
        x1 = int(round((i + 1) * bin_w))
        if x1 <= x0:
            x1 = x0 + 1
        # scale bar height (leave small top margin)
        h = int((count / max_count) * (height - 6))
        y0 = height - 1 - h
        y1 = height - 1
        # draw filled rectangle using provided color
        canvas.create_rectangle(x0, y0, x1, y1, outline=color, fill=color)

    # draw x-axis
    canvas.create_line(0, height-1, width, height-1, fill="black")
    # tick marks at a few positions
    for tick in (0, 64, 128, 192, 255):
        tx = int(round((tick / 255.0) * (width - 1)))
        canvas.create_line(tx, height-1, tx, height-5, fill="black")
        canvas.create_text(tx, height-12, text=str(tick), anchor="s", font=("TkDefaultFont", 8))

def draw_combined_histogram(canvas: tk.Canvas, rhist: List[int], ghist: List[int], bhist: List[int],
                            width: int = 512, height: int = 200, draw_grid: bool = True):
    """
    Draw combined RGB histogram: for each bin, show three narrow bars (red, green, blue)
    side-by-side. Resizes canvas to width x height.
    """
    canvas.delete("all")
    canvas.config(width=width, height=height)
    if not rhist or not ghist or not bhist or (sum(rhist)+sum(ghist)+sum(bhist) == 0):
        canvas.create_text(width//2, height//2, text="No data", fill="gray")
        return

    max_count = max(max(rhist), max(ghist), max(bhist), 1)
    bins = len(rhist)
    bin_w = float(width) / bins
    sub_w = bin_w / 3.0

    # optional grid lines
    if draw_grid:
        levels = 5
        for i in range(1, levels):
            y = int(height - 1 - (i / levels) * (height - 6))
            canvas.create_line(0, y, width, y, fill="#e6e6e6")

    colors = ("red", "green", "blue")
    for i in range(bins):
        base_x0 = i * bin_w
        # draw each sub-bar
        for j, hist in enumerate((rhist, ghist, bhist)):
            x0 = int(round(base_x0 + j * sub_w))
            x1 = int(round(base_x0 + (j + 1) * sub_w))
            if x1 <= x0:
                x1 = x0 + 1
            count = hist[i]
            h = int((count / max_count) * (height - 6))
            y0 = height - 1 - h
            y1 = height - 1
            canvas.create_rectangle(x0, y0, x1, y1, outline=colors[j], fill=colors[j])

    # axis and ticks
    canvas.create_line(0, height-1, width, height-1, fill="black")
    for tick in (0, 64, 128, 192, 255):
        tx = int(round((tick / 255.0) * (width - 1)))
        canvas.create_line(tx, height-1, tx, height-5, fill="black")
        canvas.create_text(tx, height-12, text=str(tick), anchor="s", font=("TkDefaultFont", 8))


# =============================================================
# SCROLLABLE GUI with added channel/histogram controls
# =============================================================

class PCXViewerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Manual PCX RLE Viewer")
        self.geometry("980x1000")
        self.configure(bg="white")

        # store last decoded image RGB rows and size
        self._rgb_rows: Optional[List[List[Tuple[int,int,int]]]] = None
        self._hex_rows: Optional[List[List[str]]] = None
        self._img_width = 0
        self._img_height = 0

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
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(pady=10, fill="x", padx=8)
        ttk.Button(btn_frame, text="Open PCX File", command=self.load_pcx).pack(side="left")
        ttk.Button(btn_frame, text="Show Channels", command=self.show_channels).pack(side="left", padx=6)
        ttk.Button(btn_frame, text="Show Channel Histograms", command=self.show_channel_histograms).pack(side="left", padx=6)
        ttk.Button(btn_frame, text="Grayscale (avg) & Histogram", command=self.show_grayscale_and_hist).pack(side="left", padx=6)

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

            # store rows for later analysis (hex rows and rgb rows)
            self._hex_rows = pixels
            self._rgb_rows = rgb_rows_from_hex_rows(pixels)
            self._img_width = pw
            self._img_height = ph

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

    # -------------------------
    # New features: channels, histograms, grayscale
    # -------------------------

    def ensure_image_loaded(self):
        if not self._rgb_rows:
            messagebox.showwarning("No image", "Load an image first.")
            return False
        return True

    def show_channels(self):
        """Split into R/G/B and show each channel in a separate Toplevel."""
        if not self.ensure_image_loaded():
            return

        rows_r, rows_g, rows_b = split_channels_rgb_rows(self._rgb_rows)

        top = tk.Toplevel(self)
        top.title("Channels (R, G, B)")
        canvas = tk.Canvas(top, bg="white")
        hbar = ttk.Scrollbar(top, orient="horizontal", command=canvas.xview)
        frame = ttk.Frame(canvas)
        canvas.configure(xscrollcommand=hbar.set)
        canvas.pack(side="top", fill="both", expand=True)
        hbar.pack(side="bottom", fill="x")
        canvas.create_window((0,0), window=frame, anchor="nw")
        frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        imgr = build_photoimage_from_rgb_rows(rows_r)
        imgg = build_photoimage_from_rgb_rows(rows_g)
        imgb = build_photoimage_from_rgb_rows(rows_b)

        cr = tk.Canvas(frame, width=self._img_width, height=self._img_height, bg="white")
        cr.create_image(0, 0, anchor="nw", image=imgr)
        cr.image = imgr
        cr.pack(side="left", padx=6, pady=6)

        cg = tk.Canvas(frame, width=self._img_width, height=self._img_height, bg="white")
        cg.create_image(0, 0, anchor="nw", image=imgg)
        cg.image = imgg
        cg.pack(side="left", padx=6, pady=6)

        cb = tk.Canvas(frame, width=self._img_width, height=self._img_height, bg="white")
        cb.create_image(0, 0, anchor="nw", image=imgb)
        cb.image = imgb
        cb.pack(side="left", padx=6, pady=6)

        # labels below each image using simple placements (dependent on total width)
        lbl_frame = ttk.Frame(frame)
        lbl_frame.place(x=10, y=self._img_height + 10)
        ttk.Label(lbl_frame, text="Red Channel").grid(row=0, column=0, padx=(0, self._img_width+12))
        ttk.Label(lbl_frame, text="Green Channel").grid(row=0, column=1, padx=(0, self._img_width+12))
        ttk.Label(lbl_frame, text="Blue Channel").grid(row=0, column=2)

    def show_channel_histograms(self):
        """Compute and show histograms for R, G, B with controls to switch Combined/Single."""
        if not self.ensure_image_loaded():
            return

        rhist, ghist, bhist = compute_rgb_histograms(self._rgb_rows)

        top = tk.Toplevel(self)
        top.title("Channel Histograms (Combined or Single)")

        # Control frame with radio buttons
        ctrl_frame = ttk.Frame(top)
        ctrl_frame.pack(fill="x", padx=6, pady=6)

        mode_var = tk.StringVar(value="Combined")  # Combined, Red, Green, Blue

        def on_mode_change():
            mode = mode_var.get()
            if mode == "Combined":
                draw_combined_histogram(canvas_hist, rhist, ghist, bhist, width=640, height=200, draw_grid=True)
            elif mode == "Red":
                draw_histogram_on_canvas(canvas_hist, rhist, width=640, height=200, color="red")
            elif mode == "Green":
                draw_histogram_on_canvas(canvas_hist, ghist, width=640, height=200, color="green")
            elif mode == "Blue":
                draw_histogram_on_canvas(canvas_hist, bhist, width=640, height=200, color="blue")

        ttk.Label(ctrl_frame, text="View:").pack(side="left", padx=(0,8))
        ttk.Radiobutton(ctrl_frame, text="Combined (RGB)", variable=mode_var, value="Combined", command=on_mode_change).pack(side="left")
        ttk.Radiobutton(ctrl_frame, text="Red only", variable=mode_var, value="Red", command=on_mode_change).pack(side="left", padx=6)
        ttk.Radiobutton(ctrl_frame, text="Green only", variable=mode_var, value="Green", command=on_mode_change).pack(side="left", padx=6)
        ttk.Radiobutton(ctrl_frame, text="Blue only", variable=mode_var, value="Blue", command=on_mode_change).pack(side="left", padx=6)

        # Histogram canvas
        canvas_hist = tk.Canvas(top)
        canvas_hist.pack(padx=6, pady=(0,6))

        # Draw initial combined histogram
        draw_combined_histogram(canvas_hist, rhist, ghist, bhist, width=640, height=200, draw_grid=True)

        # Legend
        legend = ttk.Frame(top)
        legend.pack(padx=6, pady=(0,6), anchor="w")
        ttk.Label(legend, text="Legend:").pack(side="left")
        ttk.Label(legend, text=" ", background="red").pack(side="left", padx=4)
        ttk.Label(legend, text="Red").pack(side="left", padx=(0,8))
        ttk.Label(legend, text=" ", background="green").pack(side="left", padx=4)
        ttk.Label(legend, text="Green").pack(side="left", padx=(0,8))
        ttk.Label(legend, text=" ", background="blue").pack(side="left", padx=4)
        ttk.Label(legend, text="Blue").pack(side="left", padx=(0,8))

    def show_grayscale_and_hist(self):
        """Create grayscale image (average) and show image + histogram."""
        if not self.ensure_image_loaded():
            return

        gray_rows, ghist = compute_grayscale_rows_and_hist(self._rgb_rows)
        gimg = build_photoimage_from_rgb_rows(gray_rows)

        top = tk.Toplevel(self)
        top.title("Grayscale (avg) and Histogram")

        # show grayscale
        gcanvas = tk.Canvas(top, width=self._img_width, height=self._img_height, bg="white")
        gcanvas.pack(padx=6, pady=6)
        gcanvas.create_image(0, 0, anchor="nw", image=gimg)
        gcanvas.image = gimg

        # show histogram below
        hcanvas = tk.Canvas(top)
        hcanvas.pack(padx=6, pady=6)
        draw_histogram_on_canvas(hcanvas, ghist, width=512, height=150, color="gray")
        hcanvas.create_text(10, 10, anchor="nw", text="Grayscale histogram")


if __name__ == "__main__":
    app = PCXViewerApp()
    app.mainloop()