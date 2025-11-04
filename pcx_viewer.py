import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from pcxdecoder import decode_pcx
import viewer_style as style
import image_processing as ip
import filters as ft  
from typing import List, Tuple
from io import BytesIO

# ==== Utility functions ====
def hex_to_rgb(hexstr: str) -> Tuple[int, int, int]:
    if hexstr and hexstr.startswith("#") and len(hexstr) == 7:
        r = int(hexstr[1:3], 16)
        g = int(hexstr[3:5], 16)
        b = int(hexstr[5:7], 16)
        return r, g, b
    return 0, 0, 0

def rgb_rows_from_hex_rows(hex_rows: List[List[str]]) -> List[List[Tuple[int,int,int]]]:
    return [[hex_to_rgb(px) for px in row] for row in hex_rows]

def build_photoimage_from_rgb_rows(rgb_rows: List[List[Tuple[int,int,int]]]) -> tk.PhotoImage:
    height = len(rgb_rows)
    width = len(rgb_rows[0]) if height else 0
    img = tk.PhotoImage(width=width, height=height)
    for y, row in enumerate(rgb_rows):
        row_str = " ".join(f"#{r:02x}{g:02x}{b:02x}" for (r,g,b) in row)
        img.put("{" + row_str + "}", to=(0, y))
    return img

def split_channels_rgb_rows(rgb_rows: List[List[Tuple[int,int,int]]]):
    rows_r = [[(r,0,0) for (r,_,_) in row] for row in rgb_rows]
    rows_g = [[(0,g,0) for (_,g,_) in row] for row in rgb_rows]
    rows_b = [[(0,0,b) for (_,_,b) in row] for row in rgb_rows]
    return rows_r, rows_g, rows_b

def compute_rgb_histograms(rgb_rows: List[List[Tuple[int,int,int]]]):
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
    gray_rows = []
    ghist = [0]*256
    for row in rgb_rows:
        prow = []
        for (r,g,b) in row:
            s = int((r+g+b)/3)
            prow.append((s,s,s))
            ghist[s] += 1
        gray_rows.append(prow)
    return gray_rows, ghist

def plot_histogram_image(hist, color="gray", width=128, height=128):
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
    ax.bar(range(256), hist, color=color)
    ax.set_xlim(0,255)
    ax.set_ylim(0, max(hist)*1.1 if hist else 1)
    ax.axis('off')
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

# ==== PCX Viewer ====
class PCXViewer(tk.Frame):
    def __init__(self, master, file_path=None):
        super().__init__(master, bg=style.BG_MAIN)
        self.master = master
        self.pack(fill="both", expand=True)
        self.file_path = file_path

        # Toolbar
        toolbar = tk.Frame(self, bg=style.BG_TOOLBAR, padx=10, pady=8)
        toolbar.pack(side="top", fill="x")
        tk.Button(toolbar, text="Open PCX", command=self.open_pcx,
                  bg=style.BG_BUTTON, fg=style.FG_BUTTON,
                  font=("Segoe UI",10,"bold"), relief="flat", padx=10,pady=4).pack(side="left", padx=5)
        tk.Button(toolbar, text="Zoom In", command=self.zoom_in,
                  bg=style.BG_BUTTON, fg=style.FG_BUTTON,
                  font=("Segoe UI",10,"bold"), relief="flat", padx=10,pady=4).pack(side="left", padx=5)
        tk.Button(toolbar, text="Zoom Out", command=self.zoom_out,
                  bg=style.BG_BUTTON, fg=style.FG_BUTTON,
                  font=("Segoe UI",10,"bold"), relief="flat", padx=10,pady=4).pack(side="left", padx=5)

        # ==== Toolbar Container ====
        toolbar_section = tk.Frame(self, bg=style.BG_TOOLBAR)
        toolbar_section.pack(side="top", fill="x", padx=10, pady=5)

        # ==== Left Container: Point Processing ====
        point_container = tk.Frame(toolbar_section, bg=style.BG_TOOLBAR)
        point_container.pack(side="left", padx=10, pady=5)

        point_frame = tk.LabelFrame(point_container, text="Point Processing Methods",
                                    bg=style.BG_TOOLBAR, fg="black",
                                    font=("Segoe UI", 9, "bold"), padx=10, pady=5, labelanchor="n")
        point_frame.pack(side="top")

        tk.Button(point_frame, text="Grayscale", command=self.apply_grayscale,
                bg="#5c6f75", fg="white", font=("Segoe UI", 10, "bold"),
                relief="flat", padx=10, pady=4).pack(side="left", padx=4)
        tk.Button(point_frame, text="Negative", command=self.apply_negative,
                bg="#5c6f75", fg="white", font=("Segoe UI", 10, "bold"),
                relief="flat", padx=10, pady=4).pack(side="left", padx=4)
        tk.Button(point_frame, text="Threshold", command=self.apply_threshold,
                bg="#5c6f75", fg="white", font=("Segoe UI", 10, "bold"),
                relief="flat", padx=10, pady=4).pack(side="left", padx=4)
        tk.Button(point_frame, text="Gamma", command=self.apply_gamma,
                bg="#5c6f75", fg="white", font=("Segoe UI", 10, "bold"),
                relief="flat", padx=10, pady=4).pack(side="left", padx=4)
        tk.Button(point_frame, text="Hist Eq", command=self.apply_hist_eq,
                bg="#5c6f75", fg="white", font=("Segoe UI", 10, "bold"),
                relief="flat", padx=10, pady=4).pack(side="left", padx=4)
        tk.Button(point_frame, text="Reset", command=self.reset_image,
                bg="#37474F", fg="white", font=("Segoe UI", 10, "bold"),
                relief="flat", padx=10, pady=4).pack(side="left", padx=4)

        # ==== Right Container: Filtering ====
        filter_container = tk.Frame(toolbar_section, bg=style.BG_TOOLBAR)
        filter_container.pack(side="left", padx=10, pady=5)

        filter_frame = tk.LabelFrame(filter_container, text="Filtering Methods",
                                    bg=style.BG_TOOLBAR, fg="black",
                                    font=("Segoe UI", 9, "bold"), padx=10, pady=5, labelanchor="n")
        filter_frame.pack(side="top")

        tk.Button(filter_frame, text="Averaging Filter", command=self.apply_averaging,
                bg="#5c6f75", fg="white", font=("Segoe UI", 10, "bold"),
                relief="flat", padx=10, pady=4).pack(side="left", padx=4)

        tk.Button(filter_frame, text="Median Filter", command=self.apply_median,
                bg="#5c6f75", fg="white", font=("Segoe UI", 10, "bold"),
                relief="flat", padx=10, pady=4).pack(side="left", padx=4)

        tk.Button(filter_frame, text="Laplacian Highpass", command=self.apply_laplacian,
                bg="#5c6f75", fg="white", font=("Segoe UI", 10, "bold"),
                relief="flat", padx=10, pady=4).pack(side="left", padx=4)

        tk.Button(filter_frame, text="Unsharp Masking", command=self.apply_unsharp,
                bg="#5c6f75", fg="white", font=("Segoe UI", 10, "bold"),
                relief="flat", padx=10, pady=4).pack(side="left", padx=4)

        tk.Button(filter_frame, text="Highboost Filtering", command=self.apply_highboost,
                bg="#5c6f75", fg="white", font=("Segoe UI", 10, "bold"),
                relief="flat", padx=10, pady=4).pack(side="left", padx=4)

        tk.Button(filter_frame, text="Sobel Gradient", command=self.apply_sobel,
                bg="#5c6f75", fg="white", font=("Segoe UI", 10, "bold"),
                relief="flat", padx=10, pady=4).pack(side="left", padx=4)


        # Main Frame
        main_frame = tk.Frame(self, bg=style.BG_MAIN)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Canvas frame
        canvas_frame = tk.Frame(main_frame, bg=style.BG_MAIN)
        canvas_frame.pack(side="left", fill="both", expand=True, padx=(0,10))
        self.canvas = tk.Canvas(canvas_frame, bg=style.BG_PANEL, cursor="cross")
        self.canvas.pack(side="top", fill="both", expand=True)
        self.scroll_y = tk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        self.scroll_y.pack(side="right", fill="y")
        self.scroll_x = tk.Scrollbar(main_frame, orient="horizontal", command=self.canvas.xview)
        self.scroll_x.pack(side="bottom", fill="x")
        self.canvas.configure(yscrollcommand=self.scroll_y.set, xscrollcommand=self.scroll_x.set)

        self.canvas.bind("<Button-1>", self.get_pixel_info)
        self.canvas.bind("<ButtonPress-2>", self.start_pan)
        self.canvas.bind("<B2-Motion>", self.pan_image)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<Button-4>", self.on_mousewheel_linux)
        self.canvas.bind("<Button-5>", self.on_mousewheel_linux)

        # Info Panel
        info_frame = tk.Frame(main_frame, bg=style.BG_PANEL, bd=2, relief="groove", padx=15, pady=15)
        info_frame.pack(side="right", fill="y")
        tk.Label(info_frame, text="Pixel Info", font=style.FONT_HEADER,
                 bg=style.BG_PANEL, fg=style.FG_TEXT).pack(anchor="w", pady=(0,5))
        self.pixel_label = tk.Label(info_frame,
            text="Click on the image to view pixel RGB values.",
            font=style.FONT_TEXT, justify="left", bg=style.BG_PANEL, fg=style.FG_SUBTEXT)
        self.pixel_label.pack(anchor="w", pady=(0,10))
        self.color_preview = tk.Canvas(info_frame, width=80, height=50, bg="#cccccc", bd=1, relief="solid")
        self.color_preview.pack(anchor="w", pady=(0,20))
        tk.Frame(info_frame, height=2, bg="#e0e0e0").pack(fill="x", pady=10)
        tk.Label(info_frame, text="Header Info", font=style.FONT_HEADER,
                 bg=style.BG_PANEL, fg=style.FG_TEXT).pack(anchor="w", pady=(0,5))
        self.header_text = tk.Text(info_frame, height=12, width=36,
                                   font=style.FONT_MONO, bg="#f9f9f9", fg="#222",
                                   relief="flat", wrap="none")
        self.header_text.pack(anchor="w", pady=(0,5))
        self.header_text.configure(state="disabled")
        tk.Label(info_frame, text="Color Palette", font=style.FONT_HEADER,
                 bg=style.BG_PANEL, fg=style.FG_TEXT).pack(anchor="w", pady=(10,5))
        self.palette_canvas = tk.Canvas(info_frame, width=256, height=128, bg="#fff", bd=1, relief="solid")
        self.palette_canvas.pack(anchor="w")

        # Enhancement scrollable panel
        self.enhance_container = tk.Frame(canvas_frame)
        self.enhance_container.pack(side="bottom", fill="x", pady=(10,0))
        self.enhance_canvas = tk.Canvas(self.enhance_container, height=250)
        self.enhance_scroll = tk.Scrollbar(self.enhance_container, orient="vertical", command=self.enhance_canvas.yview)
        self.enhance_canvas.configure(yscrollcommand=self.enhance_scroll.set)
        self.enhance_scroll.pack(side="right", fill="y")
        self.enhance_canvas.pack(side="left", fill="both", expand=True)
        self.enhance_inner_frame = tk.Frame(self.enhance_canvas, bg=style.BG_MAIN)
        self.enhance_canvas.create_window((0,0), window=self.enhance_inner_frame, anchor="nw")
        self.enhance_inner_frame.bind("<Configure>", lambda e: self.enhance_canvas.configure(scrollregion=self.enhance_canvas.bbox("all")))

        # Vars
        self.image = None
        self.tk_img = None
        self.zoom_factor = 1.0
        self.filename = file_path
        self.header_info = None
        self.pan_start = None
        self.palette = None
        self.enhance_refs = []

        if file_path:
            self.load_pcx(file_path)

    # ==== File Handling ====
    def open_pcx(self):
        file_path = filedialog.askopenfilename(filetypes=[("PCX files","*.pcx")])
        if file_path:
            self.load_pcx(file_path)

    def load_pcx(self, file_path):
        try:
            pixels, info, palette = decode_pcx(Path(file_path))
            self.filename = file_path
            self.header_info = info
            self.palette = palette
            arr = np.array([[tuple(int(c[i:i+2],16) for i in (1,3,5)) for c in row] for row in pixels], dtype=np.uint8)
            self.image = Image.fromarray(arr)
            self.zoom_factor = 1.0
            self.display_image()
            self.show_header_info()
            self.draw_palette()
            self.show_enhancements()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open PCX file:\n{e}")

    # ==== Display & Zoom ====
    def display_image(self, img=None):
        if img is None: img = self.image
        if img:
            w = int(img.width*self.zoom_factor)
            h = int(img.height*self.zoom_factor)
            img_resized = img.resize((w,h))
            self.tk_img = ImageTk.PhotoImage(img_resized)
            self.canvas.delete("all")
            self.canvas.create_image(0,0, anchor="nw", image=self.tk_img)
            self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def zoom_in(self): self.zoom_factor*=1.25; self.display_image()
    def zoom_out(self): self.zoom_factor/=1.25; self.display_image()
    def on_mousewheel(self,event): self.zoom_in() if event.delta>0 else self.zoom_out()
    def on_mousewheel_linux(self,event):
        if event.num==4: self.zoom_in()
        elif event.num==5: self.zoom_out()
    def start_pan(self,event): self.pan_start=(event.x,event.y)
    def pan_image(self,event):
        if self.pan_start:
            dx=self.pan_start[0]-event.x
            dy=self.pan_start[1]-event.y
            self.canvas.xview_scroll(int(dx/2),"units")
            self.canvas.yview_scroll(int(dy/2),"units")
            self.pan_start=(event.x,event.y)

    # ==== Pixel info ====
    def get_pixel_info(self, event):
        if self.image:
            x = int(self.canvas.canvasx(event.x) / self.zoom_factor)
            y = int(self.canvas.canvasy(event.y) / self.zoom_factor)
            if 0 <= x < self.image.width and 0 <= y < self.image.height:
                rgb = self.image.getpixel((x, y))

                # Handle grayscale and RGB pixels safely
                if isinstance(rgb, int):
                    r = g = b = rgb
                elif isinstance(rgb, tuple):
                    if len(rgb) == 1:
                        r = g = b = rgb[0]
                    else:
                        r, g, b = rgb[:3]
                else:
                    r = g = b = 0

                # Update labels and color preview
                self.pixel_label.config(text=f"X:{x}\nY:{y}\nR:{r}\nG:{g}\nB:{b}")
                self.color_preview.config(bg=f"#{r:02x}{g:02x}{b:02x}")

    # ==== Header info ====
    def show_header_info(self):
        if not self.header_info: return
        info=self.header_info
        text="\n".join(f"{k}: {v}" for k,v in info.items())
        self.header_text.configure(state="normal")
        self.header_text.delete("1.0","end")
        self.header_text.insert("1.0",text)
        self.header_text.configure(state="disabled")

    # ==== Palette ====
    def draw_palette(self):
        self.palette_canvas.delete("all")
        if not self.palette: return
        PAD=2; cols=16; cell=16
        total=min(256,len(self.palette))
        rows=(total+cols-1)//cols
        self.palette_canvas.config(width=cols*cell+PAD*2, height=rows*cell+PAD*2)
        for i,(r,g,b) in enumerate(self.palette[:total]):
            col=i%cols; row=i//cols
            x=PAD+col*cell; y=PAD+row*cell
            self.palette_canvas.create_rectangle(x,y,x+cell,y+cell, fill=f"#{r:02x}{g:02x}{b:02x}", outline="")

    # ==== Enhancement display ====
    def show_enhancements(self):
        for w in self.enhance_inner_frame.winfo_children():
            w.destroy()
        self.enhance_refs.clear()

        arr = np.array(self.image)
        rgb_rows = arr.tolist() if arr.ndim == 3 else [[(v,v,v) for v in row] for row in arr.tolist()]
        rows_r, rows_g, rows_b = split_channels_rgb_rows(rgb_rows)
        rhist, ghist, bhist = compute_rgb_histograms(rgb_rows)
        gray_rows, gray_hist = compute_grayscale_rows_and_hist(rgb_rows)

        images = [("R", rows_r, rhist, "red"),
                  ("G", rows_g, ghist, "green"),
                  ("B", rows_b, bhist, "blue"),
                  ("Gray", gray_rows, gray_hist, "gray")]

        for name, img_rows, hist, color in images:
            frame = tk.Frame(self.enhance_inner_frame, bg=style.BG_MAIN)
            frame.pack(side="left", padx=5, pady=5)

            img = build_photoimage_from_rgb_rows(img_rows)
            tk.Label(frame, text=name, bg=style.BG_MAIN).pack()
            lbl_img = tk.Label(frame, image=img, bg=style.BG_MAIN)
            lbl_img.pack()
            self.enhance_refs.append(img)

            img_height = img.height()
            hist_img_pil = plot_histogram_image(hist, color=color, width=img.width(), height=img_height)
            tk.Label(frame, text=f"{name} Histogram", bg=style.BG_MAIN).pack()
            hist_img = ImageTk.PhotoImage(hist_img_pil)
            tk.Label(frame, image=hist_img, bg=style.BG_MAIN).pack()
            self.enhance_refs.append(hist_img)

    # ==== Point Processing Functions ====
    def apply_grayscale(self):
        if self.image:
            arr = list(self.image.convert("RGB").getdata())
            w, h = self.image.size
            rgb_rows = [arr[i*w:(i+1)*w] for i in range(h)]
            result_rows = ip.to_grayscale(rgb_rows)
            arr_out = [px for row in result_rows for px in row]
            self.image = Image.new("RGB", (w, h))
            self.image.putdata(arr_out)
            self.display_image()

    def apply_negative(self):
        if self.image:
            arr = list(self.image.convert("RGB").getdata())
            w, h = self.image.size
            rgb_rows = [arr[i*w:(i+1)*w] for i in range(h)]
            result_rows = ip.to_negative(rgb_rows)
            arr_out = [px for row in result_rows for px in row]
            self.image = Image.new("RGB", (w, h))
            self.image.putdata(arr_out)
            self.display_image()

    def apply_threshold(self):
        if self.image:
            try:
                value = simpledialog.askinteger("Threshold", "Enter threshold value (0–255):",
                                                minvalue=0, maxvalue=255)
                if value is not None:
                    arr = list(self.image.convert("RGB").getdata())
                    w, h = self.image.size
                    rgb_rows = [arr[i*w:(i+1)*w] for i in range(h)]
                    result_rows = ip.manual_threshold(rgb_rows, threshold=value)
                    arr_out = [px for row in result_rows for px in row]
                    self.image = Image.new("RGB", (w, h))
                    self.image.putdata(arr_out)
                    self.display_image()
            except Exception as e:
                messagebox.showerror("Error", f"Thresholding failed:\n{e}")

    def apply_gamma(self):
        if self.image:
            try:
                value = simpledialog.askfloat("Gamma Correction", "Enter gamma value (>0):", minvalue=0.1)
                if value is not None:
                    arr = list(self.image.convert("RGB").getdata())
                    w, h = self.image.size
                    rgb_rows = [arr[i*w:(i+1)*w] for i in range(h)]
                    result_rows = ip.gamma_transform(rgb_rows, gamma=value)
                    arr_out = [px for row in result_rows for px in row]
                    self.image = Image.new("RGB", (w, h))
                    self.image.putdata(arr_out)
                    self.display_image()
            except Exception as e:
                messagebox.showerror("Error", f"Gamma transformation failed:\n{e}")

    def apply_hist_eq(self):
        if self.image:
            try:
                import matplotlib.pyplot as plt

                # --- Step 1: Convert current image to RGB rows ---
                arr = list(self.image.convert("RGB").getdata())
                w, h = self.image.size
                rgb_rows = [arr[i * w:(i + 1) * w] for i in range(h)]

                # --- Step 2: Apply histogram equalization using image_processing.py ---
                result_rows = ip.histogram_equalization(rgb_rows)

                # --- Step 3: Flatten and rebuild the equalized image ---
                arr_out = [px for row in result_rows for px in row]
                self.image = Image.new("RGB", (w, h))
                self.image.putdata(arr_out)
                self.display_image()

                # --- Step 4: Compute histogram of the equalized image ---
                hist = [0] * 256
                for row in result_rows:
                    for (r, g, b) in row:
                        gray = (r + g + b) // 3
                        hist[gray] += 1

                # --- Step 5: Plot histogram in a separate window ---
                plt.figure("Histogram after Equalization")
                plt.title("Histogram after Equalization")
                plt.xlabel("Intensity Value (0–255)")
                plt.ylabel("Frequency")
                plt.bar(range(256), hist, width=1.0, color='gray')
                plt.tight_layout()
                plt.show()

            except Exception as e:
                messagebox.showerror("Error", f"Histogram equalization failed:\n{e}")

    def reset_image(self):
        """Restore the original loaded image"""
        if self.filename:
            try:
                pixels, info, palette = decode_pcx(Path(self.filename))
                arr = np.array([[tuple(int(c[i:i+2],16) for i in (1,3,5)) for c in row] for row in pixels], dtype=np.uint8)
                self.image = Image.fromarray(arr)
                self.display_image()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to reset image:\n{e}")

    # ---------------------------------------------------------------------
    # Filtering Methods Integration
    # ---------------------------------------------------------------------
    
    def apply_averaging(self):
        if self.image:
            arr = list(self.image.convert("RGB").getdata())
            w, h = self.image.size
            rgb_rows = [arr[i*w:(i+1)*w] for i in range(h)]
            result_rows, desc = ft.apply_averaging(rgb_rows)
            arr_out = [px for row in result_rows for px in row]
            self.image = Image.new("RGB", (w, h))
            self.image.putdata(arr_out)
            self.display_image()
            messagebox.showinfo("Filter Applied", desc)

    def apply_median(self):
        if self.image:
            arr = list(self.image.convert("RGB").getdata())
            w, h = self.image.size
            rgb_rows = [arr[i*w:(i+1)*w] for i in range(h)]
            result_rows, desc = ft.apply_median(rgb_rows)
            arr_out = [px for row in result_rows for px in row]
            self.image = Image.new("RGB", (w, h))
            self.image.putdata(arr_out)
            self.display_image()
            messagebox.showinfo("Filter Applied", desc)

    def apply_laplacian(self):
        if self.image:
            arr = list(self.image.convert("RGB").getdata())
            w, h = self.image.size
            rgb_rows = [arr[i*w:(i+1)*w] for i in range(h)]
            result_rows, desc = ft.apply_laplacian_highpass(rgb_rows)
            arr_out = [px for row in result_rows for px in row]
            self.image = Image.new("RGB", (w, h))
            self.image.putdata(arr_out)
            self.display_image()
            messagebox.showinfo("Filter Applied", desc)

    def apply_unsharp(self):
        if self.image:
            value = simpledialog.askfloat("Unsharp Masking", "Enter amount (e.g., 1.0–2.0):", minvalue=0.1)
            if value is not None:
                arr = list(self.image.convert("RGB").getdata())
                w, h = self.image.size
                rgb_rows = [arr[i*w:(i+1)*w] for i in range(h)]
                result_rows, desc = ft.apply_unsharp(rgb_rows, amount=value)
                arr_out = [px for row in result_rows for px in row]
                self.image = Image.new("RGB", (w, h))
                self.image.putdata(arr_out)
                self.display_image()
                messagebox.showinfo("Filter Applied", desc)

    def apply_highboost(self):
        if self.image:
            value = simpledialog.askfloat("Highboost Filter", "Enter amplification factor A (>1):", minvalue=1.0)
            if value is not None:
                arr = list(self.image.convert("RGB").getdata())
                w, h = self.image.size
                rgb_rows = [arr[i*w:(i+1)*w] for i in range(h)]
                result_rows, desc = ft.apply_highboost(rgb_rows, A=value)
                arr_out = [px for row in result_rows for px in row]
                self.image = Image.new("RGB", (w, h))
                self.image.putdata(arr_out)
                self.display_image()
                messagebox.showinfo("Filter Applied", desc)

    def apply_sobel(self):
        if self.image:
            arr = list(self.image.convert("RGB").getdata())
            w, h = self.image.size
            rgb_rows = [arr[i*w:(i+1)*w] for i in range(h)]
            result_rows, desc = ft.apply_sobel_gradient(rgb_rows)
            arr_out = [px for row in result_rows for px in row]
            self.image = Image.new("RGB", (w, h))
            self.image.putdata(arr_out)
            self.display_image()
            messagebox.showinfo("Filter Applied", desc)



