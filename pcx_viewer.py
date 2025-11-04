import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from pcxdecoder import decode_pcx
import viewer_style as style  # style settings


class PCXViewer(tk.Frame):
    def __init__(self, master, file_path=None):
        super().__init__(master, bg=style.BG_MAIN)
        self.master = master
        self.pack(fill="both", expand=True)
        self.file_path = file_path

        # === Toolbar ===
        toolbar = tk.Frame(self, bg=style.BG_TOOLBAR, padx=10, pady=8)
        toolbar.pack(side="top", fill="x")

        tk.Button(toolbar, text="Open PCX", command=self.open_pcx,
                  bg=style.BG_BUTTON, fg=style.FG_BUTTON,
                  font=("Segoe UI", 10, "bold"), relief="flat", padx=10, pady=4).pack(side="left", padx=5)

        tk.Button(toolbar, text="Zoom In", command=self.zoom_in,
                  bg=style.BG_BUTTON, fg=style.FG_BUTTON,
                  font=("Segoe UI", 10, "bold"), relief="flat", padx=10, pady=4).pack(side="left", padx=5)

        tk.Button(toolbar, text="Zoom Out", command=self.zoom_out,
                  bg=style.BG_BUTTON, fg=style.FG_BUTTON,
                  font=("Segoe UI", 10, "bold"), relief="flat", padx=10, pady=4).pack(side="left", padx=5)

        tk.Button(toolbar, text="Enhance Image", command=self.image_enhancement,
                  bg=style.BG_BUTTON, fg=style.FG_BUTTON,
                  font=("Segoe UI", 10, "bold"), relief="flat", padx=10, pady=4).pack(side="left", padx=5)

        # === Main Frame ===
        main_frame = tk.Frame(self, bg=style.BG_MAIN)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Canvas frame
        canvas_frame = tk.Frame(main_frame, bg=style.BG_MAIN)
        canvas_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

        self.canvas = tk.Canvas(canvas_frame, bg=style.BG_PANEL, cursor="cross")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.scroll_y = tk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        self.scroll_y.pack(side="right", fill="y")
        self.scroll_x = tk.Scrollbar(main_frame, orient="horizontal", command=self.canvas.xview)
        self.scroll_x.pack(side="bottom", fill="x")

        self.canvas.configure(yscrollcommand=self.scroll_y.set, xscrollcommand=self.scroll_x.set)

        # Bindings
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
                 bg=style.BG_PANEL, fg=style.FG_TEXT).pack(anchor="w", pady=(0, 5))
        self.pixel_label = tk.Label(info_frame,
            text="Click on the image to view pixel RGB values.",
            font=style.FONT_TEXT, justify="left", bg=style.BG_PANEL, fg=style.FG_SUBTEXT)
        self.pixel_label.pack(anchor="w", pady=(0, 10))

        self.color_preview = tk.Canvas(info_frame, width=80, height=50,
                                       bg="#cccccc", bd=1, relief="solid")
        self.color_preview.pack(anchor="w", pady=(0, 20))

        tk.Frame(info_frame, height=2, bg="#e0e0e0").pack(fill="x", pady=10)

        tk.Label(info_frame, text="Header Info", font=style.FONT_HEADER,
                 bg=style.BG_PANEL, fg=style.FG_TEXT).pack(anchor="w", pady=(0, 5))
        self.header_text = tk.Text(info_frame, height=12, width=36,
                                   font=style.FONT_MONO, bg="#f9f9f9", fg="#222",
                                   relief="flat", wrap="none")
        self.header_text.pack(anchor="w", pady=(0, 5))
        self.header_text.configure(state="disabled")

        # Color Palette Canvas
        tk.Label(info_frame, text="Color Palette", font=style.FONT_HEADER,
                 bg=style.BG_PANEL, fg=style.FG_TEXT).pack(anchor="w", pady=(10, 5))
        self.palette_canvas = tk.Canvas(info_frame, width=256, height=128, bg="#fff", bd=1, relief="solid")
        self.palette_canvas.pack(anchor="w")

        # Vars
        self.image = None
        self.tk_img = None
        self.zoom_factor = 1.0
        self.filename = file_path
        self.header_info = None
        self.pan_start = None
        self.palette = None

        if file_path:
            self.load_pcx(file_path)

    # === File Handling ===
    def open_pcx(self):
        file_path = filedialog.askopenfilename(filetypes=[("PCX files", "*.pcx")])
        if file_path:
            self.load_pcx(file_path)

    def load_pcx(self, file_path):
        try:
            pixels, info, palette = decode_pcx(Path(file_path))
            self.filename = file_path
            self.header_info = info
            self.palette = palette
            arr = np.array([[tuple(int(c[i:i+2], 16) for i in (1, 3, 5)) for c in row] for row in pixels], dtype=np.uint8)
            self.image = Image.fromarray(arr)
            self.zoom_factor = 1.0
            self.display_image()
            self.show_header_info()
            self.draw_palette()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open PCX file:\n{e}")

    # === Display & Zoom ===
    def display_image(self):
        if self.image:
            w = int(self.image.width * self.zoom_factor)
            h = int(self.image.height * self.zoom_factor)
            img_resized = self.image.resize((w, h))
            self.tk_img = ImageTk.PhotoImage(img_resized)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
            self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def zoom_in(self): self.zoom_factor *= 1.25; self.display_image()
    def zoom_out(self): self.zoom_factor /= 1.25; self.display_image()

    # Scroll wheel
    def on_mousewheel(self, event): self.zoom_in() if event.delta > 0 else self.zoom_out()
    def on_mousewheel_linux(self, event):
        if event.num == 4: self.zoom_in()
        elif event.num == 5: self.zoom_out()

    # Pan
    def start_pan(self, event): self.pan_start = (event.x, event.y)
    def pan_image(self, event):
        if self.pan_start:
            dx = self.pan_start[0] - event.x
            dy = self.pan_start[1] - event.y
            self.canvas.xview_scroll(int(dx/2), "units")
            self.canvas.yview_scroll(int(dy/2), "units")
            self.pan_start = (event.x, event.y)

    # Pixel info
    def get_pixel_info(self, event):
        if self.image:
            x = int(self.canvas.canvasx(event.x)/self.zoom_factor)
            y = int(self.canvas.canvasy(event.y)/self.zoom_factor)
            if 0 <= x < self.image.width and 0 <= y < self.image.height:
                rgb = self.image.getpixel((x, y))
                r, g, b = rgb[:3]
                self.pixel_label.config(text=f"X: {x}\nY: {y}\nR: {r}\nG: {g}\nB: {b}")
                self.color_preview.config(bg=f"#{r:02x}{g:02x}{b:02x}")

    # Header info
    def show_header_info(self):
        if not self.header_info: return
        info = self.header_info
        text = "\n".join(f"{k}: {v}" for k, v in info.items())
        self.header_text.configure(state="normal")
        self.header_text.delete("1.0", "end")
        self.header_text.insert("1.0", text)
        self.header_text.configure(state="disabled")

    # Draw palette
    def draw_palette(self):
        self.palette_canvas.delete("all")
        if not self.palette: return
        PAD = 2
        cols = 16
        cell = 16
        total = min(256, len(self.palette))
        rows = (total + cols - 1)//cols
        self.palette_canvas.config(width=cols*cell+PAD*2, height=rows*cell+PAD*2)
        for i, (r,g,b) in enumerate(self.palette[:total]):
            col = i % cols
            row = i // cols
            x = PAD + col*cell
            y = PAD + row*cell
            self.palette_canvas.create_rectangle(x, y, x+cell, y+cell, fill=f"#{r:02x}{g:02x}{b:02x}", outline="")

    # === Image Enhancement ===
    def image_enhancement(self):
        if not self.image:
            messagebox.showwarning("Warning", "Open an image first")
            return
        img_arr = np.array(self.image)

        # Split channels
        R, G, B = img_arr[:,:,0], img_arr[:,:,1], img_arr[:,:,2]

        # Grayscale
        gray_arr = np.mean(img_arr, axis=2).astype(np.uint8)

        # Display histograms
        fig, axes = plt.subplots(2,3, figsize=(12,6))
        axes = axes.flatten()
        axes[0].imshow(R, cmap="Reds"); axes[0].set_title("Red Channel")
        axes[1].imshow(G, cmap="Greens"); axes[1].set_title("Green Channel")
        axes[2].imshow(B, cmap="Blues"); axes[2].set_title("Blue Channel")
        axes[3].hist(R.flatten(), bins=256, color="red"); axes[3].set_title("Red Histogram")
        axes[4].hist(G.flatten(), bins=256, color="green"); axes[4].set_title("Green Histogram")
        axes[5].hist(B.flatten(), bins=256, color="blue"); axes[5].set_title("Blue Histogram")
        plt.tight_layout()
        plt.show()

        # Grayscale display & histogram
        gray_img = Image.fromarray(gray_arr)
        self.display_image(gray_img)
        plt.figure(figsize=(6,4))
        plt.hist(gray_arr.flatten(), bins=256, color="gray")
        plt.title("Grayscale Histogram")
        plt.show()
        self.image = gray_img  # update current image


if __name__ == "__main__":
    root = tk.Tk()
    root.title("PCX Viewer with Enhancement")
    root.geometry("1200x800")
    app = PCXViewer(root)
    app.pack(fill="both", expand=True)
    root.mainloop()
