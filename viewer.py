import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import viewer_style as style

class ImageViewer(tk.Frame):
    def __init__(self, master, file_path=None):
        super().__init__(master, bg=style.BG_MAIN)
        self.master = master
        self.file_path = file_path
        self.pack(fill="both", expand=True)

        # === Top Toolbar ===
        toolbar = tk.Frame(self, bg=style.BG_TOOLBAR, padx=10, pady=8)
        toolbar.pack(side="top", fill="x")

        tk.Button(toolbar, text="Open Image", command=self.open_image,
                  bg=style.BG_BUTTON, fg=style.FG_BUTTON,
                  font=("Segoe UI", 10, "bold"), relief="flat",
                  padx=10, pady=4).pack(side="left", padx=5)

        tk.Button(toolbar, text="Zoom In", command=self.zoom_in,
                  bg=style.BG_BUTTON, fg=style.FG_BUTTON,
                  font=("Segoe UI", 10, "bold"), relief="flat",
                  padx=10, pady=4).pack(side="left", padx=5)

        tk.Button(toolbar, text="Zoom Out", command=self.zoom_out,
                  bg=style.BG_BUTTON, fg=style.FG_BUTTON,
                  font=("Segoe UI", 10, "bold"), relief="flat",
                  padx=10, pady=4).pack(side="left", padx=5)

        # === Main Content Layout ===
        main_frame = tk.Frame(self, bg=style.BG_MAIN)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # === Canvas with Scrollbars ===
        canvas_frame = tk.Frame(main_frame, bg=style.BG_MAIN)
        canvas_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

        self.canvas = tk.Canvas(canvas_frame, bg=style.BG_PANEL, cursor="cross")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.scroll_y = tk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        self.scroll_y.pack(side="right", fill="y")
        self.scroll_x = tk.Scrollbar(main_frame, orient="horizontal", command=self.canvas.xview)
        self.scroll_x.pack(side="bottom", fill="x")

        self.canvas.configure(yscrollcommand=self.scroll_y.set, xscrollcommand=self.scroll_x.set)

        # === Bindings ===
        self.canvas.bind("<Button-1>", self.get_pixel_info)
        self.canvas.bind("<ButtonPress-2>", self.start_pan)  # Middle button drag
        self.canvas.bind("<B2-Motion>", self.pan_image)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)  # Windows/macOS
        self.canvas.bind("<Button-4>", self.on_mousewheel_linux)  # Linux scroll up
        self.canvas.bind("<Button-5>", self.on_mousewheel_linux)  # Linux scroll down

        # === Info Panel ===
        info_frame = tk.Frame(main_frame, bg=style.BG_PANEL, bd=2,
                              relief="groove", padx=15, pady=15)
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
                                   font=style.FONT_MONO, bg="#f9f9f9",
                                   fg="#222", relief="flat", wrap="none")
        self.header_text.pack(anchor="w", pady=(0, 5))
        self.header_text.configure(state="disabled")

        # === Initialize Variables ===
        self.image = None
        self.tk_img = None
        self.zoom_factor = 1.0
        self.filename = file_path
        self.pan_start = None
        if file_path:
            self.load_image(file_path)

    # === File Handling ===
    def open_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif;*.tiff")]
        )
        if file_path:
            self.load_image(file_path)

    def load_image(self, file_path):
        try:
            self.image = Image.open(file_path)
            self.filename = file_path
            self.zoom_factor = 1.0
            self.display_image()
            self.show_header_info()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open image: {e}")

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

    def zoom_in(self):
        self.zoom_factor *= 1.25
        self.display_image()

    def zoom_out(self):
        self.zoom_factor /= 1.25
        self.display_image()

    # === Scroll Wheel Zoom Handlers ===
    def on_mousewheel(self, event):
        if event.delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()

    def on_mousewheel_linux(self, event):
        if event.num == 4:  # scroll up
            self.zoom_in()
        elif event.num == 5:  # scroll down
            self.zoom_out()

    # === Pan Handlers ===
    def start_pan(self, event):
        self.pan_start = (event.x, event.y)

    def pan_image(self, event):
        if self.pan_start:
            dx = self.pan_start[0] - event.x
            dy = self.pan_start[1] - event.y
            self.canvas.xview_scroll(int(dx / 2), "units")
            self.canvas.yview_scroll(int(dy / 2), "units")
            self.pan_start = (event.x, event.y)

    # === Pixel Info ===
    def get_pixel_info(self, event):
        if self.image:
            x = int(self.canvas.canvasx(event.x) / self.zoom_factor)
            y = int(self.canvas.canvasy(event.y) / self.zoom_factor)
            if 0 <= x < self.image.width and 0 <= y < self.image.height:
                rgb = self.image.getpixel((x, y))
                if isinstance(rgb, int):
                    rgb = (rgb, rgb, rgb)
                elif len(rgb) == 4:
                    rgb = rgb[:3]
                r, g, b = rgb
                self.pixel_label.config(text=f"X: {x}\nY: {y}\nR: {r}\nG: {g}\nB: {b}")
                self.color_preview.config(bg=f"#{r:02x}{g:02x}{b:02x}")

    # === Header Info ===
    def show_header_info(self):
        if not self.image or not self.filename:
            return

        file_stats = os.stat(self.filename)
        header_info = (
            f"Filename: {os.path.basename(self.filename)}\n"
            f"File Size: {file_stats.st_size} bytes\n"
            f"Format: {self.image.format}\n"
            f"Mode: {self.image.mode}\n"
            f"Dimensions: {self.image.width} × {self.image.height}\n"
            f"DPI: {self.image.info.get('dpi', 'N/A')}\n"
            f"Compression: {self.image.info.get('compression', 'N/A')}"
        )

        self.header_text.configure(state="normal")
        self.header_text.delete("1.0", "end")
        self.header_text.insert("1.0", header_info)
        self.header_text.configure(state="disabled")
