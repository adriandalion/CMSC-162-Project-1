import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk


class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 1 - Image Viewer")
        self.root.geometry("800x700")

        # --- Menu Bar ---
        self.menu_bar = tk.Menu(self.root)
        self.root.config(menu=self.menu_bar)

        file_menu = tk.Menu(self.menu_bar, tearoff=0)
        file_menu.add_command(label="Open Image", command=self.open_image)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        self.menu_bar.add_cascade(label="File", menu=file_menu)

        # --- Main Layout (vertical stacking) ---
        self.frame_top = tk.Frame(self.root, bg="lightgray")
        self.frame_top.pack(side="top", fill="both", expand=True)

        self.frame_bottom = tk.Frame(self.root, height=200, bg="white")
        self.frame_bottom.pack(side="bottom", fill="x")

        # --- Image Canvas ---
        self.canvas = tk.Canvas(self.frame_top, bg="black")
        self.canvas.pack(fill="both", expand=True)

        self.image = None
        self.tk_image = None

        # --- Pixel Info Display (below image) ---
        self.info_label = tk.Label(self.frame_bottom, text="Pixel Info", font=("Arial", 12, "bold"))
        self.info_label.pack(pady=5)

        self.pixel_text = tk.Text(self.frame_bottom, width=80, height=10, wrap="word")
        self.pixel_text.pack(padx=10, pady=5, fill="x")

        # Bind mouse click to get RGB values
        self.canvas.bind("<Button-1>", self.get_pixel_value)

    def open_image(self):
        """Open an image file and display it on the canvas."""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.tiff *.bmp")]
        )
        if not file_path:
            return

        self.image = Image.open(file_path).convert("RGB")
        self.tk_image = ImageTk.PhotoImage(self.image)

        # Clear previous canvas content
        self.canvas.delete("all")

        # Resize canvas to image size and draw
        self.canvas.config(width=self.tk_image.width(), height=self.tk_image.height())
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

    def get_pixel_value(self, event):
        """Get RGB values of the clicked pixel and mark it with a crosshair and color swatch."""
        if self.image is None:
            return

        x, y = event.x, event.y
        if 0 <= x < self.image.width and 0 <= y < self.image.height:
            r, g, b = self.image.getpixel((x, y))

            # Create a frame for one entry (text + color square)
            entry_frame = tk.Frame(self.pixel_text, pady=2)
            entry_frame.pack(anchor="w")

            # Show text info
            info = f"X: {x}, Y: {y} → R: {r}, G: {g}, B: {b}"
            label = tk.Label(entry_frame, text=info, anchor="w")
            label.pack(side="left")

            # Add a small color swatch
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            color_box = tk.Canvas(entry_frame, width=20, height=20, bg=hex_color, highlightthickness=1, highlightbackground="black")
            color_box.pack(side="right", padx=5)

            # Scroll down automatically
            self.pixel_text.window_create("end", window=entry_frame)
            self.pixel_text.insert("end", "\n")
            self.pixel_text.see("end")

            self.draw_marker(x, y)

    def draw_marker(self, x, y):
        """Draw a small red crosshair at (x, y); only the latest marker remains."""
        # Remove any previous marker
        self.canvas.delete("marker")

        size = 5
        self.canvas.create_line(x - size, y, x + size, y, fill="red", width=2, tags=("marker",))
        self.canvas.create_line(x, y - size, x, y + size, fill="red", width=2, tags=("marker",))


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()