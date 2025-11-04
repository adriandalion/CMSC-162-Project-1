import tkinter as tk
from tkinter import filedialog
from viewer import ImageViewer
from pcx_viewer import PCXViewer

class ImageApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CMSC162 Final Project 1")
        self.geometry("1000x700")
        self.viewer_frame = None

        # Menu
        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Image", command=self.open_image)
        file_menu.add_command(label="Open PCX", command=self.open_pcx)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        self.config(menu=menubar)

    def show_viewer(self, viewer_class, file_path):
        if self.viewer_frame:
            self.viewer_frame.destroy()
        self.viewer_frame = viewer_class(self, file_path)
        self.viewer_frame.pack(fill="both", expand=True)

    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if file_path:
            self.show_viewer(ImageViewer, file_path)

    def open_pcx(self):
        file_path = filedialog.askopenfilename(filetypes=[("PCX Files", "*.pcx")])
        if file_path:
            self.show_viewer(PCXViewer, file_path)

if __name__ == "__main__":
    app = ImageApp()
    app.mainloop()
