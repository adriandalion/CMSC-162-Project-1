#!/usr/bin/env python3
"""
CMSC 162 – Project 1 Main Launcher (Guides 1–5)

This GUI keeps each project guide in its own file for modularity.
It simply provides buttons to launch each guide script in a separate
process using the same Python interpreter.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import subprocess
import sys


# Names of your guide files (must be in the same folder as this file)
GUIDE_SCRIPTS = {
    "Guide 1 – Basic PCX Decoder": "CMSC162_Guide1_Dalion_Mazo.py",
    "Guide 2 – PCX Viewer (Header & Palette)": "CMSC162_Guide2_Dalion_Mazo.py",
    "Guide 3 – Channels & Histograms": "CMSC162_Guide3_Dalion_Mazo.py",
    "Guide 4 – Point Processing (Negative / Gamma / Equalization)": "CMSC162_Guide4_Dalion_Mazo.py",
    "Guide 5 – Spatial Filters (Smoothing / Sharpening / Gradient)": "CMSC162_Guide5_Dalion_Mazo.py",
}


def run_guide(script_name: str):
    """
    Launch the given guide script as a separate Python process.
    This avoids multiple Tk root windows in the same process and
    keeps each guide file unchanged.
    """
    script_path = Path(__file__).resolve().parent / script_name

    if not script_path.exists():
        messagebox.showerror(
            "File not found",
            f"Cannot find:\n{script_path}\n\n"
            "Make sure this main launcher file is in the same folder\n"
            "as all CMSC162_GuideX_Dalion_Mazo.py files.",
        )
        return

    try:
        # Use the same Python interpreter that is running this script
        subprocess.Popen([sys.executable, str(script_path)])
    except Exception as e:
        messagebox.showerror("Error launching guide", f"Failed to run {script_name}:\n{e}")


def main():
    root = tk.Tk()
    root.title("CMSC 162 – Project 1 Main GUI (Guides 1–5)")
    root.geometry("520x360")
    root.configure(bg="white")

    # Main frame
    main_frame = ttk.Frame(root, padding=12)
    main_frame.pack(fill="both", expand=True)

    title_label = ttk.Label(
        main_frame,
        text="CMSC 162 – Project 1 Launcher",
        font=("Segoe UI", 14, "bold"),
    )
    title_label.pack(pady=(0, 8))

    desc = (
        "This main GUI keeps each project guide in a separate file.\n"
        "Click a button below to open the corresponding GUI for that guide."
    )
    ttk.Label(main_frame, text=desc, justify="center").pack(pady=(0, 10))

    # Separator
    ttk.Separator(main_frame, orient="horizontal").pack(fill="x", pady=6)

    # Buttons for each guide
    btn_frame = ttk.Frame(main_frame)
    btn_frame.pack(fill="x", pady=4)

    for label, filename in GUIDE_SCRIPTS.items():
        def make_cmd(f=filename):  # bind current filename
            return lambda: run_guide(f)

        b = ttk.Button(btn_frame, text=label, command=make_cmd())
        b.pack(fill="x", pady=3)

    # Small note at the bottom
    ttk.Separator(main_frame, orient="horizontal").pack(fill="x", pady=8)
    note = (
        "Note:\n"
        "Each guide opens in its own window using its original GUI.\n"
        "You can work on one guide at a time while keeping the code modular."
    )
    ttk.Label(main_frame, text=note, justify="left", foreground="#555").pack(anchor="w", pady=(0, 4))

    root.mainloop()


if __name__ == "__main__":
    main()
