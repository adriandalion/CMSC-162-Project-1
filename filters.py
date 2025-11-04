#!/usr/bin/env python3
"""
filters.py

Spatial domain image enhancement filters for PCX viewer:
- Averaging filter
- Median filter
- Laplacian high-pass
- Unsharp masking
- Highboost filtering
- Gradient magnitude via Sobel operator
"""

from typing import List, Tuple

# ------------------ Utility functions ------------------

def to_gray_rows(rgb_rows: List[List[Tuple[int,int,int]]]) -> List[List[int]]:
    return [[(r+g+b)//3 for (r,g,b) in row] for row in rgb_rows]

def gray_to_rgb_rows(gray_rows: List[List[int]]) -> List[List[Tuple[int,int,int]]]:
    return [[(v,v,v) for v in row] for row in gray_rows]

def clip8(x: int) -> int:
    return max(0, min(255, x))

def pad_replicate(gray: List[List[int]], r: int) -> List[List[int]]:
    h, w = len(gray), len(gray[0])
    out = [[0]*(w + 2*r) for _ in range(h + 2*r)]
    for y in range(h + 2*r):
        sy = 0 if y < r else (h-1 if y >= h+r else y - r)
        for x in range(w + 2*r):
            sx = 0 if x < r else (w-1 if x >= w+r else x - r)
            out[y][x] = gray[sy][sx]
    return out

# ------------------ Convolution / Median ------------------

def conv3x3_gray(gray: List[List[int]], k: List[List[int]], scale: float = 1.0, bias: float = 0.0) -> List[List[int]]:
    h, w = len(gray), len(gray[0])
    p = pad_replicate(gray, 1)
    out = [[0]*w for _ in range(h)]
    for y in range(h):
        for x in range(w):
            s = sum(p[y+dy][x+dx]*k[dy][dx] for dy in range(3) for dx in range(3))
            out[y][x] = clip8(int(round(s*scale + bias)))
    return out

def conv3x3_gray_signed(gray: List[List[int]], k: List[List[int]]) -> List[List[float]]:
    h, w = len(gray), len(gray[0])
    p = pad_replicate(gray, 1)
    out = [[0.0]*w for _ in range(h)]
    for y in range(h):
        for x in range(w):
            s = sum(p[y+dy][x+dx]*k[dy][dx] for dy in range(3) for dx in range(3))
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
            win = [p[y+dy][x+dx] for dy in range(3) for dx in range(3)]
            win.sort()
            out[y][x] = win[4]
    return out

# ------------------ Filters ------------------

def apply_averaging(rgb_rows: List[List[Tuple[int,int,int]]]):
    gray = to_gray_rows(rgb_rows)
    g2 = box_blur3_gray(gray)
    return gray_to_rgb_rows(g2), "Averaging filter: 3x3 box (1/9)"

def apply_median(rgb_rows: List[List[Tuple[int,int,int]]]):
    gray = to_gray_rows(rgb_rows)
    g2 = median3_gray(gray)
    return gray_to_rgb_rows(g2), "Median filter: 3x3 window"

def apply_laplacian_highpass(rgb_rows: List[List[Tuple[int,int,int]]]):
    gray = to_gray_rows(rgb_rows)
    k = [[0,-1,0],[-1,4,-1],[0,-1,0]]
    resp = conv3x3_gray_signed(gray, k)
    h, w = len(gray), len(gray[0])
    vmin = min(min(row) for row in resp)
    vmax = max(max(row) for row in resp)
    rng = vmax - vmin if vmax>vmin else 1.0
    norm = [[int(round((resp[y][x]-vmin)/rng*255)) for x in range(w)] for y in range(h)]
    return gray_to_rgb_rows(norm), "Laplacian High-pass (normalized)"

def apply_unsharp(rgb_rows: List[List[Tuple[int,int,int]]], amount: float):
    gray = to_gray_rows(rgb_rows)
    blur = box_blur3_gray(gray)
    h, w = len(gray), len(gray[0])
    out = [[clip8(int(round(gray[y][x] + amount*(gray[y][x]-blur[y][x])))) for x in range(w)] for y in range(h)]
    return gray_to_rgb_rows(out), f"Unsharp masking (amount={amount})"

def apply_highboost(rgb_rows: List[List[Tuple[int,int,int]]], A: float):
    if A <= 1.0: A = 1.1
    gray = to_gray_rows(rgb_rows)
    blur = box_blur3_gray(gray)
    h, w = len(gray), len(gray[0])
    out = [[clip8(int(round(A*gray[y][x] - (A-1)*blur[y][x]))) for x in range(w)] for y in range(h)]
    return gray_to_rgb_rows(out), f"Highboost filtering (A={A})"

def apply_sobel_gradient(rgb_rows: List[List[Tuple[int,int,int]]]):
    gray = to_gray_rows(rgb_rows)
    gxk = [[-1,0,1],[-2,0,2],[-1,0,1]]
    gyk = [[-1,-2,-1],[0,0,0],[1,2,1]]
    gx = conv3x3_gray_signed(gray, gxk)
    gy = conv3x3_gray_signed(gray, gyk)
    h, w = len(gray), len(gray[0])
    mag = [[(gx[y][x]**2 + gy[y][x]**2)**0.5 for x in range(w)] for y in range(h)]
    maxm = max(max(row) for row in mag) or 1.0
    norm = [[int(round(mag[y][x]/maxm*255)) for x in range(w)] for y in range(h)]
    return gray_to_rgb_rows(norm), "Gradient magnitude (Sobel)"
