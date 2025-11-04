# image_processing.py
"""
Manual point-processing methods for PCX viewer.
Uses manually implemented pixel operations (no NumPy/PIL built-ins).
"""

from typing import List, Tuple

# RGB row type alias
RGBRows = List[List[Tuple[int, int, int]]]

# ---------------------------------------------------------------------
# 1. Grayscale Transformation
# ---------------------------------------------------------------------
def to_grayscale(rgb_rows: RGBRows) -> RGBRows:
    """Convert to grayscale using s = (R + G + B) / 3."""
    gray_rows = []
    for row in rgb_rows:
        prow = []
        for (r, g, b) in row:
            s = (r + g + b) // 3
            prow.append((s, s, s))
        gray_rows.append(prow)
    return gray_rows

# ---------------------------------------------------------------------
# 2. Negative Transformation
# ---------------------------------------------------------------------
def to_negative(rgb_rows: RGBRows) -> RGBRows:
    """Negative transformation: s = 255 - r (for each channel)."""
    neg_rows = []
    for row in rgb_rows:
        prow = []
        for (r, g, b) in row:
            prow.append((255 - r, 255 - g, 255 - b))
        neg_rows.append(prow)
    return neg_rows

# ---------------------------------------------------------------------
# 3. Manual Threshold (Black/White)
# ---------------------------------------------------------------------
def manual_threshold(rgb_rows: RGBRows, threshold: int = 128) -> RGBRows:
    """Convert to black/white using grayscale average and threshold [0..255]."""
    threshold = max(0, min(255, int(threshold)))
    bw_rows = []
    for row in rgb_rows:
        prow = []
        for (r, g, b) in row:
            s = (r + g + b) // 3
            v = 255 if s >= threshold else 0
            prow.append((v, v, v))
        bw_rows.append(prow)
    return bw_rows

# ---------------------------------------------------------------------
# 4. Gamma (Power-law) Transformation
# ---------------------------------------------------------------------
def gamma_transform(rgb_rows: RGBRows, gamma: float = 1.0) -> RGBRows:
    """Power-law (gamma) transformation: s = 255 * (r/255)^gamma."""
    if gamma <= 0:
        gamma = 1.0
    # Precompute LUT
    lut = [min(255, max(0, int(round(255 * ((i / 255.0) ** gamma))))) for i in range(256)]
    gamma_rows = []
    for row in rgb_rows:
        prow = []
        for (r, g, b) in row:
            prow.append((lut[r], lut[g], lut[b]))
        gamma_rows.append(prow)
    return gamma_rows

# ---------------------------------------------------------------------
# 5. Histogram Equalization (Grayscale)
# ---------------------------------------------------------------------
def histogram_equalization(rgb_rows: RGBRows) -> RGBRows:
    """Histogram Equalization (grayscale version, step-by-step)."""
    # Step 1: Convert to grayscale
    gray_rows = to_grayscale(rgb_rows)

    # Step 2: Compute histogram
    hist = [0] * 256
    for row in gray_rows:
        for (r, _, _) in row:
            hist[r] += 1

    total = sum(hist)
    if total == 0:
        return gray_rows

    # Step 3: Compute cumulative distribution function (CDF)
    cdf = []
    csum = 0.0
    for h in hist:
        csum += h / total
        cdf.append(csum)

    # Step 4: Create mapping LUT
    lut = [min(255, max(0, int(round(255 * c)))) for c in cdf]

    # Step 5: Apply mapping to pixels
    eq_rows = []
    for row in gray_rows:
        prow = []
        for (r, _, _) in row:
            v = lut[r]
            prow.append((v, v, v))
        eq_rows.append(prow)

    return eq_rows
