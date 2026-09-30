import matplotlib.pyplot as plt

"""
Shared look for report figures. Colours are the first three slots of a
colour-blind-validated categorical palette, and each follows one entity
across every figure: 16-bit configs are blue, 8-bit configs orange, the
float model (its own error, its accuracy) aqua; reference levels are grey.
"""

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8984"
GRID = "#e8e7e2"
GRID_STRONG = "#c9c8c1"   # separators that must read above the grid
SURFACE = "#ffffff"

COLOR_16BIT = "#2a78d6"
COLOR_8BIT = "#eb6834"
COLOR_MODEL_ERROR = "#1baf7a"
COLOR_FLOAT = COLOR_MODEL_ERROR
COLOR_REFERENCE = INK_SECONDARY

# Categorical slots in fixed order, for figures whose series are not bit widths
SLOT_1, SLOT_2, SLOT_3 = COLOR_16BIT, COLOR_8BIT, COLOR_MODEL_ERROR

# Ordinal ramp (one hue, light -> dark) for series ordered by bit width
BIT_WIDTH_RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]

# Marker shape doubles the colour encoding for greyscale print
MARKER_16BIT = "o"
MARKER_8BIT = "s"
MARKER_MODEL_ERROR = "^"


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK_MUTED,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "font.size": 10,
        "legend.frameon": False,
        "legend.labelcolor": INK_SECONDARY,
        "lines.linewidth": 2,
        "lines.markersize": 7,
        "lines.markeredgecolor": SURFACE,   # thin surface ring keeps overlapping markers apart
        "lines.markeredgewidth": 1.2,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })
