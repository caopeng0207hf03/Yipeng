"""Shared style/helpers for generating report figures.

All figures are original schematic illustrations created with matplotlib,
so they can be redistributed with the report without copyright concerns.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Register a CJK font so Chinese labels render correctly.
_CJK_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]
CJK = None
for _p in _CJK_CANDIDATES:
    if os.path.exists(_p):
        fm.fontManager.addfont(_p)
        CJK = fm.FontProperties(fname=_p).get_name()
        break

if CJK:
    plt.rcParams["font.family"] = CJK
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["figure.dpi"] = 150

# A consistent, readable colour palette.
BLUE = "#2E5EAA"
GREEN = "#2E8B57"
ORANGE = "#E07B39"
RED = "#C0392B"
PURPLE = "#7D5BA6"
GREY = "#7F8C8D"
LIGHT = "#EAF1FB"
LIGHT2 = "#E9F7EF"
LIGHT3 = "#FDEBD0"

IMG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "images")
os.makedirs(IMG_DIR, exist_ok=True)


def save(fig, name):
    path = os.path.join(IMG_DIR, name)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", path)
    return path


def box(ax, x, y, w, h, text, fc=LIGHT, ec=BLUE, fs=11, lw=1.5, rounded=True, tc="black"):
    style = "round,pad=0.02,rounding_size=0.08" if rounded else "square,pad=0.02"
    from matplotlib.patches import FancyBboxPatch
    p = FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=lw, zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            zorder=3, color=tc, wrap=True)
    return (x + w / 2, y + h / 2)


def arrow(ax, xy1, xy2, color=GREY, lw=2, style="-|>", ls="-"):
    from matplotlib.patches import FancyArrowPatch
    a = FancyArrowPatch(xy1, xy2, arrowstyle=style, mutation_scale=15,
                        color=color, lw=lw, linestyle=ls, zorder=1,
                        shrinkA=2, shrinkB=2)
    ax.add_patch(a)
