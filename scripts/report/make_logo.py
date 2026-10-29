"""
make_logo.py
============
The pySUTO logo: a lavender pixel forest spirit on the left, ``pySUTO`` in
pastel-green pixel letters on the right, written as one SVG and one transparent PNG
from the same cell grid.

    python scripts/report/make_logo.py [--out figs/logo]

The letters are original glyphs on a 72 x 21 cell grid, drawn in the manner of
pixel wordmarks (three-cell strokes, one-cell bevels on the curved letters, the
``p`` and ``y`` hanging below the baseline). The spirit is an original 29 x 26
sprite (round ears, round eyes, a plain belly, a leaf sprout in the letters' green)
whose feet sit on the capitals' baseline. Every colour is flat; the colours live in
the USER CONFIGURATION block.

Writes ``pysuto-logo.svg`` (one rect per horizontal run of equal colour,
``shape-rendering="crispEdges"``) and ``pysuto-logo.png`` (``PX_PER_CELL``
pixels per cell, RGBA, transparent background) to ``--out``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

# =============================================================================
# USER CONFIGURATION
# =============================================================================
GREEN         = "#8fd1a6"   # pastel green: the letters and the leaf
MASCOT        = {
    "b": "#b9a7e3",         # lavender body
    "l": "#f2eefb",         # belly
    "e": "#ffffff",         # eye whites
    "k": "#3b3252",         # pupils and nose
    "g": GREEN,             # leaf sprout
}
MASCOT_GAP    = 6           # empty cells between the spirit and the ``p``
SVG_CELL      = 10          # SVG user units per cell
PX_PER_CELL   = 28          # PNG pixels per cell (107 cells -> 2996 px wide)

# =============================================================================
# GLYPHS: '#' filled, '.' empty. Caps occupy rows 0-16; lowercase rows 5-20.
# =============================================================================
GLYPHS = {
    "S": [
        "..#######..",
        ".#########.",
        "###########",
        "###.....###",
        "###.....###",
        "###........",
        "###........",
        "#########..",
        ".#########.",
        "..#########",
        "........###",
        "........###",
        "###.....###",
        "###.....###",
        "###########",
        ".#########.",
        "..#######..",
    ],
    "U": [
        "###.....###",
        "###.....###",
        "###.....###",
        "###.....###",
        "###.....###",
        "###.....###",
        "###.....###",
        "###.....###",
        "###.....###",
        "###.....###",
        "###.....###",
        "###.....###",
        "###.....###",
        "###.....###",
        "###########",
        ".#########.",
        "..#######..",
    ],
    "T": [
        "#########",
        "#########",
        "#########",
        "...###...",
        "...###...",
        "...###...",
        "...###...",
        "...###...",
        "...###...",
        "...###...",
        "...###...",
        "...###...",
        "...###...",
        "...###...",
        "...###...",
        "...###...",
        "...###...",
    ],
    "O": [
        "..#######..",
        ".#########.",
        "###########",
        "###.....###",
        "###.....###",
        "###.....###",
        "###.....###",
        "###.....###",
        "###.....###",
        "###.....###",
        "###.....###",
        "###.....###",
        "###.....###",
        "###.....###",
        "###########",
        ".#########.",
        "..#######..",
    ],
    "p": [
        "########..",
        "#########.",
        "##########",
        "###....###",
        "###....###",
        "###....###",
        "###....###",
        "###....###",
        "###....###",
        "##########",
        "#########.",
        "########..",
        "###.......",
        "###.......",
        "###.......",
        "###.......",
    ],
    "y": [
        "###....###",
        "###....###",
        "###....###",
        "###....###",
        "###....###",
        "###....###",
        "###....###",
        "###....###",
        "###....###",
        "##########",
        ".#########",
        "..########",
        ".......###",
        ".#########",
        "##########",
        ".########.",
    ],
}
WORD       = "pySUTO"
ROWS       = 21
LOWER_ROW0 = 5
GAP        = 2
BASELINE   = 16             # last row of the capitals

# =============================================================================
# MASCOT: one character per cell, '.' empty, colours in MASCOT.
# =============================================================================
SPIRIT = [
    "...............ggg...........",
    ".....bbb......ggg....bbb.....",
    "....bbbbb.....g.....bbbbb....",
    "....bbbbbb....g....bbbbbb....",
    ".....bbbbbbbbbbbbbbbbbbb.....",
    "....bbbbbbbbbbbbbbbbbbbbb....",
    "....bbbeebbbbkkkbbbbeebbb....",
    "...bbbekkebbbbbbbbbekkebbb...",
    "...bbbekkebbbbbbbbbekkebbb...",
    "...bbbbeebbbbbbbbbbbeebbbb...",
    "...bbbbbbbbbbbbbbbbbbbbbbb...",
    "...bbbbbbbbbbbbbbbbbbbbbbb...",
    "..bbbbbbbbbbbbbbbbbbbbbbbbb..",
    "..bbbbbbblllllllllllbbbbbbb..",
    "..bbbbblllllllllllllllbbbbb..",
    ".bbbbblllllllllllllllllbbbbb.",
    ".bbbblllllllllllllllllllbbbb.",
    ".bbbblllllllllllllllllllbbbb.",
    ".bbblllllllllllllllllllllbbb.",
    ".bbblllllllllllllllllllllbbb.",
    ".bbblllllllllllllllllllllbbb.",
    ".bbblllllllllllllllllllllbbb.",
    "..bbblllllllllllllllllllbbb..",
    "..bbbblllllllllllllllllbbbb..",
    "...bbbbblllllllllllllbbbbb...",
    "....bbbbbbb.......bbbbbbb....",
]


# =============================================================================
# Grid
# =============================================================================
def compose(word: str) -> np.ndarray:
    """Lay the word's glyphs side by side.

    Returns
    -------
    np.ndarray
        Bool mask of shape ``(ROWS, width)``; True where a cell is filled.
    """
    blocks = []
    for ch in word:
        glyph = GLYPHS[ch]
        block = np.zeros((ROWS, len(glyph[0])), dtype=bool)
        row0  = 0 if ch.isupper() else LOWER_ROW0
        for i, row in enumerate(glyph):
            block[row0 + i] = np.frombuffer(row.encode(), dtype=np.uint8) == ord("#")
        blocks.append(block)
        blocks.append(np.zeros((ROWS, GAP), dtype=bool))
    return np.concatenate(blocks[:-1], axis=1)


def logo() -> np.ndarray:
    """Place the spirit left of the word, its feet on the capitals' baseline.

    Returns
    -------
    np.ndarray
        Object array of hex strings, ``""`` where the cell is empty.
    """
    word   = compose(WORD)
    sprite = np.array([list(row) for row in SPIRIT])
    th, tw = sprite.shape
    wh, ww = word.shape
    top_w  = th - 1 - BASELINE                 # word row 0 on the logo grid
    out    = np.full((max(th, top_w + wh), tw + MASCOT_GAP + ww), "", dtype=object)

    for ch, hx in MASCOT.items():
        out[:th, :tw][sprite == ch] = hx
    out[top_w:top_w + wh, tw + MASCOT_GAP:][word] = GREEN
    return out


# =============================================================================
# Writers
# =============================================================================
def write_svg(colours: np.ndarray, path: Path) -> None:
    """One ``<rect>`` per horizontal run of equal colour, grouped by colour."""
    h, w = colours.shape
    runs: dict = {}
    for r in range(h):
        c = 0
        while c < w:
            hx = colours[r, c]
            if not hx:
                c += 1
                continue
            c1 = c
            while c1 < w and colours[r, c1] == hx:
                c1 += 1
            runs.setdefault(hx, []).append((c, r, c1 - c))
            c = c1
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w * SVG_CELL} {h * SVG_CELL}" '
        f'width="{w * SVG_CELL}" height="{h * SVG_CELL}" shape-rendering="crispEdges">',
        "  <title>pySUTO</title>",
    ]
    for hx in sorted(runs):
        lines.append(f'  <g fill="{hx}">')
        for c, r, n in runs[hx]:
            lines.append(
                f'    <rect x="{c * SVG_CELL}" y="{r * SVG_CELL}" '
                f'width="{n * SVG_CELL}" height="{SVG_CELL}"/>'
            )
        lines.append("  </g>")
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_png(colours: np.ndarray, path: Path) -> None:
    """RGBA raster, ``PX_PER_CELL`` pixels per cell, transparent where empty."""
    h, w = colours.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for r in range(h):
        for c in range(w):
            hx = colours[r, c]
            if hx:
                rgba[r, c] = [int(hx[i:i + 2], 16) for i in (1, 3, 5)] + [255]
    big = np.repeat(np.repeat(rgba, PX_PER_CELL, axis=0), PX_PER_CELL, axis=1)
    Image.fromarray(big, "RGBA").save(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="figs/logo", help="output directory")
    args = ap.parse_args()
    odir = Path(args.out)
    odir.mkdir(parents=True, exist_ok=True)

    colours = logo()
    write_svg(colours, odir / "pysuto-logo.svg")
    write_png(colours, odir / "pysuto-logo.png")
    print(f"grid {colours.shape[1]} x {colours.shape[0]} cells, "
          f"{int((colours != '').sum())} filled -> {odir}/pysuto-logo.{{svg,png}}")


if __name__ == "__main__":
    main()
