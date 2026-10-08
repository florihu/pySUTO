"""
make_logo.py
============
The pySUTO wordmark: ``pySUTO`` as pixel letters, coloured like oxidised copper,
written as one SVG and one transparent PNG from the same cell grid.

    python scripts/report/make_logo.py [--out figs/logo]

The letters are original glyphs on a 72 x 21 cell grid, drawn in the manner of
pixel wordmarks (three-cell strokes, one-cell bevels on the curved letters, the
``p`` and ``y`` hanging below the baseline). Every filled cell takes one shade of
verdigris from a smooth seeded noise field; a second field exposes patches of bare
copper and darkens the verdigris cells around them (cuprite rim). The seed is fixed,
so re-running the script reproduces the same pixels; the knobs that set the look
live in the USER CONFIGURATION block.

Writes ``pysuto-wordmark.svg`` (one rect per horizontal run of equal colour,
``shape-rendering="crispEdges"``) and ``pysuto-wordmark.png`` (``PX_PER_CELL``
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
SEED          = 7
VERDIGRIS     = ["#1f6f62", "#2a8c7a", "#3a9f8e", "#4fb5a2", "#6cc4b0"]  # dark -> light
COPPER        = ["#5a3620", "#9a5a2b", "#b87333", "#d08a4a"]             # cuprite -> bright
COPPER_ABOVE  = 0.80     # share of the copper field above which a cell is bare copper
RIM_PROB      = 0.5      # probability that a verdigris cell touching copper darkens
SVG_CELL      = 10       # SVG user units per cell
PX_PER_CELL   = 28       # PNG pixels per cell (72 cells -> 2016 px wide)

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


def value_noise(shape: tuple, cell: int, rng: np.random.Generator) -> np.ndarray:
    """Smooth field in [0, 1]: a coarse random lattice, bilinearly upsampled."""
    h, w   = shape
    lat    = rng.random((h // cell + 2, w // cell + 2))
    ys, xs = np.arange(h) / cell, np.arange(w) / cell
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    fy, fx = (ys - y0)[:, None], (xs - x0)[None, :]
    fy, fx = fy * fy * (3 - 2 * fy), fx * fx * (3 - 2 * fx)
    a, b   = lat[y0][:, x0], lat[y0][:, x0 + 1]
    c, d   = lat[y0 + 1][:, x0], lat[y0 + 1][:, x0 + 1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def patina(mask: np.ndarray, seed: int) -> np.ndarray:
    """Colour every filled cell.

    Returns
    -------
    np.ndarray
        Object array of hex strings, ``""`` where the cell is empty.
    """
    rng  = np.random.default_rng(seed)
    h, w = mask.shape
    out  = np.full((h, w), "", dtype=object)

    tone = 0.75 * value_noise((h, w), 4, rng) + 0.25 * rng.random((h, w))
    idx  = np.clip((tone * len(VERDIGRIS)).astype(int), 0, len(VERDIGRIS) - 1)
    out[mask] = np.array(VERDIGRIS, dtype=object)[idx][mask]

    field  = 0.7 * value_noise((h, w), 3, rng) + 0.3 * rng.random((h, w))
    copper = (field > COPPER_ABOVE) & mask
    cu_idx = np.clip((rng.random((h, w)) * len(COPPER)).astype(int), 0, len(COPPER) - 1)
    out[copper] = np.array(COPPER, dtype=object)[cu_idx][copper]

    pad  = np.pad(copper, 1)
    near = (pad[:-2, 1:-1] | pad[2:, 1:-1] | pad[1:-1, :-2] | pad[1:-1, 2:]) & mask & ~copper
    out[near & (rng.random((h, w)) < RIM_PROB)] = VERDIGRIS[0]
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

    mask    = compose(WORD)
    colours = patina(mask, SEED)
    write_svg(colours, odir / "pysuto-wordmark.svg")
    write_png(colours, odir / "pysuto-wordmark.png")
    n_cu = int(np.isin(colours, COPPER).sum())
    print(f"grid {mask.shape[1]} x {mask.shape[0]} cells, {int(mask.sum())} filled, "
          f"{n_cu} bare copper -> {odir}/pysuto-wordmark.{{svg,png}}")


if __name__ == "__main__":
    main()
