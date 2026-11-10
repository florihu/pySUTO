"""
make_logo.py
============
The pySUTO logo: an original Alpine marmot standing sentinel, drawn in code as a wood
engraving and hand-coloured, beside ``pySUTO`` in JetBrains Mono and its tagline in
Inter.

    python scripts/report/make_logo.py [--out figs/logo]

The marmot is a stack of closed splines (tail, far arm, body, foot, arm, throat, head,
ear) in a 1000 x 1400 design space, y down, facing right. Each part is inflated into a
height field (the square root of a Poisson membrane, plus thigh, shoulder, chest and
belly masses on the body, and skull, jowl and muzzle masses on the head), shaded against
an upper-right light, and engraved: hatch lines laid along a per-part line field, their
width set by the tone, broken like hair and grizzled with short pale and dark flicks.
The head is drawn level in its own frame and raised on the neck; its lines run from the
nose back over skull and cheek and down the nape into the neck, and its chin overhangs a
pale throat into which the jowl's fur merges.
Where a limb grows out of the body the two furs interlock over a feather band; where it
overhangs, the edge stays crisp and casts a shadow.

Colour is laid like a hand-coloured natural-history plate. Each part carries a local
coat colour (``COAT``: grizzled grey-brown back, golden-tawny belly and forelegs,
grey head with a blackish crown, pale snout and chin, orange incisors, slate digits,
a tail dark over its distal half), composited with the same feathered alphas as the
engraving and shaded by the same height field: warm on the lit side, cooler into the
shadows and under cast shadows. On white it is a translucent wash under umber
linework, its pigment pooled at the edge, with the pale guard-hair tips laid over it.
The dark version is painted colour-first: the wash becomes an opaque, darker
underpainting and the white-line engraving (lines where light falls) is printed over
it in light tints of the coat, with a pale rim on the silhouette.

Writes the colour logo, ``pysuto-logo-light.png`` and ``pysuto-logo-dark.png`` (the
README picks one by ``prefers-color-scheme``), and the engraving alone in one ink,
``pysuto-logo-bw-light.png`` and ``pysuto-logo-bw-dark.png``: RGBA on a transparent
background, ``WIDTH_PX`` wide and cropped alike, to ``--out``. JetBrains Mono and Inter
(both SIL OFL 1.1, variable) ship in ``scripts/report/fonts/``. All four take about 40 s.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import interpolate, ndimage, sparse
from scipy.sparse.linalg import spsolve
from scipy.special import ndtr

# =============================================================================
# USER CONFIGURATION
# =============================================================================
FONTS      = Path(__file__).parent / "fonts"
WORD_FONT  = (FONTS / "JetBrainsMono[wght].ttf", 700)     # variable font, weight
TAG_FONT   = (FONTS / "Inter[opsz,wght].ttf", 400)
WORD       = "pySUTO"
TAGLINE    = "Python Supply–Use Table Optimiser"
LINE       = {"light": "#221b16", "dark": "#f1e8d6"}   # engraving: umber / bone
TEXT_INK   = {"light": "#221b16", "dark": "#ece4d6"}   # word and tagline
BW_INK     = {"light": "#111111", "dark": "#ececec"}   # the one-ink logo
WASH_ALPHA = 0.82                                      # wash density, light mode
WASH_INSET = {"light": 1.0, "dark": 0.0}               # wash edge inside the line
TINT = {
    "cool":        "#c8cce0",   # multiplier into the shadow side: cooler, darker
    "warm":        "#f2c27a",   # lift on the lit side
    "grizzle":     "#dccfb3",   # pale guard-hair tips
    "ground":      "#5f6671",   # cast-shadow lines
    "floor":       "#a3a9b2",   # cast-shadow wash
    "glint":       "#fffdf8",   # catch lights
    "ground_dark": "#8d97a4",
    "rim":         "#cfc4b2",   # silhouette and fringe hairs in the dark version
    "floor_dark":  "#3a4350",
    "eye":         "#120e0c",   # eye, nose and mouth in the dark version
    "claw_dark":   "#77706a",   # horn-grey claws in the dark version
}
CAP_FRAC   = 0.38       # cap height of the word / marmot height
GAP_FRAC   = 0.01       # marmot -> word gap / marmot height
TAG_GAP    = 0.55       # word baseline -> tagline cap top / cap height
WIDTH_PX   = 1200       # PNG width (the README shows it at 600)
SCALE      = 1.0        # px per design unit while drawing the marmot
SEED       = 7

COAT = {                     # Alpine marmot coat, local colour before shading
    "back":      "#7a6c5d",  # grizzled grey-brown back and flank top
    "back_dark": "#4a433d",  # nape, spine and the back of the head
    "flank":     "#b27d4a",  # golden-tawny belly and lower flank
    "throat":    "#cdb08a",  # pale buff throat and upper chest
    "thigh":     "#8c6648",  # haunch
    "limb":      "#ad7745",  # front legs, the chest's tawny
    "paw":       "#4a4440",  # slate-skinned digits
    "foot":      "#665a50",  # hind foot
    "crown":     "#33302e",  # blackish-grey crown
    "cheek":     "#88827a",  # grizzled grey cheek
    "muzzle":    "#e9e2d4",  # pale snout, lips and chin
    "ear":       "#5e554d",  # small furred ear, the head's grey-brown
    "tail":      "#7a5e45",  # tail base
    "tail_tip":  "#221c18",  # dark distal tail
    "incisor":   "#d0842e",  # orange enamel of the upper incisors
    "incisor_tip": "#f0dcae",  # their worn, paler cutting edge
}

W, H   = 1000, 1400                 # design units
S      = SCALE
LIGHT  = np.array([0.45, -0.60, 0.66]); LIGHT /= np.linalg.norm(LIGHT)
RNG    = np.random.default_rng(SEED)
NX, NY = int(W * S), int(H * S)
YY, XX = np.mgrid[0:NY, 0:NX] / S   # design coordinates of every pixel


# =============================================================================
# Geometry and noise
# =============================================================================
def spline(pts, closed=True, n=800):
    p = np.asarray(pts, float)
    if closed:
        p = np.vstack([p, p[:1]])
    tck, _ = interpolate.splprep([p[:, 0], p[:, 1]], s=0, per=int(closed),
                                 k=min(3, len(p) - 1))
    x, y = interpolate.splev(np.linspace(0, 1, n), tck)
    return np.c_[x, y]


def scallop(poly, seed, waves=((70.0, 6.0), (22.0, 2.5))):
    """Fur clumps: push a closed outline along its normal by smooth periodic noise."""
    r = np.random.default_rng(seed)
    seg = np.roll(poly, -1, axis=0) - np.roll(poly, 1, axis=0)
    t = seg / (np.hypot(*seg.T)[:, None] + 1e-12)
    nrm = np.c_[t[:, 1], -t[:, 0]]
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(poly, axis=0).T))]
    d = np.zeros(len(poly))
    for wave, amp in waves:
        m = max(4, int(s[-1] / wave))
        v = r.uniform(-1, 1, m)
        knots = np.linspace(0, s[-1], m + 1)
        d += amp * interpolate.CubicSpline(knots, np.r_[v, v[:1]], bc_type="periodic")(s)
    return poly + nrm * d[:, None]


def bump(cx, cy, r, amp):
    """A smooth compact dome added to a height field (units)."""
    q = np.clip(1 - ((XX - cx) ** 2 + (YY - cy) ** 2) / r ** 2, 0, 1)
    return amp * q ** 2


def fill(poly):
    img = Image.new("L", (NX, NY), 0)
    ImageDraw.Draw(img).polygon([tuple(q * S) for q in poly], fill=255)
    return np.asarray(img) > 127


def ellipse(cx, cy, rx, ry, rot=0.0, n=200):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c, s = np.cos(rot), np.sin(rot)
    x, y = rx * np.cos(t), ry * np.sin(t)
    return np.c_[cx + c * x - s * y, cy + s * x + c * y]


def gauss(cx, cy, sx, sy=None):
    sy = sx if sy is None else sy
    return np.exp(-0.5 * (((XX - cx) / sx) ** 2 + ((YY - cy) / sy) ** 2))


def smoothstep(e0, e1, v):
    t = np.clip((v - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def smooth_noise(sigma, amp=1.0):
    n = ndimage.gaussian_filter(RNG.standard_normal((NY, NX)), sigma * S)
    return amp * n / (n.std() + 1e-12)


def streaks(a, along=14.0, across=2.0):
    """Unit-variance noise stretched along direction a: hair-like streaks."""
    a = np.asarray(a, float) / np.hypot(*a)
    b = np.array([-a[1], a[0]])
    u = XX * a[0] + YY * a[1]
    v = XX * b[0] + YY * b[1]
    u, v = (u - u.min()) / along, (v - v.min()) / across
    lat = RNG.standard_normal((int(u.max()) + 4, int(v.max()) + 4))
    n = ndimage.map_coordinates(lat, [u.ravel(), v.ravel()], order=3).reshape(u.shape)
    return n / (n.std() + 1e-12)


# =============================================================================
# Shading and engraving
# =============================================================================
def inflate(mask, k=4, joined=None, level=0.65):
    """Height field of a mask: sqrt of the Poisson membrane, in units. Across
    ``joined`` (pixels outside the mask) the membrane is pinned at ``level`` x its
    peak instead of at 0: there the part runs on into the one it grows out of instead
    of rolling off."""
    m = mask[::k, ::k]
    n = int(m.sum())
    idx = -np.ones(m.shape, int)
    idx[m] = np.arange(n)
    ii, jj = np.nonzero(m)
    r, c, v = [idx[ii, jj]], [idx[ii, jj]], [np.full(n, 4.0)]
    nj_ = np.zeros(n)                       # joined neighbours of each pixel
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ni, nj = ii + di, jj + dj
        ok = (ni >= 0) & (ni < m.shape[0]) & (nj >= 0) & (nj < m.shape[1])
        if joined is not None:
            nj_[ok] += joined[::k, ::k][ni[ok], nj[ok]] & ~m[ni[ok], nj[ok]]
        ok[ok] = m[ni[ok], nj[ok]]
        r.append(idx[ii[ok], jj[ok]]); c.append(idx[ni[ok], nj[ok]])
        v.append(np.full(int(ok.sum()), -1.0))
    A = sparse.csc_matrix((np.concatenate(v), (np.concatenate(r), np.concatenate(c))),
                          shape=(n, n))
    u = np.zeros(m.shape)
    u[m] = spsolve(A, np.ones(n))
    if joined is not None:
        u[m] = spsolve(A, 1 + level ** 2 * u.max() * nj_)
    h = 2.0 * k * np.sqrt(u) / S
    h = ndimage.zoom(h, k, order=1)
    out = np.zeros(mask.shape)
    out[:min(NY, h.shape[0]), :min(NX, h.shape[1])] = h[:NY, :NX]
    out = ndimage.gaussian_filter(out, 2 * S)
    return np.where(mask, out, 0.0)


def lambert(h):
    gy, gx = np.gradient(h)
    gx, gy = gx * S, gy * S
    nz = 1.0 / np.sqrt(1 + gx ** 2 + gy ** 2)
    nx, ny = -gx * nz, -gy * nz
    return np.clip(nx * LIGHT[0] + ny * LIGHT[1] + nz * LIGHT[2], 0, 1)


def extents(mask, axis):
    """Per-row (axis=1) or per-column (axis=0) min/max coordinate, smoothed."""
    coord = XX if axis == 1 else YY
    big = np.where(mask, coord, np.inf).min(axis=axis)
    sml = np.where(mask, coord, -np.inf).max(axis=axis)
    ok = np.isfinite(big)
    lo, hi = np.interp(np.arange(len(big)), np.nonzero(ok)[0], big[ok]), \
        np.interp(np.arange(len(sml)), np.nonzero(ok)[0], sml[ok])
    lo, hi = ndimage.gaussian_filter1d(lo, 6 * S), ndimage.gaussian_filter1d(hi, 6 * S)
    if axis == 1:
        return lo[:, None], hi[:, None]
    return lo[None, :], hi[None, :]


def wrap_coord(mask, axis, span=0.0, wrap=0.6):
    """Across-part coordinate in [0, 1], bunched toward the edges like a cylinder.
    Where the part is narrower than ``span`` units the lines keep that spacing and run
    off the edge instead of pinching together."""
    lo, hi = extents(mask, axis)
    mid, half = (lo + hi) / 2, np.maximum((hi - lo) / 2, span / 2)
    lo, hi = mid - half, mid + half
    coord = XX if axis == 1 else YY
    rel = np.clip((coord - lo) / np.maximum(hi - lo, 1e-6), 0, 1)
    th = np.arcsin(2 * rel - 1) / np.pi + 0.5
    return (1 - wrap) * rel + wrap * th


def stream_coord(mask, poly, keys, n, k=4):
    """Hair-line coordinate of the head: harmonic inside it, 0 along the crown and the
    back of the skull and ``n`` along the jaw, rising evenly across the muzzle and across
    the neck, so its level lines run from the nose back over the skull and the cheek and
    down the nape into the neck."""
    P = poly[:-1]
    seg = np.hypot(*np.diff(np.vstack([P, P[:1]]), axis=0).T)
    s = np.r_[0, np.cumsum(seg)][:-1]
    L = seg.sum()
    t = (s - s[keys["nape"]]) % L
    ta, tb, tc = ((s[keys[q]] - s[keys["nape"]]) % L for q in ("nose", "chin", "jaw"))
    v = np.where(t <= ta, 0.0, np.where(t <= tb, n * (t - ta) / (tb - ta),
                 np.where(t <= tc, float(n), n * (L - t) / (L - tc))))
    m = mask[::k, ::k]
    ring = m & ~ndimage.binary_erosion(m)
    inner = m & ~ring
    ii, jj = np.nonzero(ring)
    pix = np.c_[jj, ii] * k / S
    near = np.argmin((pix[:, None, 0] - P[None, :, 0]) ** 2
                     + (pix[:, None, 1] - P[None, :, 1]) ** 2, axis=1)
    val = np.zeros(m.shape)
    val[ring] = v[near]
    idx = -np.ones(m.shape, int)
    nin = int(inner.sum())
    idx[inner] = np.arange(nin)
    ii, jj = np.nonzero(inner)
    r, c, w = [idx[ii, jj]], [idx[ii, jj]], [np.full(nin, 4.0)]
    rhs = np.zeros(nin)
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ni, nj = ii + di, jj + dj
        isin = inner[ni, nj]
        r.append(idx[ii[isin], jj[isin]]); c.append(idx[ni[isin], nj[isin]])
        w.append(np.full(int(isin.sum()), -1.0))
        np.add.at(rhs, idx[ii[~isin], jj[~isin]], val[ni[~isin], nj[~isin]])
    A = sparse.csc_matrix((np.concatenate(w), (np.concatenate(r), np.concatenate(c))),
                          shape=(nin, nin))
    val[inner] = spsolve(A, rhs)
    _, (fi, fj) = ndimage.distance_transform_edt(~m, return_indices=True)
    val = ndimage.zoom(val[fi, fj], k, order=1)[:NY, :NX]
    return ndimage.gaussian_filter(val, 1.5 * S)


def field_streaks(phi, psi, along=10.0, across=0.13, seed=SEED + 3):
    """Unit-variance noise stretched along the level lines of phi (psi runs along them)."""
    u, v = (psi - psi.min()) / along, (phi - phi.min()) / across
    lat = np.random.default_rng(seed).standard_normal((int(u.max()) + 4, int(v.max()) + 4))
    n = ndimage.map_coordinates(lat, [u.ravel(), v.ravel()], order=3).reshape(u.shape)
    return n / (n.std() + 1e-12)


def hashk(k, salt):
    return np.modf(np.sin(k * 12.9898 + salt * 78.233) * 43758.5453)[0] % 1.0


def engrave(phi, psi, tone, dash=40.0, salt=0.0):
    """Coverage of width-modulated, hair-broken hatch lines at integer phi."""
    k = np.round(phi)
    f = np.mod(psi / dash + hashk(k, salt), 1.0)
    duty = np.clip(0.70 + 0.5 * tone, 0, 1)
    q = np.clip(f / duty, 0, 1)
    prof = np.where(f < duty, np.clip(np.minimum(q, 1 - q) / 0.12, 0, 1), 0.0)
    solid = np.clip((tone - 0.70) / 0.25, 0, 1)
    te = tone * (solid + (1 - solid) * prof)
    gy, gx = np.gradient(phi)
    g = np.hypot(gx, gy) + 1e-9
    fr = np.abs(phi - k)
    return np.clip(0.5 + (te / 2 - fr) / g, 0, 1)


def stroke(pts, w0, w1=None, closed=False, n=400):
    """Coverage of a tapered stroke along a spline (widths in units)."""
    w1 = w0 if w1 is None else w1
    p = spline(pts, closed=closed, n=n)
    t = np.linspace(0, 1, len(p))
    w = w0 + (w1 - w0) * t
    img = Image.new("L", (NX, NY), 0)
    d = ImageDraw.Draw(img)
    for (x, y), r in zip(p, w / 2):
        d.ellipse([(x - r) * S, (y - r) * S, (x + r) * S, (y + r) * S], fill=255)
    return ndimage.gaussian_filter(np.asarray(img) / 255.0, 0.5)


def flicks(mask, phi, tone, flow, seed, pale_p=0.45, dark_p=0.35, L=16.0, w=2.4):
    """Grizzled coat: short tapered hairs along the line field, pale tips where the
    coat is dark and dark hairs where it is light. Returns (pale, dark) coverage."""
    r = np.random.default_rng(seed)
    iy, ix = np.nonzero(mask)
    n = int(len(iy) / S ** 2 * 0.012)
    pick = r.integers(0, len(iy), n)
    iy, ix = iy[pick], ix[pick]
    gy, gx = np.gradient(phi)
    th = np.arctan2(-gx[iy, ix], gy[iy, ix])
    th = np.where(np.cos(th) * flow[0] + np.sin(th) * flow[1] < 0, th + np.pi, th)
    th = th + r.normal(0, 0.15, n)
    ax, ay = np.cos(th), np.sin(th)
    l, hw = L * r.uniform(0.6, 1.3, n) * S / 2, w * S / 2
    out = []
    t = tone[iy, ix]
    for p in (pale_p * smoothstep(0.40, 0.65, t) * (1 - smoothstep(0.80, 0.95, t)),
              dark_p * smoothstep(0.32, 0.10, t)):
        keep = r.uniform(0, 1, n) < p
        img = Image.new("L", (NX, NY), 0)
        d = ImageDraw.Draw(img)
        for j in np.nonzero(keep)[0]:
            x0, y0, dx, dy = ix[j], iy[j], ax[j] * l[j], ay[j] * l[j]
            d.polygon([(x0 - dx - ay[j] * hw, y0 - dy + ax[j] * hw),
                       (x0 - dx + ay[j] * hw, y0 - dy - ax[j] * hw),
                       (x0 + dx, y0 + dy)], fill=255)
        out.append(ndimage.gaussian_filter(np.asarray(img) / 255.0, 0.6 * S))
    return out


# =============================================================================
# The marmot: parts back to front, in design units
# =============================================================================
TAIL = scallop(spline([(370, 1262), (320, 1262), (270, 1258), (220, 1248), (170, 1234),
               (125, 1222), (85, 1218), (52, 1228), (30, 1244), (20, 1260), (30, 1278),
               (60, 1302), (100, 1320), (160, 1332), (230, 1337), (300, 1336),
               (350, 1326), (375, 1300)]), seed=5,
               waves=((40.0, 9.0), (12.0, 5.0)))   # narrow at the rump, a brush distally
ARM = spline([(590, 440), (640, 452), (676, 488), (700, 526), (734, 542), (768, 550),
              (792, 566), (806, 592), (810, 626), (806, 660), (796, 684), (780, 688),
              (768, 672), (760, 646), (752, 620), (722, 616), (682, 616), (642, 604),
              (606, 566), (582, 500)])
PAW_FAR = np.array([16.0, 8.0])
ARM_FAR = ARM + PAW_FAR
BODY = scallop(spline([
    (455, 195), (418, 252), (372, 330), (330, 420), (305, 520), (288, 630),
    (266, 750), (240, 870), (228, 975), (236, 1080), (256, 1168), (296, 1244),
    (360, 1300), (450, 1333), (575, 1337), (680, 1310), (730, 1265), (744, 1215),
    (734, 1168), (746, 1110), (768, 1030), (776, 940), (766, 840), (746, 740),
    (730, 650), (718, 560), (704, 470), (690, 400), (668, 350), (600, 325),
    (520, 268)], n=1200), seed=3)
FOOT = spline([(575, 1346), (560, 1322), (590, 1290), (640, 1288), (690, 1296),
               (722, 1302), (736, 1305), (744, 1309), (749, 1317), (756, 1313),
               (763, 1318), (766, 1325), (773, 1321), (781, 1328), (784, 1338),
               (777, 1346), (762, 1348), (690, 1348), (620, 1348)])  # three toes, side on
# The head is drawn level in its own frame (head units) and set on the neck by a pose:
# scaled by HEAD_K and raised by HEAD_TILT (radians, negative lifts the nose) about
# the nape. HEAD runs from inside the neck up the back of the skull, over the crown, the
# nose and the chin, back along the lower jaw and inside the neck; HEAD_KEYS name where
# the crown's hair lines start (low on the nape, so they run down onto the back), the
# nose top, the chin and the jaw. The throat is a part of its own under the jaw.
HG = dict(
    # nape, flat crown; the nasal profile falls to the nose and bends down at its tip
    # (Kryštufek & Vohralík 2013); the cleft upper lip falls back from the nose pad to
    # the mouth, the chin sits under the incisors, the jaw line runs back into the neck
    pts=[(452, 292), (437, 252), (440, 214), (454, 190), (479, 168), (515, 153),
         (560, 142), (608, 137), (652, 140), (686, 150), (714, 165), (740, 184),
         (761, 203), (776, 220), (785, 232), (789, 245), (787, 257), (781, 268),
         (771, 278), (758, 286), (745, 290), (737, 293), (734, 299), (737, 306),
         (734, 313), (726, 320), (712, 329), (694, 338), (670, 346), (645, 350),
         (610, 350), (560, 341), (505, 317)],
    keys=dict(nape=0, nose=13, chin=24, jaw=29),
    pose=dict(pivot=(450.0, 200.0), k=0.94, tilt=-0.15),
    ear=[(513, 197), (510, 180), (515, 164), (528, 155), (543, 154), (553, 163),
         (557, 179), (554, 197)],                     # its rim level with the crown
    ear_c=(534, 176), ear_base=(183, 197), ear_rim=166,
    eye=(648, 187, 19.5, 14.0, 0.24),                 # large, almond, along the profile
    nose=[(770, 236), (775, 229), (781, 226), (786, 231), (789, 241), (788, 252),
          (783, 259), (776, 261), (771, 256), (769, 246)],   # naked pad, furred behind
    nostril=[(784, 241), (780, 248), (773, 252)],    # comma slit at its front
    nose_glint=(783, 233), cleft=[(781, 260), (775, 269)],
    tooth=[(736, 290), (739.5, 297), (741, 305), (740.5, 313), (739, 319),
           (734.5, 318.5), (730.5, 315), (729.5, 306), (729, 298), (728, 290)],
                                                      # a chisel hanging from the lip
    lip=[(743, 291), (734, 292), (725, 294)],       # the upper lip's margin over it
    gape=[(727, 302), (714, 307), (700, 310)],       # whiskers: rows on the upper lip
    whiskers=[[(763, 263), (826, 250), (888, 240)], [(758, 272), (826, 266), (888, 276)],
              [(752, 277), (814, 291), (870, 314)], [(768, 260), (808, 259), (872, 214)],
              [(746, 283), (794, 303), (838, 336)], [(765, 268), (832, 258), (894, 258)]],
    relief=[(590, 182, 115, 16), (664, 284, 82, 13), (754, 250, 48, 13),
            (650, 162, 30, 6), (694, 190, 22, -5)],
    albedo=[(0.75, 600, 142, 125, 32), (0.05, 612, 172, 30, 16), (-0.52, 770, 266, 34, 32),
            (-0.22, 722, 204, 22, 16), (-0.40, 738, 306, 34, 12), (-0.14, 640, 270, 55, 30),
            (-0.18, 662, 205, 30, 10)],               # the cap's rim runs through the eye
    coat=dict(back_dark=[(0.9, 468, 245, 40, 90)],
              crown=[(1.25, 600, 134, 160, 33), (0.35, 540, 235, 45, 40)],
              throat=[(0.55, 640, 344, 85, 20)],      # the jowl grizzles into the throat
              muzzle=[(1.2, 772, 264, 26, 30), (0.6, 722, 204, 22, 14),
                      (0.95, 738, 304, 32, 12)]),
)
HEAD_PIVOT = np.asarray(HG["pose"]["pivot"], float)
HEAD_K, HEAD_TILT = HG["pose"]["k"], HG["pose"]["tilt"]


def head_xy(x, y):
    """Design coordinates of a point given in head units."""
    c, s = np.cos(HEAD_TILT), np.sin(HEAD_TILT)
    dx, dy = HEAD_K * (x - HEAD_PIVOT[0]), HEAD_K * (y - HEAD_PIVOT[1])
    return float(HEAD_PIVOT[0] + c * dx - s * dy), float(HEAD_PIVOT[1] + s * dx + c * dy)


def _head_uv():
    c, s = np.cos(HEAD_TILT), np.sin(HEAD_TILT)
    dx, dy = XX - HEAD_PIVOT[0], YY - HEAD_PIVOT[1]
    return (HEAD_PIVOT[0] + (c * dx + s * dy) / HEAD_K,
            HEAD_PIVOT[1] + (-s * dx + c * dy) / HEAD_K)


HU, HV = _head_uv()          # head units of every pixel


def hgauss(cx, cy, sx, sy):
    """A Gaussian in head units: it scales and turns with the head."""
    return np.exp(-0.5 * (((HU - cx) / sx) ** 2 + ((HV - cy) / sy) ** 2))


def hsum(terms):
    return sum(w * hgauss(cx, cy, sx, sy) for w, cx, cy, sx, sy in terms)


def head_relief():
    """Skull, jowl and muzzle masses, brow and the stop in front of the eye (units)."""
    return sum(HEAD_K * a * np.clip(1 - ((HU - cx) ** 2 + (HV - cy) ** 2) / r ** 2, 0, 1) ** 2
               for cx, cy, r, a in HG["relief"])


_hp = [head_xy(*q) for q in HG["pts"]]
HEAD = spline(_hp)
HEAD_KEYS = {k: int(np.argmin(np.hypot(*(HEAD[:-1] - _hp[i]).T)))
             for k, i in HG["keys"].items()}
EAR = spline([head_xy(*q) for q in HG["ear"]])
TOOTH = HG["tooth"]                          # upper incisors, head units
THROAT = spline([(530, 262), (600, 262), (680, 252), (712, 268), (703, 288), (694, 312),
                 (688, 338), (686, 362), (690, 396), (650, 425), (590, 402), (545, 340),
                 (520, 292)])           # under the jaw, down into the chest; design units
CLAW_BASE = [(774, 682), (783, 687), (792, 685), (800, 677)]   # fore claws, near paw

PARTS = [
    # name, outline, line field, line pitch
    ("tail",    TAIL,    "dir",   11.0),
    ("arm_far", ARM_FAR, "dir",   10.0),
    ("body",    BODY,    "rows",  None),
    ("foot",    FOOT,    "cols",  None),
    ("arm",     ARM,     "dir",   10.0),
    ("throat",  THROAT,  "dir",   10.0),
    ("head",    HEAD,    "stream", None),
    ("ear",     EAR,     "dir",   6.0),
]
N_LINES = {"body": 44, "foot": 8, "head": 22}
SPAN    = {"body": 340.0}                  # narrowest line span: no seam at the nape
FLOW    = {"body": (0, 1, 0.38), "head": (-1, 0.3, 0.0), "tail": (-1, 0, 0.30),
           "throat": (0, 1, 0.38),
           "arm": (0.4, 1, 0.3), "arm_far": (0.4, 1, 0.3)}   # hair flow, pale tips
FEATHER = {"head": 38.0, "arm": 26.0, "arm_far": 20.0, "foot": 30.0, "ear": 10.0,
           "body": 18.0, "throat": 30.0}                     # join width, units
HAIR    = {"throat": (-0.15, 1.0),   # hair direction where it is not the part's long axis;
           "ear": (0.974, -0.229)}   # the ear's runs along its base. Changing it shifts
                                     # the shared noise and with it the body's fur breaks


def body_relief():
    """Thigh, shoulder, chest and belly masses plus fur clumps on the body's dome."""
    return (bump(445, 1110, 190, 70) + bump(715, 1225, 60, 24)      # thigh, knee
            + bump(395, 400, 130, 34) + bump(655, 520, 95, 22)      # shoulder, chest
            + bump(670, 950, 170, 30) + smooth_noise(28, 7.0))      # belly, clumps


def albedo(name):
    """Coat pattern: dark crown and nape, pale muzzle and chin, darker back, black
    tail tip, dark paws."""
    if name == "head":
        return hsum(HG["albedo"])
    if name == "body":
        return 0.18 * gauss(320, 650, 100, 500) - 0.15 * gauss(700, 850, 70, 400) \
            + 0.22 * gauss(580, 390, 100, 70) + 0.16 * gauss(320, 1210, 90, 100) \
            + 0.60 * gauss(430, 285, 55, 100)
    if name == "tail":
        return 0.15 + 0.65 * gauss(45, 1265, 100, 70)
    if name in ("arm", "arm_far"):
        return 0.05 + 0.25 * gauss(786, 640, 30, 35)
    if name == "foot":
        return 0.12 + 0.30 * gauss(755, 1328, 35, 25)
    if name == "ear":       # dark inside, a paler furred rim
        return 0.12 + 0.55 * hgauss(*HG["ear_c"], 12, 11) - 0.35 * ear_rim()
    if name == "throat":
        return -0.08 - 0.12 * smoothstep(600, 680, XX)
    return 0.0


def ear_rim():
    """1 on the ear's upper rim, fading 7 units in."""
    d = ndimage.distance_transform_edt(fill(EAR)) / S
    return (smoothstep(7.0, 2.0, d) * fill(EAR)
            * smoothstep(HG["ear_base"][1], HG["ear_rim"], HV))


def hexrgb(h: str) -> np.ndarray:
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)]) / 255.0


def mix(a, b, w):
    """a -> b by weight w (NY, NX); a and b are RGB triples or (NY, NX, 3) fields."""
    w = np.clip(w, 0, 1)[..., None]
    return a + (np.asarray(b) - a) * w


_CNOISE = None


def coat_noise():
    """Large, soft blotches for the coat colour (its own generator: RNG untouched)."""
    global _CNOISE
    if _CNOISE is None:
        n = ndimage.gaussian_filter(np.random.default_rng(SEED + 1).standard_normal((NY, NX)),
                                    34 * S)
        _CNOISE = n / (n.std() + 1e-12)
    return _CNOISE


def coat(name, mask):
    """Local coat colour of a part (NY, NX, 3), sRGB in [0, 1], before shading."""
    C = {k: hexrgb(v) for k, v in COAT.items()}
    base = np.ones((NY, NX, 1))
    nz = coat_noise()
    if name == "body":
        lo, hi = extents(mask, 1)
        rel = np.clip((XX - lo) / np.maximum(hi - lo, 1e-6), 0, 1) + 0.06 * nz
        c = mix(C["back"] * base, C["flank"], smoothstep(0.34, 0.80, rel))
        c = mix(c, C["back_dark"], 0.85 * gauss(430, 285, 60, 110)       # nape
                + 0.9 * gauss(470, 215, 45, 45)                           # under the head
                + 0.35 * gauss(300, 700, 60, 420))                        # spine
        c = mix(c, C["throat"], 0.75 * gauss(640, 385, 70, 55))
        return mix(c, C["thigh"], 0.45 * gauss(430, 1120, 150, 130))
    if name == "head":      # grey head, crown darkest; pale only on snout, lips, chin
        c = mix(C["cheek"] * base, C["back_dark"], hsum(HG["coat"]["back_dark"]))
        c = mix(c, C["crown"], hsum(HG["coat"]["crown"]))           # eye on its rim
        c = mix(c, C["throat"], hsum(HG["coat"]["throat"]))
        return mix(c, C["muzzle"], hsum(HG["coat"]["muzzle"]))
    if name in ("arm", "arm_far"):
        return mix(C["limb"] * base, C["paw"], 1.2 * gauss(786, 650, 30, 38))
    if name == "foot":
        return mix(C["foot"] * base, C["paw"], gauss(755, 1328, 35, 25))
    if name == "tail":
        return mix(C["tail"] * base, C["tail_tip"], smoothstep(215, 120, XX + 12 * nz))
    if name == "ear":
        c = mix(C["ear"] * base, C["crown"], 0.7 * hgauss(*HG["ear_c"], 12, 11))
        return mix(c, C["throat"], 0.45 * ear_rim())
    if name == "throat":    # pale buff under the chin, the body's coat behind the jaw
        return mix(coat("body", fill(BODY)), C["throat"], smoothstep(590, 670, XX + 8 * nz))
    return C["back"] * base


def attach(name):
    """1 where a part grows out of the part behind it, 0 where it overhangs it."""
    if name == "head":      # crisp at the chin, the jowl's fur merging into the throat
        img = Image.new("L", (NX, NY), 255)  # behind it; grown out of the neck elsewhere
        jaw = HEAD[HEAD_KEYS["chin"]:HEAD_KEYS["jaw"] + 1]
        ImageDraw.Draw(img).line([tuple(q * S) for q in jaw], fill=0, width=2)
        d = ndimage.distance_transform_edt(np.asarray(img) > 127) / S
        cx, cy = head_xy(*HG["pts"][HG["keys"]["chin"]])
        fade = 0.35 + 0.65 * smoothstep(110.0, 30.0, np.hypot(XX - cx, YY - cy))
        return 1 - 0.7 * smoothstep(28.0, 8.0, d) * fade
    if name in ("arm", "arm_far"):
        return 1 - smoothstep(480, 560, YY)
    if name == "foot":
        return 1 - smoothstep(645, 705, XX)
    if name == "ear":
        return smoothstep(*HG["ear_base"], HV)
    if name == "body":
        return 0.5 * smoothstep(1100, 1160, YY) * (1 - smoothstep(300, 360, XX))
    if name == "throat":
        return np.ones_like(XX)
    return np.zeros_like(XX)


def draw_marmot(invert: bool) -> dict:
    """Layers of the marmot on the design grid, each composited over the parts with
    the same feathered alphas as the engraving.

    ``ink`` is the engraving's coverage, 0 = paper, 1 = ink; ``invert`` draws the
    white-line version: lines where light falls, dark marks (eye, nose, mouth) left as
    paper, claws cut as lines. The colour layers do not depend on ``invert``:

    cover        coverage of the coat (the silhouette)
    coat         local coat colour, (NY, NX, 3) sRGB in [0, 1]
    shade        form shading, 0 lit .. 1 core shadow, cast shadows included
    cast         the cast-shadow part of ``shade``
    pale, dark   grizzle: pale-tipped and dark guard hairs
    marks        eye, nose, mouth and toes; ``glint`` their catch lights
    claws        fore and hind claws: ink, or light lines in the white-line version
    tooth        the upper incisors' enamel (paper in the engraving, outlined)
    whisk, ground  whiskers and the cast shadow on the ground
    """
    global RNG
    RNG = np.random.default_rng(SEED)
    masks = {n: fill(p) for n, p, *_ in PARTS}
    names = [n for n, *_ in PARTS]
    vis = -np.ones((NY, NX), int)
    back = np.zeros((NY, NX), bool)
    w_att = {}
    back_of = {}
    for i, n in enumerate(names):
        back_of[n] = back.copy()
        w_att[n] = attach(n) * ndimage.gaussian_filter(back.astype(float), 4 * S)
        back |= masks[n]
        vis[masks[n]] = i
    union = back
    free = {n: masks[n] * (1 - w_att[n]) for n in names}

    ink = np.zeros((NY, NX))
    lay = {k: np.zeros((NY, NX)) for k in ("cover", "shade", "cast", "pale", "dark",
                                           "marks", "claws", "tooth", "glint", "whisk",
                                           "ground")}
    lay["coat"] = np.zeros((NY, NX, 3))
    jitter = smooth_noise(14, 0.12)
    for i, (n, poly, mode, pitch) in enumerate(PARTS):
        if n == "throat":       # its noise leaves the shared stream where it was
            rng_state = RNG.bit_generator.state
        m = masks[n]
        relief = body_relief() if n == "body" else head_relief() if n == "head" else 0.0
        joined = (back_of[n] & (attach(n) > 0.6)) if n in ("head", "throat") else None
        h = inflate(m, joined=joined) + np.where(m, relief, 0.0)
        shade = 0.05 + 0.85 * (1 - lambert(h)) ** 1.4
        tone = shade + albedo(n)
        cast = np.zeros((NY, NX))
        if i + 1 < len(names):     # only overhanging edges cast a shadow
            front = np.max([free[o] for o in names[i + 1:]], axis=0)
            sh = ndimage.shift(front, (10 * S, -8 * S), order=1)
            cast = 0.32 * ndimage.gaussian_filter(sh, 9 * S)
            tone += cast
        tone = np.clip(tone, 0, 1)

        if mode == "rows":
            phi, psi, hair = N_LINES[n] * wrap_coord(m, 1, SPAN.get(n, 0.0)), YY, (0, 1)
            if n == "body":     # fur lines swing round the thigh and the belly
                phi = phi + 0.5 * relief / 11.0 + smooth_noise(32, 0.18)
        elif mode == "cols":
            phi, psi, hair = N_LINES[n] * wrap_coord(m, 0, SPAN.get(n, 0.0)), XX, (1, 0)
        elif mode == "stream":  # lines from the nose, swung round skull, jowl and muzzle
            phi = stream_coord(m, poly, HEAD_KEYS, N_LINES[n]) + 0.5 * relief / 9.0
            psi = np.hypot(XX - poly[HEAD_KEYS["nose"], 0], YY - poly[HEAD_KEYS["nose"], 1])
            hair = (1, 0)       # the shared noise below keeps the other parts' draws
        else:   # dir: lines along the part's principal axis, bent by height
            _, _, vt = np.linalg.svd(poly - poly.mean(axis=0), full_matrices=False)
            hair = np.asarray(HAIR[n], float) / np.hypot(*HAIR[n]) if n in HAIR else vt[0]
            b = np.array([-hair[1], hair[0]])
            phi = (XX * b[0] + YY * b[1] + 0.35 * h) / pitch
            psi = XX * hair[0] + YY * hair[1]
        st = streaks(hair, 10.0, 1.2)
        if mode == "stream":
            st = field_streaks(phi, psi)
        tone = tone + 0.12 * st
        if n == "body":         # clumps and partings in the coat
            tone = tone + 0.06 * streaks(hair, 42.0, 7.0)
        tone = np.clip(tone, 0, 1)
        lines = 0.9 * (1 - tone) if invert else tone
        phi = phi + jitter
        cov = engrave(phi, psi, lines, dash=90.0, salt=i)
        d_in = ndimage.distance_transform_edt(m) / S
        pale = dark = np.zeros((NY, NX))
        if n in FLOW:
            pale, dark = flicks(m & (d_in > 3), phi, tone, FLOW[n][:2], seed=20 + i,
                                pale_p=FLOW[n][2])
            cov = np.maximum(cov, pale) * (1 - dark) if invert \
                else np.maximum(cov * (1 - pale), dark)
        # feathered compositing: attached edges fade into the part behind; across a
        # join each pixel takes one part's line or the other's, decided by streaks
        # along the hair, so the two furs interlock instead of greying out
        alpha = smoothstep(0.0, 1.0, d_in / (1.2 + (FEATHER.get(n, 1.2) - 1.2) * w_att[n]))
        eps = 0.08
        u = streaks(hair, 16.0, 2.2)
        if mode == "stream":
            u = field_streaks(phi, psi, 16.0, 0.25, seed=SEED + 4)
        u = ndtr(u)
        hard = np.clip(((1 + 2 * eps) * alpha - eps - u) / (2 * eps) + 0.5, 0, 1)
        wd = np.clip(3 * w_att[n], 0, 1)
        alpha = wd * hard + (1 - wd) * alpha
        if n == "head":         # where nothing lies behind, the head's edge stays whole
            alpha = np.maximum(alpha, smoothstep(0.0, 1.0, d_in / 1.2) * ~back_of[n])
        ink = ink * (1 - alpha) + cov * alpha
        for k, v in (("cover", 1.0), ("shade", np.clip(shade + cast, 0, 1)),
                     ("cast", cast), ("pale", pale), ("dark", dark)):
            lay[k] = lay[k] * (1 - alpha) + v * alpha
        lay["coat"] = mix(lay["coat"], coat(n, m), alpha)
        if n == "throat":
            RNG.bit_generator.state = rng_state

    # silhouette, heavier on the shadow side; inner contours on overhanging edges only
    din = ndimage.distance_transform_edt(union) / S
    side = np.clip(0.5 - 0.5 * (XX - 520) / 260, 0, 1) + 0.3 * np.clip((YY - 700) / 600, 0, 1)
    wout = np.full_like(XX, 1.0) if invert else 0.9 + 2.2 * side
    brk = smooth_noise(3, 1.0) > -0.9
    ink = np.maximum(ink, np.where(union, np.clip(wout - din, 0, 1), 0) * brk)
    for n in ("head", "arm", "foot", "ear"):
        d_e = ndimage.distance_transform_edt(masks[n])
        w = (0.9 if invert else 1.3) * S * (1 - w_att[n]) ** 2
        e = np.where(masks[n] & (vis == names.index(n)), np.clip(w - d_e + 0.5, 0, 1), 0)
        ink = np.maximum(ink, e * (smooth_noise(4, 1.0) > -0.3))

    # features: eye, nose, mouth, whiskers, claws, toes
    def mark(cov, key="marks"):
        nonlocal ink
        lay[key] = np.maximum(lay[key], cov)
        if invert and key == "claws":   # white-line: claws cut as lines, like whiskers
            ink = np.maximum(ink, 0.85 * cov)
        else:
            ink = ink * (1 - cov) if invert else np.maximum(ink, cov)

    def glint(mask):
        ink[mask] = 1.0 if invert else 0.0
        lay["glint"][mask] = 1.0

    ex, ey, erx, ery, erot = HG["eye"]
    def almond(grow, n):    # the eye, its corners drawn out a little
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        rx, ry = erx + grow, ery + grow
        return [head_xy(ex + rx * np.cos(a) * (1 + 0.10 * np.cos(2 * a)),
                        ey + ry * np.sin(a) * (1 - 0.12 * np.cos(2 * a)) + rx * np.cos(a) * erot)
                for a in t]
    lid = stroke(almond(2.6, 16), 2.2, closed=True)   # a hairline of paper round the eye
    ink = np.maximum(ink, 0.8 * lid) if invert else ink * (1 - 0.85 * lid)
    mark(fill(np.asarray(almond(0, 48))).astype(float))
    glint(fill(ellipse(*head_xy(ex + 0.34 * erx, ey - 0.30 * ery), 0.20 * erx * HEAD_K,
                       0.17 * erx * HEAD_K)))
    # the rhinarium: a naked, matt dark-grey pad with a black comma-shaped nostril
    pad = ndimage.gaussian_filter(fill(spline([head_xy(*q) for q in HG["nose"]])
                                       ).astype(float), 0.5 * S)
    slit = stroke([head_xy(*q) for q in HG["nostril"]], 5.2, 1.4)
    if invert:              # a faint light tone on the pad, the nostril left dark
        ink = ink * (1 - pad) + 0.22 * pad * (1 - slit)
    else:
        ink = np.maximum(ink * (1 - pad) + 0.78 * pad, slit)
    lay["marks"] = np.maximum(lay["marks"], np.maximum(0.78 * pad, slit))
    glint(fill(ellipse(*head_xy(*HG["nose_glint"]), 2.6 * HEAD_K, 1.6 * HEAD_K,
                       rot=0.9 + HEAD_TILT)))
    mark(stroke([head_xy(*q) for q in HG["cleft"]], 2.0, 1.0))            # cleft lip
    mark(stroke([head_xy(*q) for q in HG["gape"]], 2.4, 1.0))             # gape
    for pts in HG["whiskers"]:
        w = 0.9 * stroke([head_xy(*p) for p in pts], 1.6, 0.4)
        lay["whisk"] = np.maximum(lay["whisk"], w)
        ink = np.maximum(ink, w)
    # upper incisors under the cleft lip: orange enamel, paper in the engraving
    tooth = [head_xy(*q) for q in TOOTH]
    lay["tooth"] = ndimage.gaussian_filter(fill(spline(tooth)).astype(float), 0.6 * S)
    ink = np.maximum(ink, lay["tooth"]) if invert else ink * (1 - lay["tooth"])
    mark(stroke(tooth, 1.0))                 # its sides and tip; the lip covers its root
    mark(stroke([head_xy(*q) for q in HG["lip"]], 1.8, 1.0))
    for off in (PAW_FAR, (0.0, 0.0)):       # four long digging claws, the thumb a nail
        for x, y in CLAW_BASE:
            mark(stroke([(x + off[0], y + off[1] - 8), (x + off[0] + 3, y + off[1] + 10),
                         (x + off[0] - 6, y + off[1] + 26)], 4.4, 0.8), "claws")
    for pts in ([(749, 1317), (751, 1327), (755, 1338)],       # clefts between the toes
                [(766, 1325), (768, 1332), (771, 1340)]):
        mark(stroke(pts, 2.2, 0.8))
    for pts in ([(750, 1337), (760, 1340), (768, 1347)],       # short hind claws, grounded
                [(762, 1336), (773, 1340), (781, 1347)],
                [(772, 1331), (784, 1336), (792, 1345)]):
        mark(stroke(pts, 3.6, 0.8), "claws")

    # fur: short hairs across silhouettes and overhanging edges, combed toward the tail
    for i_n, (n, poly, *_) in enumerate(PARTS):
        if n in ("foot", "ear"):
            continue
        rng = np.random.default_rng([11, i_n])
        off = PAW_FAR if n == "arm_far" else np.zeros(2)
        claw_xy = np.asarray(CLAW_BASE, float) + off
        others = np.zeros((NY, NX), bool)
        for o, *_ in PARTS[i_n + 1:]:
            others |= ndimage.binary_erosion(masks[o], iterations=int(3 * S))
        seg = np.diff(np.vstack([poly, poly[:1]]), axis=0)
        cum = np.r_[0, np.cumsum(np.hypot(*seg.T))]
        tl = n == "tail"            # the tail is a brush: denser, finer, longer hairs
        for s0 in np.arange(0, cum[-1], 4.5 if tl else 7.0):
            j = min(np.searchsorted(cum, s0, side="right") - 1, len(poly) - 1)
            x, y = poly[j]
            ix, iy = int(x * S), int(y * S)
            if not (0 <= ix < NX and 0 <= iy < NY) or others[iy, ix] \
                    or w_att[n][iy, ix] > 0.4 or y > 1312:
                continue
            tx, ty = seg[j] / (np.hypot(*seg[j]) + 1e-9)
            if ty < 0 or (abs(ty) < 0.2 and tx > 0):
                tx, ty = -tx, -ty
            nx_, ny_ = ty, -tx
            if masks[n][min(NY - 1, max(0, int((y + 4 * ny_) * S))),
                        min(NX - 1, max(0, int((x + 4 * nx_) * S)))]:
                nx_, ny_ = -nx_, -ny_
            L = rng.uniform(7, 17) * (1.4 + 1.0 * smoothstep(170, 50, x) if tl
                                      else 0.6 if n == "head" else 1.0)   # short face fur
            a = rng.uniform(0.15, 0.45) if tl else rng.uniform(0.25, 0.6)
            if n in ("arm", "arm_far") and np.hypot(*(claw_xy - (x, y)).T).min() < 22:
                continue            # the digits are short-furred: leave the claws clear
            if n == "head" and min(np.hypot(*(np.subtract(head_xy(*q), (x, y))))
                                   for q in ((780, 243), (735, 297))) < 16:
                continue            # bare nose pad, and the incisors under the lip
            if tl:                  # tail hairs stream toward the tip, fanning outward
                dx, dy = -1.0 + (1.5 * a + 0.1) * nx_, (1.5 * a + 0.1) * ny_
                dx, dy = dx / np.hypot(dx, dy), dy / np.hypot(dx, dy)
                tx, ty = (-tx, -ty) if tx > 0 else (tx, ty)     # bow toward the tip
            else:
                dx, dy = tx * np.cos(a) + nx_ * np.sin(a), ty * np.cos(a) + ny_ * np.sin(a)
            p0 = (x - 5 * dx, y - 5 * dy)
            p2 = (x + L * dx, y + L * dy)
            bow = 3.5 if tl else 1.5
            p1 = ((p0[0] + p2[0]) / 2 + bow * tx, (p0[1] + p2[1]) / 2 + bow * ty)
            wh = (1.6 if tl else 2.2) if invert else (1.9 if tl else 2.6)
            ink = np.maximum(ink, stroke([p0, p1, p2], wh, 0.3, n=40))

    # ground: a cast shadow of horizontal lines, darkest under the feet
    r2 = ((XX - 500) / 360) ** 2 + ((YY - 1342) / 20) ** 2
    t = np.clip((0.35 if invert else 0.75) * (1 - r2), 0, 1) * (~union)
    lay["ground"] = engrave(YY / 6.5 + 0.3, XX, t, dash=140.0, salt=99) * (r2 < 1)
    ink = np.maximum(ink, lay["ground"])
    lay["ink"] = np.clip(ink, 0, 1)
    return lay


# =============================================================================
# Colour: a hand-coloured plate
# =============================================================================
def over(dst, src):
    """``src`` over ``dst``, both premultiplied RGBA."""
    return src + dst * (1 - src[..., 3:4])


def layer(rgb, a):
    """Premultiplied RGBA of colour ``rgb`` (hex, triple or field) at coverage ``a``."""
    a = np.clip(a, 0, 1)
    rgb = hexrgb(rgb) if isinstance(rgb, str) else rgb
    rgb = np.broadcast_to(np.asarray(rgb, float), a.shape + (3,))
    return np.dstack([rgb * a[..., None], a])


def tooth_tip(lay: dict) -> np.ndarray:
    """The incisors' paler cutting edge: the lowest third of the tooth (head units)."""
    return lay["tooth"] * smoothstep(304.0, 313.0, HV)


def paint(lay: dict, mode: str) -> np.ndarray:
    """The coloured marmot as premultiplied RGBA on the design grid.

    light: translucent washes of the coat colour under the dark engraving, warm on the
    lit side and cool in the shadows, pooled at their edges like watercolour; the pale
    guard-hair tips laid over the wash; the cast shadow on the ground cool.
    dark: the same washes as an opaque, darker underpainting, with the white-line
    engraving printed over it in light tints of the coat.
    """
    r = np.random.default_rng(SEED + 2)

    def blot(sig):
        n = ndimage.gaussian_filter(r.standard_normal((NY, NX)), sig * S)
        return n / (n.std() + 1e-12)

    inside = lay["cover"] > 0.5
    _, (iy, ix) = ndimage.distance_transform_edt(~inside, return_indices=True)
    coat_, shade = lay["coat"][iy, ix], lay["shade"][iy, ix]   # reach the fringe hairs
    s = smoothstep(0.15, 0.95, shade)
    sd = (ndimage.distance_transform_edt(inside) - ndimage.distance_transform_edt(~inside)) / S
    edge = sd - WASH_INSET[mode] + 2.2 * blot(8)               # a hand-laid wash edge
    wash = smoothstep(0.0, 2.0, edge)
    pool = 1 + 0.30 * np.exp(-np.maximum(edge, 0) / 5)          # pigment at the edge
    grain = 1 + 0.10 * blot(40) + 0.06 * blot(12)
    r2 = ((XX - 500) / 380) ** 2 + ((YY - 1342) / 22) ** 2
    floor = np.clip(1 - r2, 0, 1) ** 1.5 * (1 - lay["cover"])
    line = np.clip(lay["ink"] - lay["ground"], 0, 1)
    out = np.zeros((NY, NX, 4))
    if mode == "light":
        c = mix(coat_, coat_ * hexrgb(TINT["cool"]), 0.6 * s)
        c = mix(c, hexrgb(TINT["warm"]), 0.16 * (1 - s))
        out = over(out, layer(TINT["floor"], 0.30 * floor))
        out = over(out, layer(c, WASH_ALPHA * wash * pool * grain))
        out = over(out, layer(TINT["grizzle"], 0.85 * lay["pale"]))
        out = over(out, layer(COAT["incisor"], 0.95 * lay["tooth"]))
        out = over(out, layer(COAT["incisor_tip"], 0.9 * tooth_tip(lay)))
        out = over(out, layer(TINT["ground"], lay["ground"]))
        out = over(out, layer(LINE["light"], line))
        return over(out, layer(TINT["glint"], lay["glint"]))
    v = (0.42 + 0.40 * (1 - s))[..., None]
    c = mix(coat_ * v, coat_ * v * hexrgb(TINT["cool"]) * 1.1, 0.6 * s)
    # light lines keep the coat's hue and its dark zones; the silhouette and its fringe
    # hairs catch a pale rim so the dark cap and tail tip hold against the background
    gain = np.minimum(1.30 + 0.45 * (1 - s), 0.84 / coat_.max(axis=2))   # keeps the hue
    lc = mix(coat_ * gain[..., None], hexrgb(LINE["dark"]), 0.25)
    lum = coat_ @ np.array([0.30, 0.59, 0.11])
    lc = mix(lc, hexrgb(TINT["rim"]),
             0.75 * (0.45 + 0.55 * smoothstep(0.10, 0.50, lum)) * smoothstep(2.5, 0.0, sd))
    out = over(out, layer(TINT["floor_dark"], 0.35 * floor))
    out = over(out, layer(c, 0.96 * wash * (1 + 0.05 * blot(40))))
    out = over(out, layer(TINT["ground_dark"], 0.8 * lay["ground"]))
    out = over(out, layer(lc, 0.92 * line))
    out = over(out, layer(COAT["incisor"], 0.95 * lay["tooth"]))
    out = over(out, layer(COAT["incisor_tip"], 0.9 * tooth_tip(lay)))
    out = over(out, layer(TINT["claw_dark"], lay["claws"]))
    out = over(out, layer(TINT["eye"], lay["marks"]))
    return over(out, layer(LINE["dark"], lay["glint"]))


# =============================================================================
# Lettering and layout
# =============================================================================
def font(spec, size: int):
    """``spec`` = (variable font file, weight); other axes stay at their default."""
    path, wght = spec
    f = ImageFont.truetype(str(path), size)
    f.set_variation_by_axes([wght if a["name"] in (b"Weight", "Weight") else a["default"]
                             for a in f.get_variation_axes()])
    return f


def text_alpha(txt: str, spec, cap_px: float):
    """Coverage of ``txt`` in font ``spec`` with capitals ``cap_px`` tall, and its
    baseline row."""
    probe = font(spec, 1000)
    f = font(spec, int(round(1000 * cap_px / -probe.getbbox("H", anchor="ls")[1])))
    l, t, r, b = f.getbbox(txt, anchor="ls")
    img = Image.new("L", (r - l + 4, b - t + 4), 0)
    ImageDraw.Draw(img).text((2 - l, 2 - t), txt, font=f, fill=255, anchor="ls")
    return np.asarray(img) / 255.0, 2 - t


def layout(m: np.ndarray, sole: int, ink: str) -> np.ndarray:
    """Marmot left (premultiplied RGBA); the word right, in colour ``ink``; the tagline
    under the word, set to the word's width, its baseline on the marmot's soles (row
    ``sole``)."""
    hm, wm = m.shape[:2]
    ta, base = text_alpha(WORD, WORD_FONT, CAP_FRAC * hm)
    probe, _ = text_alpha(TAGLINE, TAG_FONT, 100.0)
    ga, gb = text_alpha(TAGLINE, TAG_FONT,
                        100.0 * (ta.shape[1] - 4) / (probe.shape[1] - 4))
    cap = int(CAP_FRAC * hm)
    gap = int(TAG_GAP * cap)
    yt = sole                                           # tagline baseline
    yb = yt - gap - gb                                  # word baseline
    x0 = wm + int(GAP_FRAC * hm)
    out = np.zeros((max(hm, yt + ga.shape[0] - gb) + 4,
                    x0 + max(ta.shape[1], ga.shape[1]) + 16, 4))
    out[:hm, :wm] = m
    for a, y in ((ta, yb - base), (ga, yt - gb)):
        sl = (slice(y, y + a.shape[0]), slice(x0, x0 + a.shape[1]))
        out[sl] = over(out[sl], layer(ink, a))
    return out


def resize(p: np.ndarray, width: int) -> np.ndarray:
    """Lanczos-resize premultiplied RGBA to ``width`` px."""
    size = (width, int(round(p.shape[0] * width / p.shape[1])))
    return np.clip(np.dstack([np.asarray(Image.fromarray(p[..., k].astype(np.float32), "F")
                                         .resize(size, Image.LANCZOS)) for k in range(4)]),
                   0, 1)


def write_png(p: np.ndarray, path: Path) -> None:
    """Straight-alpha RGBA PNG, ``WIDTH_PX`` wide, from premultiplied RGBA."""
    q = resize(p, WIDTH_PX)
    a = q[..., 3:4]
    rgb = np.where(a > 1e-6, q[..., :3] / np.maximum(a, 1e-6), 0)
    Image.fromarray((255 * np.clip(np.dstack([rgb, a]), 0, 1)).round().astype(np.uint8),
                    "RGBA").save(path, optimize=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="figs/logo", help="output directory")
    args = ap.parse_args()
    odir = Path(args.out)
    odir.mkdir(parents=True, exist_ok=True)

    lays = {"light": draw_marmot(invert=False), "dark": draw_marmot(invert=True)}
    ys, xs = np.nonzero((lays["light"]["ink"] > 0.02) | (lays["dark"]["ink"] > 0.02))
    box = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    sole = int(round(FOOT[:, 1].max() * S)) - ys.min()
    for mode, lay in lays.items():
        write_png(layout(paint(lay, mode)[box], sole, TEXT_INK[mode]),
                  odir / f"pysuto-logo-{mode}.png")
        write_png(layout(layer(BW_INK[mode], lay["ink"])[box], sole, BW_INK[mode]),
                  odir / f"pysuto-logo-bw-{mode}.png")
    print(f"marmot {xs.max() - xs.min() + 1} x {ys.max() - ys.min() + 1} px -> "
          f"{odir}/pysuto-logo-{{,bw-}}{{light,dark}}.png, {WIDTH_PX} px wide")


if __name__ == "__main__":
    main()
