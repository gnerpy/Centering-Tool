"""Card centering measurement.

Takes a flatbed scan holding one or more trading cards and, for each card,
locates two rectangles: the physical cut edge and the printed frame just
inside it.  The ratio of opposing border widths is what PSA prints on the
flip as "55/45".

Nothing here is clever about card designs.  The frame colour is learned from
a thin ring just inside each cut edge of the card being measured, so a yellow
Pokemon border, a silver holo frame and a 1999 white border are all handled
the same way; a full-art card, where that ring is not one flat colour, is
reported as "no frame found" rather than guessed at.
"""

from __future__ import annotations

import math
import os

import numpy as np
from PIL import Image, ImageFilter

Image.MAX_IMAGE_PIXELS = None

# Trading card stock, in millimetres.  Every card in the hobby is this size.
CARD_SHORT_MM = 63.0
CARD_LONG_MM = 88.0

DEFAULT_DPI = 600
SEGMENT_LONG_SIDE = 1600  # segmentation runs on a downscaled copy
SIDES = ("left", "right", "top", "bottom")
CORNERS = (("left", "top"), ("right", "top"), ("left", "bottom"), ("right", "bottom"))

# Condition heuristics, all approximate.  A flatbed scan is flat, top-down
# light -- it cannot see what a grader sees under a raking light, so these
# only catch damage that shows as a plain colour shift or a missing patch of
# card stock.  Every hit is drawn on the image for a human to judge, the same
# way an unreadable frame is left for a human to nudge.
DEFAULT_BORDER_MM = 2.4         # fallback border width when no frame was traced
GUESS_BLUR_MM = 0.4             # quiets holo/foil speckle enough to read a rough position
GUESS_RESID_FACTOR = 3.0        # looser than trace_card's own straight-line bar
GUESS_MIN_INLIER = 0.5
CORNER_BOX_MM = 2.5             # size of the corner test square
CORNER_FLAG_SEVERITY = 0.35
WHITENING_DEPTH_MM = 4.0        # how far in from the cut edge to sample
EDGE_SEGMENT_MM = 3.0           # length of each edge sample along the border
WHITENING_FLAG = 0.30           # fraction of the way from border colour to white
SURFACE_MARGIN_MM = 2.0         # kept clear of the frame, where the trace is weakest
SURFACE_BLUR_MM = 16.0          # wider than any normal print detail or holo texture
SURFACE_GRID_MM = 2.0
SURFACE_DEVIATION_FLAG = 95.0
SURFACE_MIN_AREA_MM2 = 20.0
SURFACE_MAX_REPORT = 4


# --------------------------------------------------------------------------
# small numeric helpers
# --------------------------------------------------------------------------

def robust_line(points, iters: int = 4):
    """Least squares line through (t, v) samples, re-fit without outliers.

    Returns (slope, intercept, mean_abs_residual, inlier_fraction) or None.
    """
    if points is None or len(points) < 8:
        return None
    pts = np.asarray(points, float)
    t, v = pts[:, 0], pts[:, 1]
    m, c = np.polyfit(t, v, 1)
    keep = np.ones(len(t), bool)
    for _ in range(iters):
        resid = v - (m * t + c)
        spread = max(float(resid[keep].std()), 0.8)
        keep = np.abs(resid) < 2.5 * spread
        if keep.sum() < 8:
            break
        m, c = np.polyfit(t[keep], v[keep], 1)
    resid = np.abs(v - (m * t + c))
    return float(m), float(c), float(resid[keep].mean()), float(keep.mean())


def _refine_rect_side(rgb, side: str, pos: float, lo: int, hi: int, ppmm: float,
                       band_mm: float = 3.0, step_mm: float = 3.0):
    """Precisely locate one side of a roughly-known rectangle.

    A contour-detected quad gives a good approximate position but not a
    trustworthy angle -- a handful of imprecise corners can tilt the whole
    thing by a degree or so. This instead samples many points along the side,
    each independently finding the strongest local contrast step within a
    narrow band around the rough position, and fits a robust line through
    them -- the same outlier-rejecting fit used for every other edge in this
    file, just seeded from a rough position instead of a background search.
    """
    h, w = rgb.shape[:2]
    band = max(6, int(round(band_mm * ppmm)))
    step = max(4, int(round(step_mm * ppmm)))
    vertical = side in ("left", "right")
    limit = w if vertical else h
    lo, hi = max(0, lo), min(limit if not vertical else (h if vertical else w), hi)
    a0, b0 = max(0, int(pos) - band), min(w if vertical else h, int(pos) + band)
    if b0 - a0 < 6:
        return None

    pts = []
    for t in range(lo, hi, step):
        seg = (rgb[t, a0:b0] if vertical else rgb[a0:b0, t]).astype(np.float32)
        if seg.shape[0] < 6:
            continue
        diffs = np.abs(np.diff(seg, axis=0)).sum(axis=1)
        idx = int(np.argmax(diffs))
        pts.append((t, a0 + idx + 0.5))
    return robust_line(pts)


def luma(rgb: np.ndarray) -> np.ndarray:
    a = rgb.astype(np.float32)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


# --------------------------------------------------------------------------
# sheet loading
# --------------------------------------------------------------------------

class Sheet:
    """A decoded scan, kept around because decoding one is expensive."""

    def __init__(self, path: str):
        self.path = path
        im = Image.open(path)
        dpi = im.info.get("dpi")
        self.dpi = int(round(dpi[0])) if dpi and dpi[0] else DEFAULT_DPI
        self.image = im.convert("RGB")
        self.h, self.w = self.image.size[1], self.image.size[0]

    @property
    def px_per_mm(self) -> float:
        return self.dpi / 25.4

    def info(self) -> dict:
        return {
            "path": self.path,
            "name": os.path.basename(self.path),
            "width": self.w,
            "height": self.h,
            "dpi": self.dpi,
        }


_CACHE: dict[str, Sheet] = {}


def get_sheet(path: str) -> Sheet:
    path = os.path.abspath(path)
    sheet = _CACHE.get(path)
    if sheet is None:
        _CACHE.clear()  # one sheet at a time; these are hundreds of megabytes
        sheet = Sheet(path)
        _CACHE[path] = sheet
    return sheet


# --------------------------------------------------------------------------
# step 1 - find the cards on the sheet
# --------------------------------------------------------------------------

def _ink_mask(rgb: np.ndarray, thresh: int = 26) -> np.ndarray:
    """True where the scan is not blank platen white."""
    lo = rgb.min(axis=2).astype(np.int16)
    hi = rgb.max(axis=2).astype(np.int16)
    return ((255 - lo) > thresh) | ((hi - lo) > thresh)


def _white_mask(rgb: np.ndarray) -> np.ndarray:
    lo = rgb.min(axis=2).astype(np.int16)
    hi = rgb.max(axis=2).astype(np.int16)
    return (lo > 228) & ((hi - lo) < 20)


def _runs(profile: np.ndarray, thresh: float, min_len: int) -> list[tuple[int, int]]:
    on = profile > thresh
    out, start = [], None
    for i, v in enumerate(on):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                out.append((start, i - 1))
            start = None
    if start is not None and len(on) - start >= min_len:
        out.append((start, len(on) - 1))
    return out


def segment(sheet: Sheet) -> list[dict]:
    """Split the sheet into candidate card regions, in full-resolution pixels."""
    scale = min(1.0, SEGMENT_LONG_SIDE / max(sheet.w, sheet.h))
    small = np.asarray(sheet.image.resize(
        (max(1, int(sheet.w * scale)), max(1, int(sheet.h * scale))), Image.BILINEAR))
    mask = _ink_mask(small)
    min_short = max(4, int(CARD_SHORT_MM * sheet.px_per_mm * scale * 0.55))

    boxes = []
    for y0, y1 in _runs(mask.mean(axis=1), 0.03, min_short):
        strip = mask[y0:y1 + 1]
        for x0, x1 in _runs(strip.mean(axis=0), 0.03, min_short):
            cell = strip[:, x0:x1 + 1]
            rows = np.flatnonzero(cell.mean(axis=1) > 0.15)
            cols = np.flatnonzero(cell.mean(axis=0) > 0.15)
            if rows.size < 4 or cols.size < 4:
                continue
            boxes.append((x0 + cols[0], y0 + rows[0], x0 + cols[-1], y0 + rows[-1]))

    out = []
    for bx0, by0, bx1, by1 in boxes:
        x0, y0 = int(bx0 / scale), int(by0 / scale)
        x1, y1 = int(math.ceil(bx1 / scale)), int(math.ceil(by1 / scale))
        w_mm, h_mm = (x1 - x0) / sheet.px_per_mm, (y1 - y0) / sheet.px_per_mm
        if min(w_mm, h_mm) < CARD_SHORT_MM * 0.55 or max(w_mm, h_mm) < CARD_LONG_MM * 0.5:
            continue
        out.append({
            "index": len(out),
            "box": [x0, y0, x1, y1],
            "touchesSheet": {
                "left": x0 <= 1, "top": y0 <= 1,
                "right": x1 >= sheet.w - 2, "bottom": y1 >= sheet.h - 2,
            },
        })

    # Each card is traced inside its own crop, and the trace works by walking
    # in through blank platen -- so every crop edge has to sit on clean
    # background, not on the sleeve of the card lying next to it.
    for region in out:
        region["cropBox"] = _clean_crop(sheet, region["box"])
    return out


def _clean_crop(sheet: Sheet, box, margin_mm: float = 4.0) -> list[int]:
    """Grow the box outward as far as the background stays blank."""
    x0, y0, x1, y1 = box
    reach = int(round(margin_mm * sheet.px_per_mm))
    arr = np.asarray(sheet.image)

    def blank(line) -> bool:
        return float(_white_mask(line[:, None, :] if line.ndim == 2 else line).mean()) > 0.92

    lead = int(round(0.6 * sheet.px_per_mm))  # the box edge itself may be fuzzy

    def grow(fixed_lo, fixed_hi, start, limit, stepdir, horizontal):
        best, established = start, False
        for step in range(1, max(2, reach // 4)):
            nxt = start + stepdir * 4 * step
            if not (0 <= nxt < limit):
                break
            line = (arr[nxt, fixed_lo:fixed_hi] if horizontal
                    else arr[fixed_lo:fixed_hi, nxt])
            if blank(line):
                best, established = nxt, True
            elif established or 4 * step > lead:
                break
        return best

    top = grow(x0, x1, y0, sheet.h, -1, True)
    bottom = grow(x0, x1, y1, sheet.h, +1, True)
    left = grow(y0, y1, x0, sheet.w, -1, False)
    right = grow(y0, y1, x1, sheet.w, +1, False)
    return [max(0, left), max(0, top), min(sheet.w, right + 1), min(sheet.h, bottom + 1)]


# --------------------------------------------------------------------------
# step 2 - the cut edge
# --------------------------------------------------------------------------

def _scanlines(arr2d, side: str, step: int):
    """Yield (t, line, to_coord) with the line running from outside inward."""
    h, w = arr2d.shape[:2]
    if side == "left":
        for y in range(0, h, step):
            yield y, arr2d[y], (lambda i: i)
    elif side == "right":
        for y in range(0, h, step):
            yield y, arr2d[y][::-1], (lambda i, w=w: w - 1 - i)
    elif side == "top":
        for x in range(0, w, step):
            yield x, arr2d[:, x], (lambda i: i)
    else:
        for x in range(0, w, step):
            yield x, arr2d[:, x][::-1], (lambda i, h=h: h - 1 - i)


def _runs_along(mask: np.ndarray, side: str, ppmm: float, step: int = 8):
    """Walk in from one side; record where the masked band starts and ends.

    Returns (outer points, inner points, fraction of scanlines with no margin).
    The band is allowed short interruptions -- a sleeve seam or a print speck
    inside a border should not be mistaken for the end of it.
    """
    run = max(3, int(round(0.25 * ppmm)))
    hole = max(4, int(round(0.55 * ppmm)))
    outer, inner, no_margin, total = [], [], 0, 0
    for t, line, to_coord in _scanlines(mask, side, step):
        n = line.size
        if n < run + hole:
            continue
        total += 1
        on = line.astype(np.int32)
        solid = np.convolve(on, np.ones(run, np.int32), "valid")
        starts = np.flatnonzero(solid == run)
        if starts.size == 0:
            continue
        s = int(starts[0])
        if s == 0:
            no_margin += 1
        empty = np.convolve(1 - on, np.ones(hole, np.int32), "valid")
        ends = np.flatnonzero(empty == hole)
        ends = ends[ends > s]
        if ends.size == 0:
            continue
        outer.append((t, to_coord(s - 0.5)))
        inner.append((t, to_coord(int(ends[0]) - 0.5)))
    return outer, inner, (no_margin / total if total else 0.0)


def _trace_band(mask: np.ndarray, ppmm: float, lenient: bool = False) -> dict:
    """Outer and inner line of the masked band, on all four sides.

    A scanline with no margin before the band starts is normally a sign the
    edge ran past the platen, and is always rejected outright on a scan. A
    photo can pass lenient=True: there, no-margin is just as often one
    scanline's worth of glare, which robust_line's own outlier rejection
    already discards fine, so the fit is trusted unless it *itself* lands
    within a pixel or two of the crop's own edge -- what a genuinely absent
    margin looks like once fit rather than raw-counted, since a confident
    line through nothing but noise (every scanline clipped, as under heavy
    glare) still lands there too.
    """
    h, w = mask.shape[:2]
    out = {}
    for side in SIDES:
        o, i, no_margin = _runs_along(mask, side, ppmm)
        clipped = no_margin > 0.30
        fit = robust_line(o)
        if fit is not None and clipped:
            reject = True
            if lenient:
                at = h / 2.0 if side in ("left", "right") else w / 2.0
                pos = fit[0] * at + fit[1]
                limit = 0.0 if side in ("left", "top") else (w if side == "right" else h)
                reject = abs(pos - limit) < 2.0
            if reject:
                fit = None
        out[side] = {
            "outer": fit,
            "inner": robust_line(i),
            "clipped": clipped,
        }
    return out


def _colour_band(rgb: np.ndarray, colour: np.ndarray, tol: float) -> np.ndarray:
    return np.abs(rgb.astype(np.float32) - colour).sum(axis=2) <= tol


def _ring(rgb: np.ndarray, box, ppmm: float, near_mm: float, far_mm: float):
    """Median colour and spread of a ring at a given depth inside a box."""
    x0, y0, x1, y1 = box
    a, b = int(round(near_mm * ppmm)), int(round(far_mm * ppmm))
    if x1 - x0 < 3 * b or y1 - y0 < 3 * b:
        return None, 999.0
    strips = [rgb[y0 + a:y1 - a, x0 + a:x0 + b], rgb[y0 + a:y1 - a, x1 - b:x1 - a],
              rgb[y0 + a:y0 + b, x0 + a:x1 - a], rgb[y1 - b:y1 - a, x0 + a:x1 - a]]
    strips = [s.reshape(-1, 3) for s in strips if s.size]
    if not strips:
        return None, 999.0
    ring = np.concatenate(strips).astype(np.float32)
    colour = np.median(ring, axis=0)
    return colour, float(np.median(np.abs(ring - colour).sum(axis=1)))


def _card_band(rgb: np.ndarray, bg=None) -> np.ndarray:
    """Everything meaningfully different from the background behind the card.

    On a scan that background is the scanner platen, always near-white; on a
    photo it can be anything, so a photo caller passes its own sampled colour.
    The threshold is set well above what a clear sleeve contributes, so a
    dark full-art card still reads while the sleeve around it does not.
    """
    bg = np.array([255.0, 255.0, 255.0]) if bg is None else np.asarray(bg, dtype=np.float32)
    return np.abs(rgb.astype(np.float32) - bg).sum(axis=2) > 130.0


def _ink_box(rgb: np.ndarray):
    ink = ~_white_mask(rgb)
    rows = np.flatnonzero(ink.mean(axis=1) > 0.5)
    cols = np.flatnonzero(ink.mean(axis=0) > 0.5)
    if rows.size < 8 or cols.size < 8:
        return None
    return int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1])


def _evaluate(band: dict, side: str, shape) -> float | None:
    fit = band[side]["outer"]
    if fit is None:
        return None
    at = shape[0] / 2.0 if side in ("left", "right") else shape[1] / 2.0
    return fit[0] * at + fit[1]


def _rect_from(band: dict, shape, ppmm: float):
    """The card rectangle implied by a traced band, rebuilding missing sides."""
    pos = {s: _evaluate(band, s, shape) for s in SIDES}
    nominal = {"x": CARD_SHORT_MM * ppmm, "y": CARD_LONG_MM * ppmm}
    for a, b, axis in (("left", "right", "x"), ("top", "bottom", "y")):
        if pos[a] is None and pos[b] is None:
            return None
        if pos[a] is None:
            pos[a] = pos[b] - nominal[axis]
        elif pos[b] is None:
            pos[b] = pos[a] + nominal[axis]
    h, w = shape[0], shape[1]
    x0, x1 = max(0, int(pos["left"])), min(w, int(pos["right"]))
    y0, y1 = max(0, int(pos["top"])), min(h, int(pos["bottom"]))
    if x1 - x0 < 40 or y1 - y0 < 40:
        return None
    return x0, y0, x1, y1


def _span_error(band: dict, shape, ppmm: float) -> float:
    nominal = {"x": CARD_SHORT_MM * ppmm, "y": CARD_LONG_MM * ppmm}
    worst = 0.0
    for a, b, axis in (("left", "right", "x"), ("top", "bottom", "y")):
        pa, pb = _evaluate(band, a, shape), _evaluate(band, b, shape)
        if pa is None or pb is None:
            continue
        worst = max(worst, abs((pb - pa) - nominal[axis]))
    return worst


def trace_card(rgb: np.ndarray, ppmm: float, bg=None, lenient: bool = False,
               extra_depths: tuple = (), known_rect=None, refine_known_rect: bool = True) -> dict:
    """Find the cut edge and the print frame in a deskewed card crop.

    A card's border colour runs right up to the cut edge, so both rectangles
    come out of one band: its outside is the cut edge, its inside is the
    frame.  The border has to be sampled at a depth measured from the card
    itself -- a penny sleeve sits a millimetre outside it and would otherwise
    be sampled instead -- so the card is located first, roughly, by what
    stands out from the background behind it (the platen for a scan, a
    caller-supplied colour for a photo).  Depths are then tried from the very
    edge inward: a shallow one catches the thin printed frame of a full-art
    card, a deep one the wide border of an ordinary one.  Whichever produces
    a 63 x 88 mm rectangle is the reading that was right.  extra_depths adds
    further (near, far) candidates beyond the ones tuned for a bare card's
    own border -- a rigid toploader holds the card much further in than any
    sleeve does, so a photo of one needs depths a scan never has to try.

    known_rect skips all of that and trusts a cut edge measured some other
    way (a photo's own contour-detected quad, mapped through its perspective
    correction) -- useful exactly when a toploader sits so close to the card
    that no clean background margin is left for a colour search to find. The
    ring-sampling depth search still runs, but only to find the frame inside
    that already-known rectangle.

    refine_known_rect additionally re-measures each side with a narrow-band
    scanline search, for when known_rect is only approximate (a toploader's
    edge, found a little imprecisely). Turn it off when known_rect has no
    real margin to refine against at all (an image cropped tight to the
    card, with nothing outside it) -- refining there has nothing genuine to
    find and fits whatever internal print detail or compression noise is
    nearest instead, by a different amount on each side, which distorts the
    rectangle rather than sharpening it.
    """
    shape = rgb.shape[:2]
    straight = max(3.0, 0.22 * ppmm)
    if known_rect is None:
        base = _trace_band(_card_band(rgb, bg), ppmm, lenient)
        rect = _rect_from(base, shape, ppmm)
        if rect is None:
            return {"cut": base, "frame": None, "spread": 999.0, "kind": "unreadable"}
    else:
        rect = known_rect
        x0, y0, x1, y1 = rect
        flat = lambda pos: (0.0, pos, 0.0, 1.0)

        def refine(*a):
            if not refine_known_rect:
                return None
            fit = _refine_rect_side(*a)
            # a real margin, even a thin one, fits cleanly (low residual, high
            # inlier fraction); a side with nothing genuine to find still
            # returns *a* fit, just a noisy one -- reject those individually
            # rather than trusting or discarding all four sides together, since
            # one image can have a real, if thin, margin on some sides and
            # none at all on others.
            if fit is None or fit[2] > straight or fit[3] < 0.85:
                return None
            return fit

        base = {
            "left": {"outer": refine(rgb, "left", x0, y0, y1, ppmm) or flat(x0),
                     "inner": None, "clipped": False},
            "right": {"outer": refine(rgb, "right", x1, y0, y1, ppmm) or flat(x1),
                      "inner": None, "clipped": False},
            "top": {"outer": refine(rgb, "top", y0, x0, x1, ppmm) or flat(y0),
                    "inner": None, "clipped": False},
            "bottom": {"outer": refine(rgb, "bottom", y1, x0, x1, ppmm) or flat(y1),
                       "inner": None, "clipped": False},
        }
    best = None
    depths = ((0.15, 0.70), (0.35, 1.05), (0.70, 1.70),
              (1.30, 2.30), (2.10, 3.10)) + tuple(extra_depths)
    for near, far in depths:
        colour, spread = _ring(rgb, rect, ppmm, near, far)
        if colour is None or spread > 90.0:
            continue
        band = _trace_band(_colour_band(rgb, colour, max(48.0, 3.0 * spread)), ppmm, lenient)
        err = _span_error(band, shape, ppmm)
        if err > 0.9 * ppmm:
            continue
        good = sum(1 for s in SIDES if band[s]["inner"] is not None
                   and band[s]["inner"][2] < straight)
        if good < 2:
            continue
        score = 6.0 * good - err / ppmm - 0.03 * spread
        if best is None or score > best[0]:
            best = (score, spread, band, colour)

    if best is None:
        return {"cut": base, "frame": None, "spread": 999.0, "kind": "no-frame"}
    # traced["cut"] is what downstream reads for edge position; traced["frame"]
    # is only ever read for its inner line, so trusting the known rect for cut
    # here doesn't discard anything the frame search found.
    return {"cut": base if known_rect is not None else best[2], "frame": best[2],
            "spread": best[1], "kind": "bordered" if best[1] <= 40.0 else "full-art",
            "colour": best[3]}


def guess_frame(rgb: np.ndarray, rect: tuple[int, int, int, int], ppmm: float) -> dict:
    """A rough per-side frame position for a card trace_card gave up on.

    Never feeds a measurement, a flag or a confidence score -- only where an
    unplaced guide starts, so the bar here is deliberately looser than
    trace_card's own. A holo or full-art border is often too speckled for one
    colour and depth to satisfy every side at once (the speckle inflates the
    ring's colour spread, which widens the match tolerance, which lets the
    band bleed into the artwork); a mild blur quiets that speckle, and each
    side is scored against whichever depth suits it best rather than requiring
    one depth to win on all four at once. Where the border is not just noisy
    but genuinely absent on a side -- full art bleeding across it unevenly --
    no depth fits cleanly even after blurring, and that side is left for a
    human to place, same as trace_card would leave it.
    """
    shape = rgb.shape[:2]
    straight = max(3.0, 0.22 * ppmm)
    cap = GUESS_RESID_FACTOR * straight
    radius = max(1.0, GUESS_BLUR_MM * ppmm / 2.0)
    blurred = np.asarray(Image.fromarray(rgb).filter(ImageFilter.GaussianBlur(radius)))

    best: dict[str, tuple[float, float] | None] = {s: None for s in SIDES}
    depths = ((0.15, 0.70), (0.35, 1.05), (0.70, 1.70), (1.30, 2.30), (2.10, 3.10))
    for near, far in depths:
        colour, spread = _ring(blurred, rect, ppmm, near, far)
        if colour is None or spread > 90.0:
            continue
        band = _trace_band(_colour_band(blurred, colour, max(48.0, 3.0 * spread)), ppmm)
        for side in SIDES:
            fit = band[side]["inner"]
            if fit is None or fit[2] > cap or fit[3] < GUESS_MIN_INLIER:
                continue
            at = shape[0] / 2.0 if side in ("left", "right") else shape[1] / 2.0
            pos = fit[0] * at + fit[1]
            if best[side] is None or fit[2] < best[side][0]:
                best[side] = (fit[2], pos)
    return {s: (v[1] if v else None) for s, v in best.items()}


def _skew_from(band: dict) -> float:
    """Card rotation in the crop, in degrees, from whichever edges were found."""
    angles = []
    for side in SIDES:
        fit = band[side]["outer"] if isinstance(band[side], dict) else band[side]
        if fit is None or fit[2] > 5.0 or fit[3] < 0.6:
            continue
        a = math.atan(fit[0])
        angles.append(a if side in ("top", "bottom") else -a)
    if not angles:
        return 0.0
    return math.degrees(float(np.median(angles)))


# --------------------------------------------------------------------------
# step 3 - corner and edge condition, and gross surface anomalies
# --------------------------------------------------------------------------

def _toward_white(colour: np.ndarray):
    """A unit-ish projection axis from the border colour to white, and its norm."""
    to_white = np.array([255.0, 255.0, 255.0]) - colour
    return to_white, float(np.dot(to_white, to_white)) or 1.0


def corner_defects(rgb: np.ndarray, cut: dict, ppmm: float, colour) -> list[dict]:
    """Rounding, chipping or whitening at each of the four corners.

    A small square just inside each corner should, on an undamaged card, be
    entirely card stock at the border colour.  A corner that has rounded off
    or chipped leaves platen showing through in that square; a corner that
    has worn through to the white core shifts the sampled colour toward
    white without necessarily losing area.  Both are reported so a human can
    tell rounding from whitening at a glance.
    """
    if colour is None:
        return []
    colour = np.asarray(colour, dtype=np.float32)
    to_white, to_white_norm = _toward_white(colour)
    ch, cw = rgb.shape[:2]
    ink = _card_band(rgb)
    box = max(4, int(round(CORNER_BOX_MM * ppmm)))

    out = []
    for sx, sy in CORNERS:
        cx, cy = cut[sx], cut[sy]
        x0 = int(cx) - box if sx == "right" else int(cx)
        x1 = int(cx) if sx == "right" else int(cx) + box
        y0 = int(cy) - box if sy == "bottom" else int(cy)
        y1 = int(cy) if sy == "bottom" else int(cy) + box
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(cw, x1), min(ch, y1)
        if x1 - x0 < 3 or y1 - y0 < 3:
            continue
        patch_ink = ink[y0:y1, x0:x1]
        fill = float(patch_ink.mean())
        deficit = 1.0 - fill
        whiteness = 0.0
        if fill > 0.4:
            sample = np.median(rgb[y0:y1, x0:x1][patch_ink], axis=0).astype(np.float32)
            whiteness = max(0.0, min(1.0, float(np.dot(sample - colour, to_white)) / to_white_norm))
        severity = max(deficit, whiteness)
        if deficit < 0.12 and whiteness < WHITENING_FLAG:
            continue
        out.append({
            "corner": f"{sy}-{sx}", "box": [x0, y0, x1, y1],
            "fillDeficit": round(deficit, 3), "whiteness": round(whiteness, 3),
            "severity": round(severity, 3),
        })
    return out


def edge_whitening(rgb: np.ndarray, cut: dict, ppmm: float, colour) -> list[dict]:
    """Short stretches of a cut edge that read lighter than the rest of the border.

    Sampled in short segments along each side, away from the corners (those
    are handled separately, and often skew the average) -- a segment that has
    shifted noticeably toward white can mean the surface layer has worn
    through along that stretch.
    """
    if colour is None:
        return []
    colour = np.asarray(colour, dtype=np.float32)
    to_white, to_white_norm = _toward_white(colour)
    ch, cw = rgb.shape[:2]
    ink = _card_band(rgb)
    depth = max(3, int(round(WHITENING_DEPTH_MM * ppmm)))
    seg = max(6, int(round(EDGE_SEGMENT_MM * ppmm)))
    margin = max(2, int(round(CORNER_BOX_MM * ppmm)))

    out = []
    for side in SIDES:
        vertical = side in ("left", "right")
        lo, hi = ((int(cut["top"]) + margin, int(cut["bottom"]) - margin) if vertical
                  else (int(cut["left"]) + margin, int(cut["right"]) - margin))
        if hi - lo < seg:
            continue
        x = int(cut[side])
        for t in range(lo, hi - seg, seg):
            if vertical:
                a, b = (x, x + depth) if side == "left" else (x - depth, x)
                box = [max(0, a), t, min(cw, b), t + seg]
                patch_rgb, patch_ink = rgb[t:t + seg, box[0]:box[2]], ink[t:t + seg, box[0]:box[2]]
            else:
                a, b = (x, x + depth) if side == "top" else (x - depth, x)
                box = [t, max(0, a), t + seg, min(ch, b)]
                patch_rgb, patch_ink = rgb[box[1]:box[3], t:t + seg], ink[box[1]:box[3], t:t + seg]
            if patch_rgb.size == 0 or patch_ink.mean() < 0.6:
                continue
            sample = np.median(patch_rgb[patch_ink], axis=0).astype(np.float32)
            whiteness = max(0.0, min(1.0, float(np.dot(sample - colour, to_white)) / to_white_norm))
            if whiteness < WHITENING_FLAG:
                continue
            out.append({"side": side, "box": [int(v) for v in box], "whiteness": round(whiteness, 3)})
    return out


def _inner_box_estimate(cut: dict, frame: dict | None, ppmm: float) -> dict:
    """Best-guess face boundary: the traced frame where found, else a nominal inset."""
    inset = DEFAULT_BORDER_MM * ppmm
    frame = frame or {}
    return {
        "left": frame.get("left") if frame.get("left") is not None else cut["left"] + inset,
        "right": frame.get("right") if frame.get("right") is not None else cut["right"] - inset,
        "top": frame.get("top") if frame.get("top") is not None else cut["top"] + inset,
        "bottom": frame.get("bottom") if frame.get("bottom") is not None else cut["bottom"] - inset,
    }


def surface_anomalies(rgb: np.ndarray, inner: dict, ppmm: float) -> list[dict]:
    """Patches on the card face that read lighter than their surroundings.

    Not a defect classifier -- it has no idea what the artwork is supposed to
    look like.  Rather than flag any patch that differs from its neighbours
    (which a busy or holo-foil face trips on constantly, since normal art has
    hard colour edges everywhere), this only counts deviation that moves
    toward white in every channel at once.  A hue change at an art edge
    usually raises some channels and lowers others; a scratch, crease or worn
    patch exposing lighter stock underneath raises all of them together.
    That is a narrower net -- it will miss colour-only damage -- but it does
    not fire on ordinary artwork the way a plain contrast test does.
    """
    ch, cw = rgb.shape[:2]
    margin = max(2, int(round(SURFACE_MARGIN_MM * ppmm)))
    x0, x1 = max(0, int(inner["left"]) + margin), min(cw, int(inner["right"]) - margin)
    y0, y1 = max(0, int(inner["top"]) + margin), min(ch, int(inner["bottom"]) - margin)
    if x1 - x0 < 40 or y1 - y0 < 40:
        return []

    region = rgb[y0:y1, x0:x1]
    blur_radius = max(2.0, SURFACE_BLUR_MM * ppmm / 2.0)
    blurred = np.asarray(Image.fromarray(region).filter(ImageFilter.GaussianBlur(blur_radius)))
    diff = region.astype(np.float32) - blurred.astype(np.float32)
    dev = np.clip(diff, 0, None).sum(axis=2)  # only the toward-white part of the shift

    cell = max(3, int(round(SURFACE_GRID_MM * ppmm)))
    gh, gw = dev.shape[0] // cell, dev.shape[1] // cell
    if gh < 2 or gw < 2:
        return []
    grid = dev[:gh * cell, :gw * cell].reshape(gh, cell, gw, cell).mean(axis=(1, 3))
    hot = grid > SURFACE_DEVIATION_FLAG
    visited = np.zeros_like(hot)
    cell_mm2 = (cell / ppmm) ** 2

    comps = []
    for r in range(gh):
        for c in range(gw):
            if not hot[r, c] or visited[r, c]:
                continue
            stack, cells = [(r, c)], []
            visited[r, c] = True
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < gh and 0 <= nc < gw and hot[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            area_mm2 = len(cells) * cell_mm2
            if area_mm2 < SURFACE_MIN_AREA_MM2:
                continue
            rows, cols = [p[0] for p in cells], [p[1] for p in cells]
            comps.append({
                "box": [x0 + min(cols) * cell, y0 + min(rows) * cell,
                        x0 + (max(cols) + 1) * cell, y0 + (max(rows) + 1) * cell],
                "areaMm2": round(area_mm2, 1),
                "severity": round(float(grid[rows, cols].mean()), 1),
            })
    comps.sort(key=lambda c: -c["areaMm2"])
    return comps[:SURFACE_MAX_REPORT]


# --------------------------------------------------------------------------
# the whole job, for one card
# --------------------------------------------------------------------------

def _analyse_crop(crop: Image.Image, ppmm: float, quarter_turns: int | None = None,
                   bg=None, lenient: bool = False, extra_depths: tuple = (),
                   known_rect=None, refine_known_rect: bool = True, progress=None) -> dict:
    """Cut edge, frame, and condition checks for a roughly-upright card crop.

    Shared by the scanner path (a crop taken from a sheet, background always
    near-white, lenient always off) and the photo path (a crop produced by
    perspective-correcting a phone photo, background whatever colour the
    caller sampled, lenient on to see through glare) -- both just hand over
    an image and a scale and get the same tracing back. progress, if given,
    is called with a short label at each real stage, for a caller that wants
    to show more than an elapsed-time counter.
    """
    def stage(label):
        if progress:
            progress(label)

    if quarter_turns is None:
        quarter_turns = 1 if crop.size[0] > crop.size[1] else 0
    quarter_turns %= 4
    for _ in range(quarter_turns):
        crop = crop.transpose(Image.ROTATE_90)

    stage("tracing the cut edge and printed frame")
    arr = np.asarray(crop)
    traced = trace_card(arr, ppmm, bg, lenient, extra_depths, known_rect, refine_known_rect)
    skew = _skew_from(traced["cut"])
    if abs(skew) > 0.01:
        stage("straightening and re-tracing")
        crop = crop.rotate(skew, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
        arr = np.asarray(crop)
        traced = trace_card(arr, ppmm, bg, lenient, extra_depths, known_rect, refine_known_rect)
    ch, cw = arr.shape[:2]
    band = traced["cut"]

    def anchor(side):
        return ch / 2.0 if side in ("left", "right") else cw / 2.0

    def evaluate(fit, side):
        return None if fit is None else fit[0] * anchor(side) + fit[1]

    cut = {s: evaluate(band[s]["outer"], s) for s in SIDES}
    flags: list[str] = []
    nominal = {"x": CARD_SHORT_MM * ppmm, "y": CARD_LONG_MM * ppmm}

    # an edge that fell off the platen, or was never found, is rebuilt from
    # its opposite number using the fact that every card is 63 x 88 mm
    def rebuild(a: str, b: str, axis: str, limit: float):
        nom = nominal[axis]
        if cut[a] is None and cut[b] is None:
            return
        if cut[a] is None:
            cut[a] = cut[b] - nom
            flags.append(f"{a}-edge-reconstructed")
        elif cut[b] is None:
            cut[b] = cut[a] + nom
            flags.append(f"{b}-edge-reconstructed")
        elif nom - (cut[b] - cut[a]) > 0.12 * ppmm:
            if cut[a] < 2.0:
                cut[a] = cut[b] - nom
                flags.append(f"{a}-edge-reconstructed")
            elif cut[b] > limit - 2.0:
                cut[b] = cut[a] + nom
                flags.append(f"{b}-edge-reconstructed")
            elif "card-measures-under-size" not in flags:
                flags.append("card-measures-under-size")

    rebuild("left", "right", "x", cw)
    rebuild("top", "bottom", "y", ch)

    result = {
        "quarterTurns": quarter_turns,
        "skewDeg": round(skew, 3),
        "cropSize": [cw, ch],
        "pxPerMm": round(ppmm, 4),
        "borderSpread": round(traced["spread"], 1),
        "cut": {k: (None if v is None else round(v, 2)) for k, v in cut.items()},
        "frame": None,
        "frameConfidence": 0.0,
        "flags": flags,
        "complete": all(v is not None for v in cut.values()),
    }
    if not result["complete"]:
        flags.append("card-runs-off-the-sheet")
        result["corners"], result["edgeWhitening"], result["surfaceAnomalies"] = [], [], []
        return result

    stage("checking corners and edges")
    colour = traced.get("colour")
    result["borderColour"] = None if colour is None else [round(float(v), 1) for v in colour]
    result["corners"] = corner_defects(arr, cut, ppmm, colour)
    result["edgeWhitening"] = edge_whitening(arr, cut, ppmm, colour)
    if any(c["severity"] >= CORNER_FLAG_SEVERITY for c in result["corners"]):
        flags.append("corner-wear")
    if result["edgeWhitening"]:
        flags.append("edge-whitening")

    # Face-interior surface anomaly detection was tried and pulled: without a
    # reference image of the same card undamaged, any contrast- or whitening-
    # based heuristic flags ordinary light-coloured artwork just as readily as
    # real damage. Left unset rather than shipped noisy; see surface_anomalies().
    def finish() -> dict:
        result["surfaceAnomalies"] = []
        return result

    span_w, span_h = cut["right"] - cut["left"], cut["bottom"] - cut["top"]
    result["cardSizeMm"] = [round(span_w / ppmm, 2), round(span_h / ppmm, 2)]
    if span_w > nominal["x"] + 0.8 * ppmm or span_h > nominal["y"] + 0.8 * ppmm:
        flags.append("outer-trace-may-be-a-sleeve")

    result["kind"] = traced["kind"]
    if traced["kind"] == "full-art":
        flags.append("full-art-frame")

    frame, confs = {}, []
    if traced["frame"] is not None:
        for side in SIDES:
            fit = traced["frame"][side]["inner"]
            pos = evaluate(fit, side)
            straight = max(3.0, 0.22 * ppmm)
            if pos is None or fit[2] > straight or fit[3] < 0.62:
                frame[side], _ = None, confs.append(0.0)
                continue
            width_mm = abs(pos - cut[side]) / ppmm
            if not (0.7 <= width_mm <= 6.5):
                frame[side], _ = None, confs.append(0.0)
                continue
            frame[side] = pos
            confs.append(max(0.0, min(1.0, fit[3] * (1.0 - min(fit[2] / 6.0, 1.0)))))
        result["frameConfidence"] = round(float(np.mean(confs)), 3)
    else:
        frame = {s: None for s in SIDES}

    # Sides the strict tracer couldn't confirm still get a rough starting
    # position where one is findable, so a human is nudging a guide instead of
    # hunting for it from a flat default -- see guess_frame().
    missing = [s for s in SIDES if frame[s] is None]
    if missing:
        guess = guess_frame(arr, (int(cut["left"]), int(cut["top"]),
                                   int(cut["right"]), int(cut["bottom"])), ppmm)
        frame_guess = {}
        for side in missing:
            pos = guess.get(side)
            if pos is not None and 0.7 <= abs(pos - cut[side]) / ppmm <= 6.5:
                frame_guess[side] = round(pos, 2)
        if frame_guess:
            result["frameGuess"] = frame_guess

    found = [k for k, v in frame.items() if v is not None]
    if not found:
        flags.append("no-print-frame-found")
        return finish()
    result["frame"] = {k: (None if v is None else round(v, 2)) for k, v in frame.items()}
    if len(found) < 4:
        flags.append("frame-only-partly-found")
    return finish()


def analyse_card(sheet: Sheet, region: dict, quarter_turns: int | None = None,
                  progress=None) -> dict:
    crop = sheet.image.crop(tuple(region["cropBox"]))
    result = _analyse_crop(crop, sheet.px_per_mm, quarter_turns, progress=progress)
    result.update({
        "index": region["index"], "box": region["box"],
        "cropBox": region["cropBox"], "dpi": sheet.dpi,
    })
    return result


# --------------------------------------------------------------------------
# ratios
# --------------------------------------------------------------------------

def measure(cut: dict, frame: dict, px_per_mm: float) -> dict | None:
    """Ratios for whichever axes have all four lines. One axis is still useful."""
    def width(a: str):
        if cut.get(a) is None or frame.get(a) is None:
            return None
        return abs(frame[a] - cut[a]) / px_per_mm

    mm = {"l": width("left"), "r": width("right"),
          "t": width("top"), "b": width("bottom")}

    def split(a, b):
        if a is None or b is None or a + b <= 0:
            return None
        return [round(100 * a / (a + b), 1), round(100 * b / (a + b), 1)]

    lr, tb = split(mm["l"], mm["r"]), split(mm["t"], mm["b"])
    if lr is None and tb is None:
        return None
    worst = max((lr or []) + (tb or []))
    return {
        "bordersMm": {k: (None if v is None else round(v, 3)) for k, v in mm.items()},
        "lr": lr, "tb": tb,
        "partial": lr is None or tb is None,
        "worst": round(worst, 1),
        "ceiling": ceiling_for(worst),
    }


def ceiling_for(worst: float) -> dict:
    """Highest grade this centering still allows on the front of the card."""
    if worst <= 55:
        return {"grade": 10, "label": "Gem Mint eligible on centering"}
    if worst <= 60:
        return {"grade": 9, "label": "Mint ceiling"}
    if worst <= 65:
        return {"grade": 8, "label": "NM-MT ceiling"}
    if worst <= 70:
        return {"grade": 7, "label": "NM ceiling"}
    return {"grade": 0, "label": "Centering caps this below 7"}


# --------------------------------------------------------------------------

def analyse_sheet(path: str, progress=None) -> dict:
    if progress:
        progress("reading the scan and finding cards on it")
    sheet = get_sheet(path)
    regions = segment(sheet)
    cards = []
    for region in regions:
        stage = (lambda label, i=region["index"]:
                 progress(f"card {i + 1} of {len(regions)}: {label}")) if progress else None
        card = analyse_card(sheet, region, progress=stage)
        if card.get("frame"):
            card["measurement"] = measure(card["cut"], card["frame"], sheet.px_per_mm)
        cards.append(card)
    return {"sheet": sheet.info(), "cards": cards}


if __name__ == "__main__":
    import sys

    data = analyse_sheet(sys.argv[1])
    for card in data["cards"]:
        m = card.get("measurement")
        print(f"card {card['index']}  size {card.get('cardSizeMm')} mm  "
              f"skew {card['skewDeg']:+.3f}  conf {card['frameConfidence']}  "
              f"spread {card['borderSpread']}")
        if m:
            print(f"   mm  {m['bordersMm']}")
            print(f"   L/R {m['lr']}   T/B {m['tb']}   worst {m['worst']} "
                  f"-> PSA {m['ceiling']['grade'] or '<7'}")
        print(f"   flags {card['flags']}")
