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
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# Trading card stock, in millimetres.  Every card in the hobby is this size.
CARD_SHORT_MM = 63.0
CARD_LONG_MM = 88.0

DEFAULT_DPI = 600
SEGMENT_LONG_SIDE = 1600  # segmentation runs on a downscaled copy
SIDES = ("left", "right", "top", "bottom")


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


def _trace_band(mask: np.ndarray, ppmm: float) -> dict:
    """Outer and inner line of the masked band, on all four sides."""
    out = {}
    for side in SIDES:
        o, i, no_margin = _runs_along(mask, side, ppmm)
        clipped = no_margin > 0.30
        out[side] = {
            "outer": None if clipped else robust_line(o),
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


def _card_band(rgb: np.ndarray) -> np.ndarray:
    """Everything meaningfully darker or more coloured than the platen.

    The threshold is set well above what a clear sleeve contributes, so a
    dark full-art card still reads while the sleeve around it does not.
    """
    white = np.array([255.0, 255.0, 255.0])
    return np.abs(rgb.astype(np.float32) - white).sum(axis=2) > 130.0


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


def trace_card(rgb: np.ndarray, ppmm: float) -> dict:
    """Find the cut edge and the print frame in a deskewed card crop.

    A card's border colour runs right up to the cut edge, so both rectangles
    come out of one band: its outside is the cut edge, its inside is the
    frame.  The border has to be sampled at a depth measured from the card
    itself -- a penny sleeve sits a millimetre outside it and would otherwise
    be sampled instead -- so the card is located first, roughly, by what
    stands out from the platen.  Depths are then tried from the very edge
    inward: a shallow one catches the thin printed frame of a full-art card,
    a deep one the wide border of an ordinary one.  Whichever produces a
    63 x 88 mm rectangle is the reading that was right.
    """
    base = _trace_band(_card_band(rgb), ppmm)
    shape = rgb.shape[:2]
    rect = _rect_from(base, shape, ppmm)
    if rect is None:
        return {"cut": base, "frame": None, "spread": 999.0, "kind": "unreadable"}

    straight = max(3.0, 0.22 * ppmm)
    best = None
    for near, far in ((0.15, 0.70), (0.35, 1.05), (0.70, 1.70),
                      (1.30, 2.30), (2.10, 3.10)):
        colour, spread = _ring(rgb, rect, ppmm, near, far)
        if colour is None or spread > 90.0:
            continue
        band = _trace_band(_colour_band(rgb, colour, max(48.0, 3.0 * spread)), ppmm)
        err = _span_error(band, shape, ppmm)
        if err > 0.9 * ppmm:
            continue
        good = sum(1 for s in SIDES if band[s]["inner"] is not None
                   and band[s]["inner"][2] < straight)
        if good < 2:
            continue
        score = 6.0 * good - err / ppmm - 0.03 * spread
        if best is None or score > best[0]:
            best = (score, spread, band)

    if best is None:
        return {"cut": base, "frame": None, "spread": 999.0, "kind": "no-frame"}
    return {"cut": best[2], "frame": best[2], "spread": best[1],
            "kind": "bordered" if best[1] <= 40.0 else "full-art"}


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
# the whole job, for one card
# --------------------------------------------------------------------------

def analyse_card(sheet: Sheet, region: dict, quarter_turns: int | None = None) -> dict:
    ppmm = sheet.px_per_mm
    crop = sheet.image.crop(tuple(region["cropBox"]))

    # scanners do not turn the card upright for you
    if quarter_turns is None:
        quarter_turns = 1 if crop.size[0] > crop.size[1] else 0
    quarter_turns %= 4
    for _ in range(quarter_turns):
        crop = crop.transpose(Image.ROTATE_90)

    arr = np.asarray(crop)
    traced = trace_card(arr, ppmm)
    skew = _skew_from(traced["cut"])
    if abs(skew) > 0.01:
        crop = crop.rotate(skew, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
        arr = np.asarray(crop)
        traced = trace_card(arr, ppmm)
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
            else:
                flags.append("card-measures-under-size")

    rebuild("left", "right", "x", cw)
    rebuild("top", "bottom", "y", ch)

    result = {
        "index": region["index"],
        "box": region["box"],
        "cropBox": region["cropBox"],
        "quarterTurns": quarter_turns,
        "skewDeg": round(skew, 3),
        "cropSize": [cw, ch],
        "dpi": sheet.dpi,
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
        return result

    span_w, span_h = cut["right"] - cut["left"], cut["bottom"] - cut["top"]
    result["cardSizeMm"] = [round(span_w / ppmm, 2), round(span_h / ppmm, 2)]
    if span_w > nominal["x"] + 0.8 * ppmm or span_h > nominal["y"] + 0.8 * ppmm:
        flags.append("outer-trace-may-be-a-sleeve")

    result["kind"] = traced["kind"]
    if traced["kind"] == "full-art":
        flags.append("full-art-frame")
    if traced["frame"] is None:
        flags.append("no-print-frame-found")
        return result

    frame, confs = {}, []
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

    found = [k for k, v in frame.items() if v is not None]
    result["frameConfidence"] = round(float(np.mean(confs)), 3)
    if not found:
        flags.append("no-print-frame-found")
        return result
    result["frame"] = {k: (None if v is None else round(v, 2)) for k, v in frame.items()}
    if len(found) < 4:
        flags.append("frame-only-partly-found")
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
    if worst <= 57:
        return {"grade": 10, "label": "Inside the Gem Mint band - borderline"}
    if worst <= 60:
        return {"grade": 9, "label": "Mint ceiling"}
    if worst <= 65:
        return {"grade": 8, "label": "NM-MT ceiling"}
    if worst <= 70:
        return {"grade": 7, "label": "NM ceiling"}
    return {"grade": 0, "label": "Centering caps this below 7"}


# --------------------------------------------------------------------------

def analyse_sheet(path: str) -> dict:
    sheet = get_sheet(path)
    cards = []
    for region in segment(sheet):
        card = analyse_card(sheet, region)
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
