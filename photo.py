"""Card localization for phone photos.

A flatbed scan hands detect.py a known geometry for free: axis-aligned,
fixed DPI, blank background. A phone photo has none of that -- the card can
sit at any angle, against any background, at any distance. This module's
job is to find the card in that photo and perspective-correct it into
something that looks like a scan crop: upright, rectangular, at a known
pixels-per-mm, background still visible around it. Once that's done,
detect.py's existing tracing takes over unchanged.

Lens (barrel/fisheye) distortion is not handled here yet -- this module
only undoes camera angle (a homography), not lens curvature. A card shot
through a strongly distorting wide lens will still show curved edges after
this step; that shows up downstream as a poor line fit and low confidence,
rather than a silent wrong answer.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
from PIL import Image

import detect

CARD_SHORT_MM = detect.CARD_SHORT_MM
CARD_LONG_MM = detect.CARD_LONG_MM
OUTPUT_PPMM = 16.0          # target resolution of the corrected crop
MARGIN_MM = 12.0            # background kept around the card, for detect.py's own tracing
MIN_AREA_FRAC = 0.05        # reject contours obviously too small to be the card
BORDER_TOUCH_PX = 3         # a contour this close to the frame edge is treated as spurious

# A rigid toploader holds the card much further from its outer edge than any
# sleeve does -- detect.trace_card's own depth candidates only reach 3.1mm,
# tuned for a bare card's border. These try much deeper, in case the outer
# trace found the toploader rather than the card.
TOPLOADER_DEPTHS_MM = ((3.0, 4.5), (4.5, 6.5), (6.5, 9.0), (9.0, 12.0),
                       (12.0, 15.0), (15.0, 18.5))


def _sample_background(bgr: np.ndarray, patch: int = 40) -> np.ndarray:
    """Median colour of the photo's four corners, assumed to be background."""
    h, w = bgr.shape[:2]
    patch = min(patch, h // 4, w // 4)
    corners = [bgr[:patch, :patch], bgr[:patch, -patch:],
               bgr[-patch:, :patch], bgr[-patch:, -patch:]]
    return np.median(np.concatenate([c.reshape(-1, 3) for c in corners]), axis=0)


def _order_quad(pts: np.ndarray) -> np.ndarray:
    """Sort 4 points into (top-left, top-right, bottom-right, bottom-left)."""
    centre = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0])
    ordered = pts[np.argsort(angles)]
    # rotate so the point closest to the top-left corner comes first
    start = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    return np.roll(ordered, -start, axis=0)


def find_card_quad(bgr: np.ndarray, detect_long_side: int = 900) -> dict:
    """Locate the card (or its sleeve/toploader) as a quadrilateral in a photo.

    Returns a dict with "quad" (4x2 float array, full-resolution pixel
    coordinates, ordered tl/tr/br/bl) and diagnostics, or "quad": None with a
    "reason" if nothing plausible was found.
    """
    h, w = bgr.shape[:2]
    scale = detect_long_side / max(h, w)
    small = cv2.resize(bgr, (max(1, int(w * scale)), max(1, int(h * scale))))
    sh, sw = small.shape[:2]

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    # A bilateral filter smooths flat/textured surfaces (a leather mat, a
    # patterned tablecloth) while leaving strong, real edges sharp -- a plain
    # blur either leaves texture noise intact or blurs the card edge too.
    smooth = cv2.bilateralFilter(gray, 9, 60, 60)
    edges = cv2.Canny(smooth, 30, 90)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    img_area = sh * sw
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < img_area * MIN_AREA_FRAC:
            continue
        x, y, cw_, ch_ = cv2.boundingRect(cnt)
        if (x <= BORDER_TOUCH_PX or y <= BORDER_TOUCH_PX
                or x + cw_ >= sw - BORDER_TOUCH_PX or y + ch_ >= sh - BORDER_TOUCH_PX):
            continue  # a real photographed card has visible margin; this is probably noise
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        candidates.append((area, approx.reshape(4, 2).astype(np.float64)))

    if not candidates:
        # No background-vs-card boundary to trace at all -- but if the whole
        # photo's own aspect ratio is already close to a real card's, this is
        # probably an image that arrived pre-cropped (a marketplace listing,
        # a screenshot, an export from another tool), not a failed detection.
        # Treat its own bounds as the card rather than giving up.
        aspect = max(h, w) / min(h, w)
        target = CARD_LONG_MM / CARD_SHORT_MM
        aspect_err = abs(aspect - target) / target
        if aspect_err < 0.06:
            quad = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float64)
            return {"quad": quad, "areaFrac": 1.0, "keystone": 0.0,
                     "aspectErr": round(float(aspect_err), 3), "wholeImage": True}
        return {"quad": None, "reason": "no card-shaped outline found in the photo"}

    candidates.sort(key=lambda c: -c[0])
    area, pts = candidates[0]
    quad = _order_quad(pts) / scale  # back to full-resolution coordinates

    tl, tr, br, bl = quad
    top, bottom = np.linalg.norm(tr - tl), np.linalg.norm(br - bl)
    left, right = np.linalg.norm(bl - tl), np.linalg.norm(br - tr)
    keystone = max(abs(top - bottom) / max(top, bottom), abs(left - right) / max(left, right))
    aspect = ((left + right) / 2) / ((top + bottom) / 2)
    target = CARD_LONG_MM / CARD_SHORT_MM
    aspect_err = min(abs(aspect - target), abs(aspect - 1 / target)) / target

    return {
        "quad": quad, "areaFrac": round(area / img_area, 3),
        "keystone": round(float(keystone), 3), "aspectErr": round(float(aspect_err), 3),
    }


def find_inner_quad(bgr: np.ndarray, area_frac_range=(0.15, 0.85), aspect_tol=0.08):
    """Find the card's own rectangle nested inside an already-corrected, oversized crop.

    Used when the outer trace turned out to be a toploader rather than the
    card: the toploader's own outline was already used to perspective-correct
    the photo, so the card inside it is now close to axis-aligned too, and
    shows up as its own clean, near-63x88mm rectangle -- no more corners or
    angle to solve, only which contour is the real one.
    """
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    smooth = cv2.bilateralFilter(gray, 9, 60, 60)
    edges = cv2.Canny(smooth, 20, 70)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    img_area = h * w
    target = CARD_LONG_MM / CARD_SHORT_MM

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (img_area * area_frac_range[0] <= area <= img_area * area_frac_range[1]):
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        quad = _order_quad(approx.reshape(4, 2).astype(np.float64))
        tl, tr, br, bl = quad
        top, bottom = np.linalg.norm(tr - tl), np.linalg.norm(br - bl)
        left, right = np.linalg.norm(bl - tl), np.linalg.norm(br - tr)
        aspect = ((left + right) / 2) / ((top + bottom) / 2)
        aspect_err = min(abs(aspect - target), abs(aspect - 1 / target)) / target
        if aspect_err > aspect_tol:
            continue
        candidates.append((aspect_err, area, cnt))

    if not candidates:
        return None
    candidates.sort(key=lambda c: (c[0], -c[1]))  # closest to true card proportions, then largest
    best_cnt = candidates[0][2]
    # approxPolyDP's 4 corners are a coarse simplification, and a card's own
    # rounded corners throw them off further -- each by a few pixels, enough
    # to tilt the whole rectangle a degree or so. minAreaRect fits the entire
    # contour's point cloud instead, which is far less sensitive to that.
    box = cv2.boxPoints(cv2.minAreaRect(best_cnt))
    return _order_quad(box.astype(np.float64))


def known_rect_for_size(w: int, h: int, ppmm: float, margin_mm: float = MARGIN_MM):
    """Where the card's own edge sits in a crop of this size, at this margin.

    Depends only on the crop's own orientation (portrait or landscape), not
    on any tracing -- so it's exactly as valid after a 90 degree rotation
    (which swaps w and h) as it was before, just recomputed for the new size
    rather than carried over as fixed coordinates that no longer apply.
    """
    m = margin_mm * ppmm
    portrait = h >= w
    card_w_mm, card_h_mm = (CARD_SHORT_MM, CARD_LONG_MM) if portrait else (CARD_LONG_MM, CARD_SHORT_MM)
    card_w_px, card_h_px = card_w_mm * ppmm, card_h_mm * ppmm
    return (int(round(m)), int(round(m)),
            int(round(m + card_w_px)), int(round(m + card_h_px)))


def warp_card(bgr: np.ndarray, quad: np.ndarray, ppmm: float = OUTPUT_PPMM,
              margin_mm: float = MARGIN_MM):
    """Perspective-correct the quad onto an upright rectangle, background kept around it.

    The margin is not padding added after the fact -- the destination
    rectangle is simply drawn larger than the quad, so warpPerspective samples
    a bit further out along the same homography and pulls in real background
    pixels from just outside the card in the original photo. That gives
    detect.py's tracing the same kind of background margin a scanner crop has.
    """
    tl, tr, br, bl = quad
    top, bottom = np.linalg.norm(tr - tl), np.linalg.norm(br - bl)
    left, right = np.linalg.norm(bl - tl), np.linalg.norm(br - tr)
    w_px, h_px = (top + bottom) / 2, (left + right) / 2
    portrait = h_px >= w_px
    short_mm, long_mm = CARD_SHORT_MM, CARD_LONG_MM
    card_w_mm, card_h_mm = (short_mm, long_mm) if portrait else (long_mm, short_mm)

    m = margin_mm * ppmm
    card_w_px, card_h_px = card_w_mm * ppmm, card_h_mm * ppmm
    out_w, out_h = int(round(card_w_px + 2 * m)), int(round(card_h_px + 2 * m))
    dst = np.array([[m, m], [m + card_w_px, m], [m + card_w_px, m + card_h_px],
                    [m, m + card_h_px]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    warped = cv2.warpPerspective(bgr, matrix, (out_w, out_h), flags=cv2.INTER_LANCZOS4,
                                  borderMode=cv2.BORDER_REPLICATE)
    # the quad's own corners land exactly here, by construction of the warp --
    # a caller that trusts this contour more than any colour search can treat
    # it as the known, precise cut edge rather than re-deriving it.
    known_rect = known_rect_for_size(out_w, out_h, ppmm, margin_mm)
    return warped, ppmm, known_rect, matrix


_CACHE: dict[str, dict] = {}  # path -> {"crop": pre-rotation PIL image, "bg": colour, "ppmm": float}


def get_cached(path: str) -> dict | None:
    """The pre-rotation crop, background colour, and scale for re-rendering or re-rotating."""
    return _CACHE.get(os.path.abspath(path))


def analyse_photo(path: str, progress=None) -> dict:
    """The photo-mode equivalent of detect.analyse_sheet, for a single card."""
    if progress:
        progress("reading the photo and finding the card in it")
    bgr = cv2.imread(path)
    if bgr is None:
        return {"error": f"could not read {path} as an image"}

    found = find_card_quad(bgr)
    if found["quad"] is None:
        return {"error": found["reason"]}

    localization = {"areaFrac": found["areaFrac"], "keystone": found["keystone"],
                     "aspectErr": found["aspectErr"]}

    if found.get("wholeImage"):
        # No real background exists anywhere in this image, so a colour-vs-
        # background search has nothing to find -- go straight to trusting
        # the image's own bounds, refined the same precise way a toploader's
        # inner card edge is: a narrow-band scanline search, not colour
        # matching against a background that was never photographed.
        if progress:
            progress("no background margin found; trusting the image's own edges")
        warped, ppmm, known_rect, _ = warp_card(bgr, found["quad"], margin_mm=6.0)
        rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        crop = Image.fromarray(rgb)
        base_w, base_h = crop.size
        bg = _sample_background(warped)
        _CACHE.clear()
        _CACHE[os.path.abspath(path)] = {"crop": crop, "bg": bg, "ppmm": ppmm, "knownRectMarginMm": 6.0}
        # This kind of image can have a real margin, even a very thin and
        # asymmetric one, on some sides and none at all on others -- trace_card
        # now only accepts a refined side when it fits confidently, and falls
        # back to the flat assumption per side otherwise, so a genuinely
        # margin-less side can't be dragged off by internal print detail or
        # compression noise the way an all-or-nothing toggle would allow.
        result = detect._analyse_crop(crop, ppmm, quarter_turns=None, bg=bg, lenient=True,
                                       known_rect=known_rect, progress=progress)
        result["flags"].append("photo-had-no-background-margin")
        if result.get("frame"):
            result["measurement"] = detect.measure(result["cut"], result["frame"], ppmm)
        result.update({
            "index": 0, "box": [0, 0, base_w, base_h], "cropBox": [0, 0, base_w, base_h],
            "dpi": None, "photo": True, "localization": localization,
        })
        return result

    if progress:
        progress("correcting the camera angle")
    bg = _sample_background(bgr)
    warped, ppmm, _, matrix1 = warp_card(bgr, found["quad"])
    rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    crop = Image.fromarray(rgb)
    base_w, base_h = crop.size
    _CACHE.clear()  # one photo at a time, same spirit as detect.get_sheet's cache
    _CACHE[os.path.abspath(path)] = {"crop": crop, "bg": bg, "ppmm": ppmm, "knownRectMarginMm": None}

    # warp_card keeps whichever orientation the card had in-frame (portrait or
    # landscape); let _analyse_crop auto-detect and turn it upright, same as a scan.
    # lenient=True lets a real edge fit survive glare on individual scanlines,
    # rather than giving up whenever any single scanline shows no margin.
    result = detect._analyse_crop(crop, ppmm, quarter_turns=None, bg=bg, lenient=True,
                                   extra_depths=TOPLOADER_DEPTHS_MM, progress=progress)

    if "outer-trace-may-be-a-sleeve" in result.get("flags", []):
        if progress:
            progress("outer trace looks oversized, looking for the card inside it")
        inner = find_inner_quad(warped)
        if inner is not None:
            # inner's corners were found in the *first* warp's pixel grid.
            # Warping from there again would compound two resamplings, each
            # with its own small geometric error -- enough, on a hard photo,
            # to tilt the result by a degree or more. Mapping back through the
            # first warp's own homography and correcting straight from the
            # original photo keeps this to one resampling, one source of truth.
            inner_orig = cv2.perspectiveTransform(
                inner.reshape(-1, 1, 2).astype(np.float64), np.linalg.inv(matrix1)
            ).reshape(-1, 2)
            # the margin around this tighter crop is toploader interior/plastic,
            # not the original photo's background -- sample it fresh, from the
            # new crop's own corners, not the outer crop's. And when a toploader
            # sits nearly flush against the card, that margin can be too thin for
            # any colour search to use -- known_rect trusts this contour's own
            # geometry for the cut edge instead of re-deriving it, and only
            # searches for the frame inside it.
            warped2, ppmm2, known_rect2, _ = warp_card(bgr, inner_orig, margin_mm=6.0)
            bg2 = _sample_background(warped2)
            rgb2 = cv2.cvtColor(warped2, cv2.COLOR_BGR2RGB)
            crop2 = Image.fromarray(rgb2)
            refined = detect._analyse_crop(crop2, ppmm2, quarter_turns=0, bg=bg2,
                                            lenient=True, known_rect=known_rect2,
                                            progress=progress)
            refined["refinedFromOversizedTrace"] = True
            result, ppmm = refined, ppmm2
            base_w, base_h = crop2.size
            _CACHE[os.path.abspath(path)] = {"crop": crop2, "bg": bg2, "ppmm": ppmm2,
                                              "knownRectMarginMm": 6.0}

    if result.get("frame"):
        result["measurement"] = detect.measure(result["cut"], result["frame"], ppmm)
    result.update({
        "index": 0, "box": [0, 0, base_w, base_h], "cropBox": [0, 0, base_w, base_h],
        "dpi": None, "photo": True, "localization": localization,
    })
    return result


if __name__ == "__main__":
    import json
    import sys

    r = analyse_photo(sys.argv[1])
    print(json.dumps({k: v for k, v in r.items() if k not in ("corners", "edgeWhitening")},
                      indent=1))
