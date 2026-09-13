"""Local web service for the centering bench.

Runs on this machine only.  Scans stay on disk; nothing is uploaded.

    centering-bench\\.venv\\Scripts\\python.exe centering-bench\\server.py
"""

from __future__ import annotations

import csv
import io
import json
import os
import statistics
import sys
import time
import webbrowser
from datetime import datetime

from flask import Flask, jsonify, request, send_file, send_from_directory
from PIL import Image
from werkzeug.utils import secure_filename

import detect
import photo

PORT = int(os.environ.get("BENCH_PORT", "8787"))
HERE = os.path.dirname(os.path.abspath(__file__))
SCANS = os.path.abspath(os.environ.get("BENCH_SCANS", os.path.dirname(HERE)))
SAMPLES = os.path.join(HERE, "samples")     # ships with the repo, so a fresh
RESULTS = os.path.join(HERE, "results")     # clone has something to open
UPLOADS = os.path.join(HERE, "uploads")     # dropped in through the upload modal
SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")

# Fine-tuning: every confirmed card appends a record here (see /api/calibration),
# and /api/tuning mines it for a better starting guess than these hardcoded
# numbers -- see calibrationRows() in static/index.html for what a record holds
# and why only some sides in it are usable evidence.
CALIBRATION_PATH = os.path.join(RESULTS, "calibration.jsonl")
FALLBACK_INSET_MM = {"full-art": 1.3, "no-frame": 2.4, "unreadable": 2.4, "bordered": 2.4}
MIN_TUNING_SAMPLES = 3   # below this, one odd correction could swing the guess too far

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 300 * 1024 * 1024  # a phone photo batch, not a mistake
_analysis: dict[str, dict] = {}
_progress: dict[str, str] = {}


def _roots() -> list[tuple[str, str]]:
    """Folders scans may be read from, as (label, path)."""
    found = [("scans", SCANS)]
    if os.path.isdir(SAMPLES) and os.path.abspath(SAMPLES) != SCANS:
        found.append(("samples", SAMPLES))
    if os.path.abspath(UPLOADS) != SCANS:
        found.append(("uploads", UPLOADS))
    return found


def _safe(path: str) -> str:
    """Keep file access inside a folder we are meant to be reading."""
    full = os.path.abspath(path)
    for _, root in _roots():
        try:
            if os.path.commonpath([full, root]) == os.path.abspath(root):
                return full
        except ValueError:
            continue  # different drive
    raise ValueError("that file is outside the scans and samples folders")


def _is_photo(path: str) -> bool:
    """Uploaded photos get perspective-corrected instead of scanner-segmented."""
    return os.path.commonpath([path, os.path.abspath(UPLOADS)]) == os.path.abspath(UPLOADS)


def _analyse(path: str, refresh: bool = False) -> dict:
    key = _safe(path)
    if refresh or key not in _analysis:
        def report(label: str) -> None:
            _progress[key] = label
            print(f"[analyse] {os.path.basename(key)}: {label}", flush=True)

        start = time.perf_counter()
        try:
            if _is_photo(key):
                card = photo.analyse_photo(key, progress=report)
                if "error" in card:
                    raise ValueError(card["error"])
                _analysis[key] = {
                    "sheet": {"path": key, "name": os.path.basename(key), "dpi": None},
                    "cards": [card],
                }
            else:
                _analysis[key] = detect.analyse_sheet(key, progress=report)
        finally:
            _progress.pop(key, None)
        print(f"[analyse] {os.path.basename(key)} done in "
              f"{time.perf_counter() - start:.2f}s", flush=True)
    return _analysis[key]


def _render(base_image: Image.Image, card: dict) -> Image.Image:
    """The exact upright, deskewed crop that the card's coordinates describe."""
    crop = base_image.crop(tuple(card["cropBox"]))
    for _ in range(card["quarterTurns"] % 4):
        crop = crop.transpose(Image.ROTATE_90)
    if abs(card["skewDeg"]) > 0.01:
        crop = crop.rotate(card["skewDeg"], resample=Image.BICUBIC,
                           fillcolor=(255, 255, 255))
    return crop


# --------------------------------------------------------------------------

@app.get("/")
def index():
    return send_from_directory(os.path.join(HERE, "static"), "index.html")


@app.get("/static/<path:name>")
def static_file(name):
    return send_from_directory(os.path.join(HERE, "static"), name)


@app.get("/api/sheets")
def sheets():
    rows = []
    for label, root in _roots():
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            full = os.path.join(root, name)
            if not os.path.isfile(full) or not name.lower().endswith(SUFFIXES):
                continue
            stat = os.stat(full)
            rows.append({
                "name": name,
                "path": full,
                "source": label,
                "sizeMb": round(stat.st_size / 1e6, 1),
                "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%d %b %Y %H:%M"),
                "analysed": full in _analysis,
            })
    return jsonify({"folder": SCANS, "results": RESULTS, "sheets": rows})


@app.post("/api/upload")
def upload():
    """Save picked or dropped photos into the uploads folder, from this machine only."""
    os.makedirs(UPLOADS, exist_ok=True)
    saved, skipped = [], []
    for file in request.files.getlist("files"):
        name = secure_filename(file.filename or "")
        if not name or not name.lower().endswith(SUFFIXES):
            skipped.append(file.filename or "(unnamed)")
            continue
        stem, ext = os.path.splitext(name)
        dest = os.path.join(UPLOADS, name)
        n = 1
        while os.path.exists(dest):
            dest = os.path.join(UPLOADS, f"{stem}-{n}{ext}")
            n += 1
        file.save(dest)
        saved.append(os.path.basename(dest))
    if not saved:
        return jsonify({"error": "none of those looked like image files", "skipped": skipped}), 400
    return jsonify({"saved": saved, "skipped": skipped})


@app.get("/api/progress")
def progress():
    try:
        key = _safe(request.args.get("path", ""))
    except ValueError:
        return jsonify({"stage": ""})
    return jsonify({"stage": _progress.get(key, "")})


@app.get("/api/analyse")
def analyse():
    path = request.args.get("path", "")
    try:
        data = _analyse(path, request.args.get("refresh") == "1")
    except (ValueError, FileNotFoundError, OSError) as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(data)


@app.get("/api/card.jpg")
def card_image():
    path = request.args.get("path", "")
    index = int(request.args.get("index", "0"))
    width = max(200, min(2400, int(request.args.get("w", "900"))))
    data = _analyse(path)
    card = data["cards"][index]
    key = _safe(path)
    base = photo.get_cached(key)["crop"] if _is_photo(key) else detect.get_sheet(key).image
    im = _render(base, card)
    im = im.resize((width, max(1, round(im.size[1] * width / im.size[0]))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88)
    buf.seek(0)
    return send_file(buf, mimetype="image/jpeg")


@app.post("/api/rotate")
def rotate():
    """Turn one card by a quarter and measure it again in its new orientation."""
    body = request.get_json(force=True)
    path, index = body["path"], int(body["index"])
    data = _analyse(path)
    card = data["cards"][index]
    turns = (card["quarterTurns"] + int(body.get("turns", 1))) % 4
    key = _safe(path)
    if _is_photo(key):
        cached = photo.get_cached(key)
        margin_mm = cached.get("knownRectMarginMm")
        w0, h0 = cached["crop"].size
        # a quarter turn swaps width and height; known_rect is orientation-specific,
        # so it has to be recomputed for the post-rotation size, not reused as-is.
        post_w, post_h = (h0, w0) if turns % 2 else (w0, h0)
        known_rect = (photo.known_rect_for_size(post_w, post_h, cached["ppmm"], margin_mm)
                      if margin_mm is not None else None)
        fresh = detect._analyse_crop(cached["crop"], cached["ppmm"], quarter_turns=turns,
                                      bg=cached["bg"], lenient=True,
                                      extra_depths=() if known_rect else photo.TOPLOADER_DEPTHS_MM,
                                      known_rect=known_rect)
        if known_rect:
            fresh["refinedFromOversizedTrace"] = True
        # box/cropBox describe a region of the cached (pre-rotation) image, which
        # _render crops before applying quarterTurns -- so these stay in that
        # image's own coordinate space regardless of how many turns were requested.
        fresh.update({"index": index, "box": [0, 0, w0, h0], "cropBox": [0, 0, w0, h0],
                      "dpi": None, "photo": True, "localization": card.get("localization")})
        ppmm = cached["ppmm"]
    else:
        sheet = detect.get_sheet(key)
        region = {"index": index, "box": card["box"], "cropBox": card["cropBox"]}
        fresh = detect.analyse_card(sheet, region, quarter_turns=turns)
        ppmm = sheet.px_per_mm
    if fresh.get("frame"):
        fresh["measurement"] = detect.measure(fresh["cut"], fresh["frame"], ppmm)
    data["cards"][index] = fresh
    return jsonify(fresh)


@app.post("/api/export")
def export():
    body = request.get_json(force=True)
    path = _safe(body["path"])
    rows = body.get("rows", [])
    fmt = body.get("format", "csv")
    # Results live inside the repo so a measured sheet travels with the code.
    os.makedirs(RESULTS, exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0]
    out = os.path.join(RESULTS, f"{stem}-centering.{'json' if fmt == 'json' else 'csv'}")
    if fmt == "json":
        with open(out, "w", encoding="utf-8") as fh:
            json.dump({"sheet": os.path.basename(path), "cards": rows}, fh, indent=1)
    else:
        cols = ["card", "label", "lr", "tb", "leftMm", "rightMm", "topMm", "bottomMm",
                "worst", "ceiling", "confirmed", "flags"]
        with open(out, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    return jsonify({"written": out, "rows": len(rows)})


@app.post("/api/calibration")
def calibration():
    """Append confirmed-card records for /api/tuning to learn from later."""
    body = request.get_json(force=True)
    rows = body.get("rows", [])
    if not rows:
        return jsonify({"written": 0})
    os.makedirs(RESULTS, exist_ok=True)
    stamp = datetime.now().isoformat(timespec="seconds")
    with open(CALIBRATION_PATH, "a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps({**row, "loggedAt": stamp}) + "\n")
    return jsonify({"written": len(rows)})


def _read_calibration() -> list[dict]:
    if not os.path.isfile(CALIBRATION_PATH):
        return []
    records = []
    with open(CALIBRATION_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a half-written line from a crashed append; skip it
    return records


def _compute_tuning() -> dict:
    """Median confirmed border width per card `kind`, from real corrections only.

    A side only counts here if it started as a flat kind-level guess --
    placedByTrace false -- so a good trace's own accuracy never gets mixed
    into the number meant to replace a guess where there was no trace at all.
    """
    records = _read_calibration()
    by_kind: dict[str, list[float]] = {}
    for record in records:
        kind = record.get("kind")
        placed = record.get("placedByTrace") or {}
        confirmed = record.get("confirmedInsetMm") or {}
        for side in ("l", "r", "t", "b"):
            if placed.get(side):
                continue
            value = confirmed.get(side)
            if isinstance(value, (int, float)) and 0.3 <= value <= 8.0:
                by_kind.setdefault(kind, []).append(float(value))

    inset, counts = {}, {}
    for kind, values in by_kind.items():
        counts[kind] = len(values)
        if len(values) >= MIN_TUNING_SAMPLES:
            inset[kind] = round(statistics.median(values), 2)
    return {
        "insetMmByKind": inset,
        "counts": counts,
        "fallback": FALLBACK_INSET_MM,
        "totalRecords": len(records),
        "minSamples": MIN_TUNING_SAMPLES,
    }


@app.get("/api/tuning")
def tuning():
    return jsonify(_compute_tuning())


if __name__ == "__main__":
    url = f"http://127.0.0.1:{PORT}/"
    print(f"Centering bench reading scans from {SCANS}")
    print(f"Open {url}")
    if "--no-browser" not in sys.argv:
        webbrowser.open(url)
    app.run(host="127.0.0.1", port=PORT, debug=False, threaded=True)
