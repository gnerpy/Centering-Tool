"""Local web service for the centering bench.

Runs on this machine only.  Scans stay on disk; nothing is uploaded.

    centering-bench\\.venv\\Scripts\\python.exe centering-bench\\server.py
"""

from __future__ import annotations

import csv
import io
import json
import os
import sys
import webbrowser
from datetime import datetime

from flask import Flask, jsonify, request, send_file, send_from_directory
from PIL import Image

import detect

PORT = int(os.environ.get("BENCH_PORT", "8787"))
HERE = os.path.dirname(os.path.abspath(__file__))
SCANS = os.path.abspath(os.environ.get("BENCH_SCANS", os.path.dirname(HERE)))
SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")

app = Flask(__name__, static_folder=None)
_analysis: dict[str, dict] = {}


def _safe(path: str) -> str:
    """Keep file access inside the scans folder."""
    full = os.path.abspath(path)
    if os.path.commonpath([full, SCANS]) != SCANS:
        raise ValueError("that file is outside the scans folder")
    return full


def _analyse(path: str, refresh: bool = False) -> dict:
    key = _safe(path)
    if refresh or key not in _analysis:
        _analysis[key] = detect.analyse_sheet(key)
    return _analysis[key]


def _render(sheet: detect.Sheet, card: dict) -> Image.Image:
    """The exact upright, deskewed crop that the card's coordinates describe."""
    crop = sheet.image.crop(tuple(card["cropBox"]))
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
    for name in sorted(os.listdir(SCANS)):
        full = os.path.join(SCANS, name)
        if not os.path.isfile(full) or not name.lower().endswith(SUFFIXES):
            continue
        stat = os.stat(full)
        rows.append({
            "name": name,
            "path": full,
            "sizeMb": round(stat.st_size / 1e6, 1),
            "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%d %b %Y %H:%M"),
            "analysed": full in _analysis,
        })
    return jsonify({"folder": SCANS, "sheets": rows})


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
    sheet = detect.get_sheet(_safe(path))
    im = _render(sheet, card)
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
    sheet = detect.get_sheet(_safe(path))
    card = data["cards"][index]
    turns = (card["quarterTurns"] + int(body.get("turns", 1))) % 4
    region = {"index": index, "box": card["box"], "cropBox": card["cropBox"]}
    fresh = detect.analyse_card(sheet, region, quarter_turns=turns)
    if fresh.get("frame"):
        fresh["measurement"] = detect.measure(fresh["cut"], fresh["frame"],
                                              sheet.px_per_mm)
    data["cards"][index] = fresh
    return jsonify(fresh)


@app.post("/api/export")
def export():
    body = request.get_json(force=True)
    path = _safe(body["path"])
    rows = body.get("rows", [])
    fmt = body.get("format", "csv")
    stem = os.path.splitext(os.path.basename(path))[0]
    out = os.path.join(SCANS, f"{stem}-centering.{ 'json' if fmt == 'json' else 'csv' }")
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


if __name__ == "__main__":
    url = f"http://127.0.0.1:{PORT}/"
    print(f"Centering bench reading scans from {SCANS}")
    print(f"Open {url}")
    if "--no-browser" not in sys.argv:
        webbrowser.open(url)
    app.run(host="127.0.0.1", port=PORT, debug=False, threaded=True)
