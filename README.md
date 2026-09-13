# Centering Tool

Measures PSA-style centering ratios from a flatbed scan of trading cards. Point it at a
sheet with several cards on it; it finds each one, turns it upright, straightens it,
traces the cut edge and the printed frame, and gives you `L/R` and `T/B` in the form PSA
prints on the flip — then lets you drag all eight guides before you trust the number.

Runs entirely on your machine. Scans are read from disk and never uploaded.

## Running it

```powershell
.\run.ps1
```

First run builds `.venv` and installs Flask, Pillow and NumPy; after that it just starts
the server and opens <http://127.0.0.1:8787/>.

Two folders are offered in the dropdown: `samples/` (ships with the repo, so a fresh
clone has something to open straight away) and the folder **above** this one, where your
own scans live. To read yours from somewhere else:

```powershell
$env:BENCH_SCANS = "D:\scans"; .\run.ps1
```

`BENCH_PORT` changes the port (default 8787).

## Using it

1. Pick a scan and press **Measure sheet**. Cards appear down the left.
2. Check the eight guides on the card in the middle. Solid lines are the cut edge,
   dashed lines the printed frame.
3. Drag a guide, or tab to one and nudge with the arrow keys — 0.02 mm a press, 0.2 mm
   with Shift. The loupe under the card shows the guide at 6×.
4. **Rotate 90°** if a card came out sideways or upside down; it re-measures in the new
   orientation.
5. **Confirm card**, then **Export CSV** or **Export JSON**. The file lands in
   `results/` as `<scan name>-centering.csv`, which is tracked by git — commit it and
   your measurements are on both machines. Scans themselves stay out of the repo.

## Fine-tuning as the database grows

Every card you confirm and export is also evidence, not just a result. Alongside the
CSV/JSON, exporting appends one record per confirmed card to `results/calibration.jsonl`
— also tracked by git, so what you learn on one machine is available on the other the
next time you pull.

Only one thing gets learned from it today: where to park a guide the auto-trace couldn't
find at all. Every card gets a `kind` — `bordered`, `full-art`, `no-frame`, or
`unreadable` — and a side with no trace and no rough guess starts from a flat,
kind-level number (1.3 mm for full art, 2.4 mm for everything else) that was originally
just a guess of mine. Once three or more confirmed cards of a kind have a side that
started that same blind way, `/api/tuning` replaces the guess with the **median**
correction actually made, and every card measured afterward starts from that instead —
across sessions and machines, since the log is what's shared, not a model file. The
small `tuned from N confirmed cards` note in the header (hover it) shows the current
numbers and how many corrections built each one.

This is deliberately narrow: it only touches the single fallback number used when
nothing else was found, and only from sides that were genuinely blind (a side the
tracer or its guess did find is excluded, so fixing one doesn't quietly bias the other).
A card style you photograph often — say, a specific full-foil promo run — will pull that
number toward its own true border width after a handful of confirmations, the way the
Jolteon fix in this repo's history did by hand for one card. Corrections to a trace that
already had a real reading aren't used for anything yet; that's a natural next step
(catching a systematic bias in the trace itself, not just the blind fallback) once
`calibration.jsonl` has enough of them to be worth mining.

## How the measurement works

Centering is the ratio of opposing border widths: the gap between the card's cut edge
and the edge of the printed frame. Two rectangles, four numbers.

- **Finding cards.** Threshold against the platen, project onto rows and columns, cut at
  the gaps. Each card gets its own crop, grown outward only as far as the background
  stays blank, so it never picks up the card lying next to it.
- **Turning it upright.** Cards scanned sideways are rotated a quarter turn; then each of
  the four edges is fitted with a line and the crop is rotated by the mean, which fixes
  the degree or so of skew that laying a card on glass always introduces.
- **The two rectangles.** A card's border colour runs right up to the cut edge, so both
  rectangles come out of one band of colour: its outside is the cut edge, its inside is
  the frame. The colour is learned from a thin ring on the card being measured — no
  hard-coded Pokémon yellow — sampled at several depths, and the depth that yields a
  63 × 88 mm rectangle is the one that was right. That size check is also what rejects a
  penny sleeve, which reads about a millimetre outside the real edge.
- **Edges off the platen.** An edge that ran past the glass is rebuilt from the opposite
  edge using the 63 × 88 mm card size, and flagged, so you can see the ratio is inferred
  rather than measured.
- **Full art.** The thin holo frame on a full-art card is not a flat colour and the
  artwork bleeds across it, so no colour rule finds it reliably. Those cards come back
  with the inner guides parked where the frame usually runs, for you to place by eye.

## Scanning for good results

- **Unsleeve the cards.** A penny sleeve puts a soft edge a millimetre or two outside the
  real one, and it is the first thing any detector finds.
- **Leave 10 mm of glass around every card.** An edge past the platen has to be inferred.
- **600 dpi is plenty** — 0.042 mm per pixel, an order of magnitude finer than the
  tolerance, and much lighter than 1200.
- **Scan the backs too.** Centering is graded on both faces, and the back tolerance is
  the looser one.

## What the ratio buys you

| Grade | | Front | Back |
|---|---|---|---|
| PSA 10 | Gem Mint | 55/45 | 75/25 |
| PSA 9 | Mint | 60/40 | 90/10 |
| PSA 8 | NM-MT | 65/35 | 90/10 |
| PSA 7 | NM | 70/30 | 90/10 |

These are ceilings, not predictions — a perfectly centred card can still come back a 7 on
surface or corners. Below PSA 7, centering is almost never what caps the grade.

## Layout

    detect.py            measurement: segmentation, deskew, edge and frame tracing
    server.py            local Flask service and JSON API
    static/index.html    the whole UI
    run.ps1              first-run setup and start
    samples/             one scan to work against anywhere
    results/             exported measurements and calibration.jsonl, tracked

## The sample sheet

`samples/sheet-4-cards.jpg` is a 1200 dpi scan of four cards, kept at full resolution
because dropping it to 600 dpi visibly degrades one edge trace. It is deliberately an
awkward sheet, and covers most of what the tool has to cope with:

| Card | What it exercises |
|---|---|
| 1 Moltres ex | full art with no traceable frame; sits at +1.0 deg of skew |
| 2 Kyogre | clean measurement, 54.7 / 45.3 and 45.7 / 54.3 |
| 3 Kyogre | clean measurement, right on the PSA 9 line |
| 4 Groudon | runs off the bottom of the scan, so it cannot be scored |

Every card on it also runs slightly off the left edge of the platen, so all four exercise
the reconstruct-from-63x88 path.
