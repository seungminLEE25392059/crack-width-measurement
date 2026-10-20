"""Human comparison study (Task 1 report, Section 6.6).

Every team member independently estimates the maximum crack width (pixels) and
assigns a severity grade for the same 20 test images, without seeing the
reference masks or one another's answers. The study reports
  1. variation between observers: the standard deviation (SD) of the estimates
     for each image, and the share of images on which all observers give the
     same grade;
  2. observer error against the image-based reference, next to the error of
     each automated method on the same images.
All widths are image-level: "the widest point of any crack in this photo".

Usage:
  python scripts/human_study.py prepare                 # Seungmin, once: fixes the 20 images
  python scripts/human_study.py measure --name Zitong   # every member, alone, on their own computer
  python scripts/human_study.py analyse                 # Seungmin, after every response is in

Files (eval/human_study/):
  selection.json                  image IDs in presentation order and the grade thresholds
                                  (no reference widths, so the coordinator stays blind as well)
  instructions.md                 what participants do
  response_template.csv           for anyone who measures with another tool
  responses/response_<name>.csv   one file per participant
  tables/   per_image.csv, observers.csv, comparison.csv, summary.json     (analyse)
  figures/  human_vs_methods.png                                          (analyse)
Author : Seungmin Lee
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from src import config, data, evaluate, utils, viz  # noqa: E402

plt = viz.plt
STUDY_DIR = config.EVAL_DIR / "human_study"
RESPONSE_DIR = STUDY_DIR / "responses"
SELECTION_FILE = STUDY_DIR / "selection.json"
N_IMAGES = 20
NO_CRACK = "none"                                  # grade typed as 0: no crack visible
GRADE_CHOICES = (NO_CRACK, *config.SEVERITY_LABELS)  # typed as 0, 1, 2, 3
RESPONSE_COLUMNS = ["participant", "image_id", "max_width_px", "severity",
                    "x1", "y1", "x2", "y2", "seconds", "notes"]
MARKERS = {"A": "o", "B": "s", "C": "^", "D": "D"}


# --- prepare: choose the images (once) ---------------------------------------------------------

def reference_widths(ids: list[str]) -> dict[str, float]:
    """Image-level reference max width of every image (NaN when the mask has no crack)."""
    widths = {}
    for n, image_id in enumerate(ids, 1):
        _, ref = data.load_pair(image_id)
        cracks, _, _ = evaluate.measure_mask(ref)
        widths[image_id] = evaluate.image_max_width(cracks)
        print(f"\r  measured {n}/{len(ids)} reference masks", end="", flush=True)
    print()
    return widths


def select_images(widths: dict[str, float], n: int = N_IMAGES, seed: int = config.SEED) -> list[str]:
    """n images spread evenly from the thinnest to the widest crack, in shuffled order.

    Sorting by reference width and taking evenly spaced positions makes sure
    thin, medium and wide cracks are all represented. The order shown to
    participants is then shuffled, so it does not give the ranking away.
    """
    ranked = [image_id for image_id, _ in sorted(widths.items(), key=lambda item: (item[1], item[0]))
              if np.isfinite(widths[image_id])]
    if len(ranked) <= n:
        picked = ranked
    else:
        positions = np.linspace(0, len(ranked) - 1, n).round().astype(int)
        picked = [ranked[p] for p in positions]
    random.Random(seed).shuffle(picked)
    return picked


INSTRUCTIONS = """# Human width study: instructions

Thank you for taking part. This is the human comparison of the Task 1 report
(Section 6.6). It takes about 20-30 minutes.

## Rules

1. Work alone. Until everyone has finished, do not open the reference masks
   (`data/raw/masks`), the method outputs (`results/`), the evaluation tables
   (`eval/val`, `eval/test`) or anyone else's answers.
2. Measure the {n} images in the order the tool shows them.

## Steps

1. `git pull`, then in the repository folder run

```
   python scripts/human_study.py measure --name YourName
```

2. A window shows one photograph. Use the magnifier in the toolbar to zoom in on
   the **widest point of the widest crack** in the photo (switch the magnifier
   off again before clicking).
3. Click once on each edge of the crack at that point, straight across the
   crack (perpendicular to its direction). A line and its length in pixels
   appear. A third click starts again. **Close the window** when the line is right.
4. In the terminal, type the severity grade of the photo:

   | Type | Grade | Max width |
   |---|---|---|
   | 1 | minor | below {low:.1f} px |
   | 2 | moderate | {low:.1f} px up to {high:.1f} px |
   | 3 | severe | {high:.1f} px or more |
   | 0 | no crack visible | (width 0) |

   These grades are the project's experimental, dataset-relative categories
   (Task 1 report, Section 5.5), not engineering safety classes.
5. Add a note if something was unclear (or press Enter). The answer is saved
   at once; you can stop with Ctrl+C and run the command again later to continue.
6. Send `eval/human_study/responses/response_YourName.csv` to Seungmin.
   Do not commit it until everyone has finished.

## If the window does not open

Use any viewer that shows pixel positions (for example Fiji: Straight Line tool,
then Analyze > Measure gives the length in pixels). Copy
`eval/human_study/response_template.csv` to
`eval/human_study/responses/response_YourName.csv` and fill in `participant`,
`max_width_px`, `severity` (minor / moderate / severe / none) and, if you like,
`seconds` and `notes` for every row.
"""


def prepare(split: str, n: int, force: bool) -> None:
    """Fix the image set once and write the files participants need."""
    if SELECTION_FILE.exists() and not force:
        sys.exit(f"{SELECTION_FILE.relative_to(ROOT)} already exists: the image set must stay fixed "
                 "(use --force only if the study has not started).")
    if utils.is_placeholder(evaluate.measure.measure_cracks):
        print("Warning: src/measure.py is still the placeholder; commit Junsen's version first if possible.")
    ids = data.load_split()[split]
    thresholds = evaluate.severity_thresholds_from_train(data.load_split()["train"])
    print(f"Choosing {n} of {len(ids)} {split} images (reference widths are not shown or saved)...")
    picked = select_images(reference_widths(ids), n)
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    RESPONSE_DIR.mkdir(parents=True, exist_ok=True)
    SELECTION_FILE.write_text(json.dumps({
        "split": split, "images": picked,
        "thresholds_px": {"low": thresholds[0], "high": thresholds[1]},
        "rule": "evenly spaced over the reference image max width, then shuffled (seed 42)",
        "created": dt.date.today().isoformat(),
    }, indent=2), encoding="utf-8")
    (STUDY_DIR / "instructions.md").write_text(
        INSTRUCTIONS.format(n=len(picked), low=thresholds[0], high=thresholds[1]), encoding="utf-8")
    with open(STUDY_DIR / "response_template.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(RESPONSE_COLUMNS)
        writer.writerows([["", image_id] + [""] * (len(RESPONSE_COLUMNS) - 2) for image_id in picked])
    print(f"Saved {len(picked)} images to {SELECTION_FILE.relative_to(ROOT)}, plus instructions.md and "
          "response_template.csv. Commit eval/human_study/ and send instructions.md to the team.")


# --- measure: the on-screen distance tool ----------------------------------------------------

class WidthPicker:
    """Two clicks on an image -> a line and its length in pixels.

    Clicks made while the toolbar's zoom or pan tool is switched on are ignored,
    so zooming never adds a point. A third click starts a new line.
    """

    def __init__(self, ax, toolbar=None):
        self.ax, self.toolbar = ax, toolbar
        self.points: list[tuple[float, float]] = []
        self.artists: list = []

    @property
    def width(self) -> float | None:
        if len(self.points) < 2:
            return None
        (x1, y1), (x2, y2) = self.points
        return float(np.hypot(x2 - x1, y2 - y1))

    def clear(self) -> None:
        for artist in self.artists:
            artist.remove()
        self.points, self.artists = [], []

    def on_click(self, event) -> None:
        if event.inaxes is not self.ax or event.button != 1 or event.xdata is None:
            return
        if self.toolbar is not None and getattr(self.toolbar, "mode", "") != "":
            return
        if len(self.points) == 2:
            self.clear()
        self.points.append((float(event.xdata), float(event.ydata)))
        self.artists += self.ax.plot(event.xdata, event.ydata, "+", color="#ff2a6d", ms=12, mew=1.5)
        if len(self.points) == 2:
            xs, ys = zip(*self.points)
            self.artists += self.ax.plot(xs, ys, "-", color="#ff2a6d", lw=1.2)
            self.ax.set_xlabel(f"Width: {self.width:.1f} px   (close the window to accept, click again to redo)",
                               fontsize=11, color="#0b0b0b")
        self.ax.figure.canvas.draw_idle()


def use_interactive_backend() -> None:
    """viz.py selects the file-only Agg backend; the tool needs a window."""
    for backend in ("TkAgg", "QtAgg", "MacOSX", "WXAgg"):
        try:
            plt.switch_backend(backend)
            return
        except Exception:  # backend not installed on this computer
            continue
    sys.exit("No interactive matplotlib backend: use response_template.csv with another tool "
             "(see eval/human_study/instructions.md).")


def pick_width(image: np.ndarray, title: str) -> tuple[float | None, list[tuple[float, float]]]:
    """Open one image, let the participant draw the line, return (width, points)."""
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_xlabel("Zoom in with the magnifier, switch it off, then click both edges of the crack",
                  fontsize=11, color="#52514e")
    picker = WidthPicker(ax, getattr(fig.canvas, "toolbar", None))
    fig.canvas.mpl_connect("button_press_event", picker.on_click)
    plt.show()
    plt.close(fig)
    return picker.width, picker.points


def ask_grade(thresholds: tuple[float, float]) -> str:
    low, high = thresholds
    prompt = (f"  Grade? 1 = minor (< {low:.1f} px), 2 = moderate, 3 = severe (>= {high:.1f} px), "
              "0 = no crack: ")
    while True:
        answer = input(prompt).strip()
        if answer in ("0", "1", "2", "3"):
            return GRADE_CHOICES[int(answer)]
        print("  Please type 0, 1, 2 or 3.")


def load_selection() -> dict:
    if not SELECTION_FILE.exists():
        sys.exit("No eval/human_study/selection.json yet: git pull, or ask Seungmin to run prepare.")
    return json.loads(SELECTION_FILE.read_text(encoding="utf-8"))


def measure_session(name: str) -> None:
    """Show every selected image in turn and save one response row per image."""
    selection = load_selection()
    thresholds = (selection["thresholds_px"]["low"], selection["thresholds_px"]["high"])
    out = RESPONSE_DIR / f"response_{name}.csv"
    done = set(pd.read_csv(out, dtype={"image_id": str})["image_id"]) if out.exists() else set()
    todo = [image_id for image_id in selection["images"] if image_id not in done]
    if not todo:
        print(f"All {len(selection['images'])} images are done: send {out.relative_to(ROOT)} to Seungmin.")
        return
    use_interactive_backend()
    RESPONSE_DIR.mkdir(parents=True, exist_ok=True)
    new_file = not out.exists()
    total = len(selection["images"])
    with open(out, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if new_file:
            writer.writerow(RESPONSE_COLUMNS)
        for image_id in todo:
            number = selection["images"].index(image_id) + 1
            image, _ = data.load_pair(image_id)       # the mask is loaded but never shown
            start = time.perf_counter()
            while True:
                width, points = pick_width(image, f"Image {number} of {total}  (id {image_id})")
                if width is not None:
                    print(f"Image {number}/{total} (id {image_id}): {width:.1f} px")
                    severity = ask_grade(thresholds)
                    break
                answer = input(f"Image {number}/{total}: no line drawn. Enter = show it again, "
                               "0 = no crack visible: ").strip()
                if answer == "0":
                    severity = NO_CRACK
                    break
            if severity == NO_CRACK:
                width, points = 0.0, []
            note = input("  Note (optional, Enter to skip): ").strip()
            coords = [f"{v:.1f}" for point in points for v in point] or [""] * 4
            writer.writerow([name, image_id, f"{width:.2f}", severity, *coords,
                             f"{time.perf_counter() - start:.0f}", note])
            file.flush()
    print(f"Finished. Send {out.relative_to(ROOT)} to Seungmin (do not commit it yet).")


# --- analyse: compare observers, reference and methods -----------------------------------------------

def load_responses(selection: dict) -> pd.DataFrame:
    """All response files as one clean table (one row per participant and image)."""
    files = sorted(RESPONSE_DIR.glob("response_*.csv"))
    if not files:
        sys.exit(f"No responses in {RESPONSE_DIR.relative_to(ROOT)} yet.")
    ids = selection["images"]
    by_number = {int(image_id): image_id for image_id in ids if image_id.isdigit()}
    frames = []
    for path in files:
        frame = pd.read_csv(path, dtype=str).fillna("").reindex(columns=RESPONSE_COLUMNS, fill_value="")
        name = path.stem.removeprefix("response_")
        frame["participant"] = frame["participant"].str.strip().replace("", name)
        frames.append(frame)
    responses = pd.concat(frames, ignore_index=True)
    # Excel turns "007" into "7": map plain numbers back to the dataset IDs.
    responses["image_id"] = [by_number.get(int(i), i) if i.strip().isdigit() else i.strip()
                             for i in responses["image_id"]]
    responses["severity"] = responses["severity"].str.strip().str.lower()
    responses["max_width_px"] = pd.to_numeric(responses["max_width_px"], errors="coerce")
    responses.loc[responses["severity"] == NO_CRACK, "max_width_px"] = 0.0
    responses["seconds"] = pd.to_numeric(responses["seconds"], errors="coerce")
    unanswered = responses["max_width_px"].isna() & (responses["severity"] == "")
    invalid = ~unanswered & (~responses["severity"].isin(GRADE_CHOICES) | responses["max_width_px"].isna()
                             | ~responses["image_id"].isin(ids))
    for _, row in responses[invalid].iterrows():
        print(f"Ignoring invalid row: {row['participant']}, image {row['image_id']}, "
              f"width '{row['max_width_px']}', grade '{row['severity']}'")
    responses = responses[~unanswered & ~invalid]
    return responses.drop_duplicates(["participant", "image_id"], keep="last").reset_index(drop=True)


def image_grade(width: float, thresholds: tuple[float, float]) -> str:
    """Grade of a whole image from its max width; 0 means no crack."""
    return NO_CRACK if width == 0 else evaluate.grade(width, thresholds)


def reference_table(split: str, ids: list[str], thresholds: tuple[float, float]) -> pd.DataFrame:
    rows = []
    for image_id in ids:
        _, ref = data.load_pair(image_id)
        width = evaluate.image_max_width(evaluate.measure_mask(ref)[0])
        rows.append({"image_id": image_id, "ref_max_width": width, "ref_grade": image_grade(width, thresholds)})
    return pd.DataFrame(rows)


def method_estimates(split: str, ids: list[str], methods: list[str],
                     thresholds: tuple[float, float]) -> pd.DataFrame:
    """Image-level max width of each method's mask, measured exactly like the reference.

    A method that finds no crack reports width 0 ("none"), so a missed image
    counts with its full error instead of disappearing from the average.
    """
    rows = []
    for method in methods:
        if not all(utils.result_path(method, split, image_id).exists() for image_id in ids):
            print(f"Method {method}: no masks for all {len(ids)} images in results/{method}/{split}; skipped")
            continue
        timing = config.RESULTS_DIR / method / split / "timing.csv"
        seconds = (pd.read_csv(timing, dtype={"image_id": str}).set_index("image_id")["seconds"]
                   if timing.exists() else pd.Series(dtype=float))
        for image_id in ids:
            cracks, _, _ = evaluate.measure_mask(utils.read_mask(utils.result_path(method, split, image_id)))
            width = evaluate.image_max_width(cracks)
            width = 0.0 if np.isnan(width) else width
            rows.append({"source": f"Method {method}", "method": method, "image_id": image_id,
                         "max_width_px": width, "severity": image_grade(width, thresholds),
                         "seconds": seconds.get(image_id, np.nan)})
    return pd.DataFrame(rows, columns=["source", "method", "image_id", "max_width_px", "severity", "seconds"])


def error_summary(estimates: pd.DataFrame, reference: pd.DataFrame, source: str) -> dict:
    """MAE, signed bias and grade agreement of one observer or method against the reference."""
    merged = estimates.merge(reference, on="image_id")
    error = merged["max_width_px"] - merged["ref_max_width"]
    return {"source": source, "images": len(merged), "max_width_mae": error.abs().mean(),
            "max_width_bias": error.mean(),
            "grade_agreement": (merged["severity"] == merged["ref_grade"]).mean(),
            "severe_as_minor": int(((merged["ref_grade"] == "severe") & (merged["severity"] == "minor")).sum()),
            "seconds_per_image": merged["seconds"].median()}


def plot_study(responses: pd.DataFrame, methods: pd.DataFrame, reference: pd.DataFrame,
               comparison: pd.DataFrame, path: Path) -> None:
    """Left: error of every estimate against the reference width. Right: MAE of observers and methods.

    The left panel shows estimate - reference, so 0 is perfect, above 0 is an
    overestimate and below 0 an underestimate. Its range is set by the
    observers; method errors beyond it sit on the top or bottom edge.
    """
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.6, 4.2), gridspec_kw={"width_ratios": [1.1, 1]})
    observed = responses.merge(reference, on="image_id")
    observed["error"] = observed["max_width_px"] - observed["ref_max_width"]
    limit = max(4.0, 1.25 * float(observed["error"].abs().max()))
    left.axhline(0, color=viz.INK["muted"], lw=1, zorder=1)
    for _, group in observed.groupby("image_id"):
        x = group["ref_max_width"].iloc[0]
        left.plot([x, x], [group["error"].min(), group["error"].max()], color=viz.INK["axis"], lw=1.2, zorder=1)
    left.plot(observed["ref_max_width"], observed["error"], "o", color=viz.INK["muted"], ms=5,
              mec="white", mew=0.8, zorder=2)
    handles = [Line2D([], [], color=viz.INK["muted"], marker="o", lw=0, ms=5, label="Observers (one dot each)")]
    beyond = 0
    for method, group in methods.groupby("method"):
        merged = group.merge(reference, on="image_id")
        error = (merged["max_width_px"] - merged["ref_max_width"]).to_numpy(float)
        beyond += int((np.abs(error) > limit).sum())
        color = viz.METHOD_COLORS.get(method, viz.INK["secondary"])
        left.plot(merged["ref_max_width"], np.clip(error, -limit, limit), MARKERS.get(method, "o"), color=color,
                  ms=7, mec="white", mew=1.2, zorder=3, clip_on=False)
        handles.append(Line2D([], [], color=color, marker=MARKERS.get(method, "o"), lw=0, ms=7,
                              mec="white", mew=1.2, label=f"Method {method}"))
    left.set_ylim(-limit, limit)
    left.set_xlim(0, float(reference["ref_max_width"].max()) * 1.1)
    left.set_xlabel("Reference max width (px)")
    left.set_ylabel("Estimate - reference (px)")
    left.set_title("Error of each estimate (0 = same as reference)", loc="left")
    if beyond:
        left.text(0.99, 0.01, f"{beyond} method point(s) beyond ±{limit:.1f} px sit on the edge",
                  transform=left.transAxes, ha="right", va="bottom", fontsize=7, color=viz.INK["muted"])
    left.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=8, ncol=3)

    table = comparison.iloc[::-1].reset_index(drop=True)      # first row at the top
    rows = np.arange(len(table))
    colors = [viz.METHOD_COLORS.get(source.removeprefix("Method "), viz.INK["muted"])
              if source.startswith("Method ") else
              (viz.INK["secondary"] if source == "Observers (average)" else viz.INK["axis"])
              for source in table["source"]]
    values = table["max_width_mae"].to_numpy(float)
    right.barh(rows, values, height=0.6, color=colors)
    for y, value in zip(rows, values):
        right.text(value, y, f"  {value:.2f}", va="center", fontsize=8, color=viz.INK["primary"])
    right.set_yticks(rows, table["source"])
    right.set_xlim(0, np.nanmax(values) * 1.3 if np.isfinite(values).any() and np.nanmax(values) > 0 else 1)
    right.grid(axis="y", visible=False)
    right.set_xlabel("Max-width MAE (px)")
    right.set_title("Error against the reference (lower is better)", loc="left")
    fig.tight_layout()
    viz.save_figure(fig, path)


def analyse(method_list: list[str]) -> None:
    selection = load_selection()
    split, ids = selection["split"], selection["images"]
    thresholds = (selection["thresholds_px"]["low"], selection["thresholds_px"]["high"])
    saved = config.PARAMS_DIR / "severity.json"
    if saved.exists():
        current = json.loads(saved.read_text(encoding="utf-8"))
        if not np.allclose([current["low"], current["high"]], thresholds):
            print("Warning: params/severity.json changed after prepare; the study keeps its own thresholds.")
    viz.apply_style()
    responses = load_responses(selection)
    print(f"{responses['participant'].nunique()} participants, {len(responses)} answers")
    reference = reference_table(split, ids, thresholds)
    methods = method_estimates(split, ids, method_list, thresholds)

    # 1. Variation between observers, image by image
    stats = responses.groupby("image_id").agg(
        observers=("max_width_px", "size"), mean_estimate=("max_width_px", "mean"),
        sd_estimate=("max_width_px", "std"), min_estimate=("max_width_px", "min"),
        max_estimate=("max_width_px", "max"), grades_given=("severity", "nunique"))
    stats["all_same_grade"] = stats["grades_given"] == 1
    wide = responses.pivot(index="image_id", columns="participant", values="max_width_px").add_prefix("width_")
    per_image = reference.set_index("image_id").join(stats).join(wide).reset_index()

    # 2. Error of each observer and each method against the reference
    observers = pd.DataFrame([error_summary(group, reference, name)
                              for name, group in responses.groupby("participant")])
    average = observers.drop(columns=["source", "images"]).mean().to_dict()
    average.update({"source": "Observers (average)", "images": int(round(observers["images"].mean())),
                    "severe_as_minor": observers["severe_as_minor"].sum()})
    comparison = pd.concat([observers, pd.DataFrame([average])] +
                           [pd.DataFrame([error_summary(group, reference, source)])
                            for source, group in methods.groupby("source")], ignore_index=True)

    multi = per_image[per_image["observers"] >= 2]
    summary = {
        "split": split, "images": len(ids), "participants": int(responses["participant"].nunique()),
        "mean_sd_px": float(multi["sd_estimate"].mean()), "median_sd_px": float(multi["sd_estimate"].median()),
        "max_sd_px": float(multi["sd_estimate"].max()),
        "share_all_same_grade": float(multi["all_same_grade"].mean()),
        "observer_mae_px": {row["source"]: row["max_width_mae"] for _, row in observers.iterrows()},
        "method_mae_px": {row["source"]: row["max_width_mae"] for _, row in comparison.iterrows()
                          if row["source"].startswith("Method ")},
        "thresholds_px": selection["thresholds_px"],
    }
    tables, figures = STUDY_DIR / "tables", STUDY_DIR / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    per_image.to_csv(tables / "per_image.csv", index=False, float_format="%.3f")
    observers.to_csv(tables / "observers.csv", index=False, float_format="%.3f")
    comparison.to_csv(tables / "comparison.csv", index=False, float_format="%.3f")
    (tables / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_study(responses, methods, reference, comparison, figures / "human_vs_methods.png")

    with pd.option_context("display.width", 140):
        print("\nObservers and methods against the reference (image max width)\n",
              comparison.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print(f"\nBetween observers: mean SD {summary['mean_sd_px']:.2f} px per image; all observers gave the "
          f"same grade on {summary['share_all_same_grade']:.0%} of images")
    print(f"Tables: {tables.relative_to(ROOT)}  Figure: {figures.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    step = commands.add_parser("prepare", help="choose the images once (Seungmin)")
    step.add_argument("--split", choices=("val", "test"), default="test")
    step.add_argument("--n", type=int, default=N_IMAGES)
    step.add_argument("--force", action="store_true", help="overwrite an existing selection")
    step = commands.add_parser("measure", help="measure the images (every participant)")
    step.add_argument("--name", required=True, help="your first name, e.g. Zitong")
    step = commands.add_parser("analyse", help="compare observers and methods (Seungmin)")
    step.add_argument("--methods", default=",".join(config.METHODS))
    args = parser.parse_args()

    utils.set_seed()
    if args.command == "prepare":
        prepare(args.split, args.n, args.force)
    elif args.command == "measure":
        measure_session(args.name.strip())
    else:
        analyse([method.strip() for method in args.methods.split(",") if method.strip()])


if __name__ == "__main__":
    main()