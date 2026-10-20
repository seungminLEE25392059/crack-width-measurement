"""Failure analysis (Task 1 report, Section 6.5).

Turns each method's errors into regions a person can look at and label by
apparent cause, so the Results can say WHICH surface features fool WHICH method:
  * false-positive region: predicted crack pixels more than TOLERANCE_PX from
    any reference crack (8-connected, at least MIN_REGION_PX pixels);
  * missed region (optional): reference crack pixels more than TOLERANCE_PX
    from any predicted crack.
The PER_METHOD largest regions of each method (at most PER_IMAGE from one
photograph) go on a numbered review sheet; a person writes one category per
region into the review CSV; `summarise` counts the categories per method.

Usage:
  python scripts/failure_analysis.py prepare --split test
  python scripts/failure_analysis.py prepare --split test --kinds false_positive,missed
  (fill the "category" column of eval/test/failure_analysis/review_<method>.csv)
  python scripts/failure_analysis.py summarise --split test

Files (eval/<split>/failure_analysis/):
  categories.md                what each category means
  review_<method>.csv          one row per region: item, image, size, context, category, notes
  sheet_<method>_<kind>.png    the numbered crops of one method, for labelling
  crops/<method>_<item>.png    the same crops one by one
  failure_categories.csv       counts and shares per method and category (summarise)
  failure_categories.png       the same counts as heatmaps (summarise)
Author : Seungmin Lee
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config, data, evaluate, utils, viz  # noqa: E402

plt = viz.plt
MIN_REGION_PX = 50      # smaller error regions are boundary noise, not a visible mistake
PER_METHOD = 25         # regions per method and kind that are labelled
PER_IMAGE = 3           # at most this many from one photograph, so one bad image cannot fill a sheet
CROP_PX = 240           # side of a displayed crop, pixels
KINDS = ("false_positive", "missed")
CATEGORIES = {
    "false_positive": {
        "shadow": "a shadow edge or shaded band",
        "stain": "a stain, discolouration, wet patch or paint mark",
        "texture": "surface texture: pores, exposed aggregate, rough patch",
        "joint": "a joint, construction line, formwork mark or other straight edge",
        "annotation": "looks like a real crack that the reference mask left out",
        "other": "anything else (say what in notes)",
    },
    "missed": {
        "thin": "a very thin crack, one or two pixels wide",
        "low_contrast": "a crack hardly darker than the surface, or inside a shadow",
        "annotation": "the reference marks something that does not look like a crack",
        "other": "anything else (say what in notes)",
    },
}
OUTLINE = {"false_positive": viz.OUTCOME_COLORS["FP"], "missed": viz.OUTCOME_COLORS["FN"]}
CONTEXT_OUTLINE = viz.OUTCOME_COLORS["TP"]
REVIEW_COLUMNS = ["item", "method", "kind", "image_id", "area_px", "centre_row", "centre_col",
                  "context", "crop", "category", "notes"]


def out_dir(split: str) -> Path:
    return config.EVAL_DIR / split / "failure_analysis"


def rgb(hex_colour: str) -> tuple[int, int, int]:
    return tuple(int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))


# --- prepare: find the error regions -------------------------------------------------------

def error_regions(pred: np.ndarray, ref: np.ndarray, kind: str) -> tuple[np.ndarray, list[dict]]:
    """Label image of one kind of error region and one dict per region of >= MIN_REGION_PX.

    "context" tells whether the region hangs on a real crack: a false positive
    attached to a predicted crack that also covers the reference is
    over-segmentation; an isolated one is a false detection. A missed region on
    a crack that was partly detected is a partial miss; otherwise the whole
    crack was missed.
    """
    if kind == "false_positive":
        region = pred & ~evaluate.within_distance(ref, config.TOLERANCE_PX)
        own, other = pred, ref
        labels_if = ("attached to a true crack", "isolated false detection")
    else:
        region = ref & ~evaluate.within_distance(pred, config.TOLERANCE_PX)
        own, other = ref, pred
        labels_if = ("part of a detected crack", "whole crack missed")
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(region.astype(np.uint8), connectivity=8)
    _, own_labels = cv2.connectedComponents(own.astype(np.uint8), connectivity=8)
    touching = set(np.unique(own_labels[own & other]).tolist())
    regions = []
    for k in range(1, count):
        area = int(stats[k, cv2.CC_STAT_AREA])
        if area < MIN_REGION_PX:
            continue
        row, col = np.argwhere(labels[stats[k, 1]:stats[k, 1] + stats[k, 3],
                                      stats[k, 0]:stats[k, 0] + stats[k, 2]] == k)[0]
        component = own_labels[stats[k, 1] + row, stats[k, 0] + col]
        regions.append({"label": k, "area_px": area,
                        "centre_row": int(round(centroids[k, 1])), "centre_col": int(round(centroids[k, 0])),
                        "box": (int(stats[k, 1]), int(stats[k, 0]), int(stats[k, 3]), int(stats[k, 2])),
                        "context": labels_if[0] if component in touching else labels_if[1]})
    return labels, regions


def choose(candidates: list[dict]) -> list[dict]:
    """The largest regions, at most PER_IMAGE from one image, PER_METHOD in total."""
    picked, per_image = [], {}
    for region in sorted(candidates, key=lambda r: -r["area_px"]):
        if per_image.get(region["image_id"], 0) < PER_IMAGE:
            picked.append(region)
            per_image[region["image_id"]] = per_image.get(region["image_id"], 0) + 1
        if len(picked) == PER_METHOD:
            break
    return picked


def crop_with_outlines(image: np.ndarray, region: np.ndarray, context: np.ndarray, box: tuple,
                       kind: str) -> np.ndarray:
    """Square crop around a region: thick outline = region to label, thin green = the other mask."""
    top, left, height, width = box
    side = int(min(max(160, 1.6 * max(height, width)), *image.shape[:2]))
    r0 = int(np.clip(top + height // 2 - side // 2, 0, image.shape[0] - side))
    c0 = int(np.clip(left + width // 2 - side // 2, 0, image.shape[1] - side))
    window = (slice(r0, r0 + side), slice(c0, c0 + side))
    crop = cv2.resize(np.ascontiguousarray(image[window]), (CROP_PX, CROP_PX), interpolation=cv2.INTER_AREA)
    for mask, colour, thickness in ((context, CONTEXT_OUTLINE, 1), (region, OUTLINE[kind], 2)):
        small = cv2.resize(np.ascontiguousarray(mask[window], dtype=np.uint8), (CROP_PX, CROP_PX),
                           interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(crop, contours, -1, rgb(colour), thickness)
    return crop


def contact_sheet(crops: list[tuple[int, str, np.ndarray]], title: str, path: Path) -> None:
    """All crops of one method and kind on one numbered page."""
    columns = 5
    rows = max(1, int(np.ceil(len(crops) / columns)))
    fig, axes = plt.subplots(rows, columns, figsize=(2.2 * columns, 2.35 * rows + 0.4), squeeze=False)
    for ax in axes.flat:
        viz.hide_axes(ax)
    for ax, (item, image_id, crop) in zip(axes.flat, crops):
        ax.imshow(crop)
        ax.set_title(f"{item}   (image {image_id})", fontsize=8, fontweight="normal")
    fig.suptitle(title, x=0.01, ha="left", fontsize=10, fontweight="bold", color=viz.INK["primary"])
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def write_categories(folder: Path) -> None:
    lines = ["# Failure categories", "",
             "Write one code from the lists below into the `category` column of each row of",
             "`review_<method>.csv` (Excel is fine; keep the file as CSV). Item numbers match",
             "the numbers on `sheet_<method>_<kind>.png`. Thick outline = the region to label;",
             "thin green outline = the reference crack (false-positive sheets) or the predicted",
             "crack (missed sheets). Use `notes` for anything worth quoting in the Discussion.", ""]
    for kind, codes in CATEGORIES.items():
        lines += [f"## {kind.replace('_', ' ')}", "", "| code | meaning |", "|---|---|"]
        lines += [f"| {code} | {meaning} |" for code, meaning in codes.items()] + [""]
    (folder / "categories.md").write_text("\n".join(lines), encoding="utf-8")


def prepare(split: str, methods: list[str], kinds: list[str], force: bool) -> None:
    folder = out_dir(split)
    (folder / "crops").mkdir(parents=True, exist_ok=True)
    ids = data.load_split()[split]
    methods = [m for m in methods if all(utils.result_path(m, split, i).exists() for i in ids)]
    if not methods:
        sys.exit(f"No method has masks for every {split} image: run scripts/run_method.py first.")
    for method in methods:
        review = folder / f"review_{method}.csv"
        if review.exists() and not force:
            labelled = pd.read_csv(review, dtype=str)["category"].fillna("").str.strip().ne("").any()
            if labelled:
                sys.exit(f"{review.relative_to(ROOT)} already has labels: use --force to replace it.")
    write_categories(folder)

    candidates = {(m, k): [] for m in methods for k in kinds}
    totals = {(m, k): 0 for m in methods for k in kinds}
    for n, image_id in enumerate(ids, 1):
        _, ref = data.load_pair(image_id)
        for method in methods:
            pred = utils.read_mask(utils.result_path(method, split, image_id))
            for kind in kinds:
                _, regions = error_regions(pred, ref, kind)
                for region in regions:
                    region["image_id"] = image_id
                    totals[(method, kind)] += region["area_px"]
                candidates[(method, kind)] += regions
        print(f"\r  scanned {n}/{len(ids)} images", end="", flush=True)
    print()

    for method in methods:
        rows, item = [], 0
        for kind in kinds:
            picked = choose(candidates[(method, kind)])
            sheet = []
            for region in picked:
                item += 1
                image, ref = data.load_pair(region["image_id"])
                pred = utils.read_mask(utils.result_path(method, split, region["image_id"]))
                labels, _ = error_regions(pred, ref, kind)
                crop = crop_with_outlines(image, labels == region["label"],
                                          ref if kind == "false_positive" else pred, region["box"], kind)
                crop_file = f"crops/{method}_{item:03d}.png"
                cv2.imencode(".png", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))[1].tofile(str(folder / crop_file))
                sheet.append((item, region["image_id"], crop))
                rows.append({"item": item, "method": method, "kind": kind, "image_id": region["image_id"],
                             "area_px": region["area_px"], "centre_row": region["centre_row"],
                             "centre_col": region["centre_col"], "context": region["context"],
                             "crop": crop_file, "category": "", "notes": ""})
            sheet_file = folder / f"sheet_{method}_{kind}.png"
            sheet_file.unlink(missing_ok=True)
            if not picked:
                print(f"Method {method}, {kind}: no region of {MIN_REGION_PX} px or more")
                continue
            covered = sum(r["area_px"] for r in picked)
            share = covered / totals[(method, kind)] if totals[(method, kind)] else 0.0
            title = (f"Method {method}: {len(picked)} largest {kind.replace('_', ' ')} regions "
                     f"({share:.0%} of all such pixels). Label them in review_{method}.csv")
            contact_sheet(sheet, title, sheet_file)
            print(f"Method {method}, {kind}: {len(candidates[(method, kind)])} regions >= {MIN_REGION_PX} px; "
                  f"{len(picked)} on the sheet, covering {share:.0%} of their pixels")
        pd.DataFrame(rows, columns=REVIEW_COLUMNS).to_csv(folder / f"review_{method}.csv", index=False,
                                                          encoding="utf-8-sig")
    print(f"Review files in {folder.relative_to(ROOT)}: fill the category column, then run summarise.")


# --- summarise: count the labels -----------------------------------------------------------------

def read_review(path: Path) -> pd.DataFrame:
    """A review file as saved by prepare, or after Excel re-saved it in a local encoding."""
    try:
        return pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, dtype=str, encoding="latin-1")   # category codes are plain ASCII


def summarise(split: str) -> None:
    folder = out_dir(split)
    files = sorted(folder.glob("review_*.csv"))
    if not files:
        sys.exit(f"No review files in {folder.relative_to(ROOT)}: run prepare first.")
    reviews = pd.concat([read_review(f) for f in files], ignore_index=True)
    reviews["category"] = reviews["category"].fillna("").str.strip().str.lower()
    reviews["valid"] = [category in CATEGORIES.get(kind, {}) for kind, category
                        in zip(reviews["kind"], reviews["category"])]
    for _, row in reviews[(reviews["category"] != "") & ~reviews["valid"]].iterrows():
        print(f"Unknown category '{row['category']}' for Method {row['method']} item {row['item']} "
              f"({row['kind']}): see categories.md")
    blank = reviews[reviews["category"] == ""].groupby("method").size()
    for method, count in blank.items():
        print(f"Method {method}: {count} item(s) not labelled yet")
    labelled = reviews[reviews["valid"]]
    if labelled.empty:
        sys.exit("No valid labels yet.")

    viz.apply_style()
    rows, panels = [], []
    for kind in [k for k in KINDS if k in set(labelled["kind"])]:
        group = labelled[labelled["kind"] == kind]
        counts = pd.crosstab(group["method"], group["category"]).reindex(columns=list(CATEGORIES[kind]),
                                                                          fill_value=0)
        panels.append((kind, counts))
        for method, line in counts.iterrows():
            total = int(line.sum())
            rows += [{"method": method, "kind": kind, "category": category, "count": int(count),
                      "share": count / total if total else 0.0, "labelled": total}
                     for category, count in line.items()]
    table = pd.DataFrame(rows)
    table.to_csv(folder / "failure_categories.csv", index=False, float_format="%.3f")

    height = 0.45 * max(len(counts) for _, counts in panels) + 1.8
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), height), squeeze=False,
                             gridspec_kw={"wspace": 0.45})
    for ax, (kind, counts) in zip(axes[0], panels):
        viz.plot_confusion(ax, counts.to_numpy(), [f"Method {m}" for m in counts.index], list(counts.columns),
                           f"{kind.replace('_', ' ').capitalize()} regions by cause")
        ax.set_xlabel("Apparent cause")
    viz.save_figure(fig, folder / "failure_categories.png")
    with pd.option_context("display.width", 140):
        for kind, counts in panels:
            print(f"\n{kind.replace('_', ' ')} regions (counts)\n{counts.to_string()}")
    print(f"\nSaved {(folder / 'failure_categories.csv').relative_to(ROOT)} and failure_categories.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    step = commands.add_parser("prepare", help="make the review sheets")
    step.add_argument("--split", choices=("val", "test"), default="test")
    step.add_argument("--methods", default=",".join(config.METHODS))
    step.add_argument("--kinds", default="false_positive", help="false_positive and/or missed, comma-separated")
    step.add_argument("--force", action="store_true", help="replace review files that already have labels")
    step = commands.add_parser("summarise", help="count the labels")
    step.add_argument("--split", choices=("val", "test"), default="test")
    args = parser.parse_args()

    utils.set_seed()
    viz.apply_style()
    if args.command == "prepare":
        kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
        unknown = set(kinds) - set(KINDS)
        if unknown:
            parser.error(f"unknown kind(s): {', '.join(sorted(unknown))}")
        prepare(args.split, [m.strip() for m in args.methods.split(",") if m.strip()], kinds, args.force)
    else:
        summarise(args.split)


if __name__ == "__main__":
    main()