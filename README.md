# Crack Width Measurement (31256/42177 Task 2)

Python implementation of our Task 1 project, *From Pixels to Millimetres*. Four
segmentation methods (A to D) produce binary crack masks; one shared measurement
stage turns masks into crack length and width; one evaluation script produces
every table and figure for the report.

| Method | Approach | Owner |
|---|---|---|
| A | Traditional image processing | Wenjie |
| B | HOG + GLCM + SVM, patch-based | Bowen |
| C | U-Net | Lingyu |
| D | U-Net + post-processing | Lingyu |

## 1. Setup (once per computer)

1. Clone the repository: VS Code → Source Control → **Clone Repository**, or
   `git clone https://github.com/<owner>/crack-width-measurement.git`
2. Create a virtual environment and install the packages (Python 3.10 or later):

   ```bash
   # Windows (PowerShell)
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt

   # macOS / Linux
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

   If PowerShell refuses to run `activate`, run
   `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or pick the
   `.venv` interpreter in VS Code (Ctrl+Shift+P → *Python: Select Interpreter*).
   PyTorch is only needed for Method C: Colab already has it; for a local
   install see https://pytorch.org/get-started/locally/.
3. Download the Özgenel (2019b) segmentation dataset into `data/raw/images/`
   and `data/raw/masks/` (details in `data/README.md`). The data never goes into Git.
4. Check that everything is wired up:

   ```bash
   python scripts/check_interfaces.py
   ```

## 2. Who owns which file

| Files | Owner |
|---|---|
| `src/data.py`, `scripts/make_split.py` | Zitong |
| `src/method_a.py` | Wenjie |
| `src/method_b.py`, `scripts/train_b.py` | Bowen |
| `src/method_c.py`, `src/method_d.py`, `notebooks/train_unet_colab.ipynb` | Lingyu |
| `src/measure.py`, `scripts/run_resolution.py` | Junsen |
| `src/config.py`, `src/utils.py`, `src/evaluate.py`, `src/viz.py`, `scripts/run_method.py`, `scripts/run_evaluation.py`, `scripts/check_interfaces.py` | Seungmin |
| `log/decisions.md` | everyone |

## 3. Folder structure

```
src/         library code, one module per owner
scripts/     command-line entry points (run them from the repository root)
notebooks/   Colab notebook for U-Net training
splits/      splits.json: the fixed train/val/test image IDs (committed)
params/      validation-selected parameters, one JSON per method (committed)
eval/        report tables and figures written by run_evaluation.py
log/         decisions.md (challenges and decisions) and environment.txt
data/        dataset (not in Git)
results/     masks written by run_method.py (not in Git)
models/      trained SVM and U-Net (not in Git; share through Google Drive)
```

## 4. Function interfaces (fixed: do not rename or change inputs and outputs)

| Function | Returns | Owner |
|---|---|---|
| `data.load_pair(image_id)` | image RGB uint8 (H, W, 3), mask bool (H, W) | Zitong |
| `data.make_split(ids)` | `{"train", "val", "test"}` lists of image IDs | Zitong |
| `data.make_tiles(array)`, `data.stitch_tiles(maps, origins, shape)` | tiles, origins, valid / stitched (H, W) map | Zitong |
| `method_a.segment_a(image, params)` | mask bool (H, W) | Wenjie |
| `method_a.filter_components(mask, min_area, min_elongation)` | mask bool (H, W) | Wenjie |
| `method_b.segment_b(image, model, params)` | mask bool, confidence float32 in [0, 1] | Bowen |
| `method_c.segment_c(image, net, params)` | mask bool, probability float32 in [0, 1] | Lingyu |
| `method_d.refine_d(mask_c, params)` | mask bool (H, W) | Lingyu |
| `measure.measure_cracks(mask)` | list of crack dicts, label image | Junsen |
| `measure.severity_thresholds(max_widths)` | (low, high) in pixels | Junsen |
| `evaluate.seg_metrics(pred, ref)` | precision, recall, F1, IoU, tolerance F1 | Seungmin |
| `evaluate.match_cracks(ref_labels, pred_labels)` | {reference id: predicted id or None} | Seungmin |

Each docstring gives the exact inputs, outputs and what is still to do (`TODO(name)`).

## 5. Rules

- **Own files only.** Changes to shared files (`src/config.py`, `src/utils.py`,
  `scripts/`) go through Seungmin.
- **Placeholders.** Functions marked `@placeholder` are stand-ins that return
  the agreed types, so the whole pipeline runs from day one. Replace the body
  with your method (or confirm it), then delete the decorator.
  `check_interfaces.py` lists every placeholder that is left.
- **Validation only.** Tune on `val` and save the chosen values with
  `utils.save_params("A", params)` (writes `params/A.json`). The `test` split is
  run exactly once, on Sun 18 Oct, with `--final`.
- **Masks.** Boolean, same height and width as the image, crack = True. Saved
  as PNG (0 = background, 255 = crack).
- **Reproducibility.** `utils.set_seed()` (seed 42) at the start of every
  script and notebook.
- **Comments.** Every file starts with a docstring (purpose, inputs, outputs,
  author). The code goes into the report appendix as written.
- **Log.** Add a row to `log/decisions.md` whenever something fails or the plan
  changes; the Discussion section is written from it.

## 6. Git workflow

```bash
git pull                                  # before you start
git add src/method_a.py                   # only your own files
git commit -m "method_a: add CLAHE step"
git push                                  # when a piece of work is done
```

Never commit data, results or model weights; put large files in the team
Google Drive folder.

## 7. Running the pipeline

```bash
python scripts/check_interfaces.py                    # any time, before every push
python scripts/make_split.py                          # once (Zitong) -> splits/splits.json
python scripts/train_b.py                             # Bowen -> models/method_b.joblib
python scripts/run_method.py --method A --split val   # same for B and C
python scripts/run_method.py --method D --split val   # needs Method C masks first
python scripts/run_evaluation.py --split val          # tables and figures in eval/val/

# Final run, once, on Sun 18 Oct
python scripts/run_method.py --method A --split test --final    # then B, C, D
python scripts/run_evaluation.py --split test --examples id1,id2,id3
```

Useful options: `--limit 5` (first five images only), `--examples` (choose the
overlay images, e.g. a typical, a thin and a shadow/stain case) and
`--recompute-severity` (after `src/measure.py` changes).

## 8. Colab (Method C training)

1. GitHub → Settings → Developer settings → Personal access tokens →
   **Tokens (classic)** → generate a token with the `repo` scope.
2. In Colab, open the key icon (Secrets), add it as `GITHUB_TOKEN` and allow
   notebook access.
3. Put the dataset on Google Drive, open `notebooks/train_unet_colab.ipynb`
   (File → Open notebook → GitHub, tick *Include private repos*) and set
   Runtime → Change runtime type → **GPU**.
4. Share the trained `unet.pt` through Google Drive and commit `params/C.json`.
