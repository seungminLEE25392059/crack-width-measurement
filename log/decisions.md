# Challenges and decisions log

Add one row whenever something fails, a setting is chosen, or the plan changes.
The Discussion & Challenges section (goals met and why, challenges and plan
changes, future work, what we learned) is written from this table.

| Date | Member | What happened | Decision | Effect on results |
|---|---|---|---|---|
| 2026-10-15 | Team | Task 1 planned MATLAB, but MATLAB Online has no GPU for U-Net training | Switched to Python with teaching-staff approval; U-Net trained on a Colab GPU | Overview states the environment; Discussion explains the change |
# Challenges and decisions log

Add one row whenever something fails, a setting is chosen, or the plan changes.
The Discussion & Challenges section (goals met and why, challenges and plan
changes, future work, what we learned) is written from this table.

| Date | Member | What happened | Decision | Effect on results |
|---|---|---|---|---|
| 2026-10-15 | Team | Task 1 planned MATLAB, but MATLAB Online has no GPU for U-Net training | Switched to Python with teaching-staff approval; U-Net trained on a Colab GPU | Overview states the environment; Discussion explains the change |
| 2026-10-16 | Wenjie | Method A was planned in MATLAB; the shared split and measurement code arrived in Python | Re-implemented Method A in Python (OpenCV/SciPy), same pipeline as Task 1 Section 5.2 | Method A started one day late; no change to the method itself |
| 2026-10-20 | Wenjie | A full grid over every Section 5.2 option was too slow on 12-MP photographs | Staged search on the 69 validation images (structure → enhancement/top-hat → threshold → clean-up), each stage starting from the previous best | Search finished in about 50 min on 2 cores; stage tables in eval/val/method_a/ |
| 2026-10-20 | Wenjie | Global Otsu on the plain image failed: cracks are ~1.4 % of pixels, so Otsu split the background texture (val mean F1 0.12) | Added a black top-hat crack response (Week 3 morphology) before Otsu; chosen on validation (radius 31; 41 and 51 checked and worse) | Val mean F1 rose from 0.12 to 0.69; this is the largest single effect in the ablation |
| 2026-10-20 | Wenjie | Planned contrast enhancement hurt: histeq and CLAHE amplified surface texture together with the cracks (val F1 0.12 and 0.30) | Enhancement switched off; the median filter (7×7) is kept (removing it costs 0.12 F1) | Method A differs from the Task 1 plan; reported as a finding, not hidden |
| 2026-10-20 | Wenjie | The major/minor axis ratio can delete curved or branched cracks (near-round equivalent ellipse) | Added an optional "thinness" rule (boundary²/4·area) to filter_components; validation still chose the ellipse rule (thinness 0.033 F1 lower) | Ellipse rule kept as planned; thinness available to Method D |
| 2026-10-20 | Wenjie | Mean per-image precision is 0.89 but pooled precision only 0.59: a few textured images (e.g. 040, 466) produce very large false-positive areas | Both averages reported; textured surfaces flagged for the failure analysis | Explains why Method A's pooled scores are much lower than its per-image scores |