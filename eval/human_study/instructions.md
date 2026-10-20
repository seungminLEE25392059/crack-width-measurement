# Human width study: instructions

Thank you for taking part. This is the human comparison of the Task 1 report
(Section 6.6). It takes about 20-30 minutes.

## Rules

1. Work alone. Until everyone has finished, do not open the reference masks
   (`data/raw/masks`), the method outputs (`results/`), the evaluation tables
   (`eval/val`, `eval/test`) or anyone else's answers.
2. Measure the 20 images in the order the tool shows them.

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
   | 1 | minor | below 20.0 px |
   | 2 | moderate | 20.0 px up to 56.3 px |
   | 3 | severe | 56.3 px or more |
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
