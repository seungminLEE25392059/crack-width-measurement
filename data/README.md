# Dataset (not in Git)

Download the Özgenel (2019b) concrete crack segmentation dataset and arrange it as:

```
data/raw/images/<id>.jpg    original photographs
data/raw/masks/<id>.jpg     crack masks with the same file names
```

Rename the dataset's own folders if they are called something else. Only this
README is tracked; everything else in `data/` is ignored by Git.

On Colab, set the `CRACK_DATA_DIR` environment variable to the Google Drive
folder that contains `images/` and `masks/` before importing `src`.
