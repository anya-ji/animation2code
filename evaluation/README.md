# Evaluation

Methods for comparing a rendered animation video against a gold reference video.

All commands are run from the repo root using `uv run python` (or `python` directly).

DTW alignment is on by default. Use `--without-dtw` to disable it.

## Methods

| Method | Description |
|--------|-------------|
| `precision` | Similarity within the render's animated bounding box |
| `recall` | Similarity within the gold's animated bounding box |
| `fullframe` | Similarity over the full frame |
| `bbox` | Each video cropped to its own animated bbox, then compared |
| `temporal_fidelity` | CoTracker motion direction cosine similarity |
| `tsed` | HTML AST edit distance (code similarity) |
| `tsed_tripart` | TSED with tripartite structure weighting (HTML/CSS/JS) |

All bounding box methods (`precision`, `recall`, `fullframe`, `bbox`) support `--sim ssim` (default), `--sim clip`, or `--sim dreamsim`. DreamSim requires `pip install dreamsim`.

## Single-method

```bash
uv run python evaluation/methods/precision.py <render.mp4> <gold.mp4>
uv run python evaluation/methods/recall.py    <render.mp4> <gold.mp4>
uv run python evaluation/methods/bbox.py      <render.mp4> <gold.mp4>
uv run python evaluation/methods/tsed.py      <render.html> <gold.html>
```

Output is JSON to stdout.

## Batch evaluation

```bash
uv run python evaluation/run_eval.py \
    --result-dir <result_dir> \
    --gold-dir   <gold_dir>   \
    --method     precision,recall,fullframe,bbox \
    --outpath    results.csv

# Code similarity
uv run python evaluation/extract_gold_html.py --metadata dataset/metadata.jsonl --out evaluation/gold_html
uv run python evaluation/run_eval.py \
    --result-dir <result_dir> \
    --gold-html-dir evaluation/gold_html \
    --method tsed --outpath results.csv
```

Expected layout:
```
result_dir/
  codepen-ABCD-1234/animation.mp4
gold_dir/
  codepen-ABCD-1234/animation.mp4
```
