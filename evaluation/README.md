# Evaluation

Scores rendered animations against the gold videos, including the paper's appearance and temporal metrics.

## Setup

```bash
pip install -r requirements.txt
```

## Run

`<run_dir>` contains `<id>/animation.mp4`, and `--gold-dir` is a dataset split. Add `--render-filename NAME` if your videos use another filename.

```bash
# Appearance similarity: DreamSim on the animated crop, aligned with DTW
python run_eval.py --result-dir <run_dir> --gold-dir ../anim2code_data/data/test \
    --method bbox --sim dreamsim --outpath appearance.json --format json

# Temporal similarity: CoTracker3 tracklets
python run_eval.py --result-dir <run_dir> --gold-dir ../anim2code_data/data/test \
    --method temporal_fidelity --outpath temporal.json --format json

# Code similarity, using <id>/full.html
python run_eval.py --result-dir <run_dir> --gold-html-dir ../anim2code_data/data/test \
    --method tsed --outpath tsed.json --format json
```

Finished pairs are skipped when you rerun with the same `--outpath`. Add `--overwrite` to start over.

`temporal_fidelity` uses every frame by default. Add `--mf-max-frames 120` to subsample each video to 120 frames, as in the zero-shot and refinement results.

To score one pair, which prints JSON:

```bash
python methods/bbox.py <render.mp4> <gold.mp4> --sim dreamsim
python methods/temporal_fidelity.py <render.mp4> <gold.mp4>
```

## Methods

| `--method` | Input | Measures |
|---|---|---|
| `bbox` | Video | Similarity with each video cropped to its own animated square. **Appearance similarity** in paper with `--sim dreamsim` |
| `temporal_fidelity` | Video | Direction and speed agreement of CoTracker3 tracks. **Temporal similarity** in paper |
| `precision` | Video | Similarity inside the render's animated square |
| `recall` | Video | Similarity inside the gold's animated square |
| `fullframe` | Video | Similarity over the full frame |
| `tsed` | HTML | AST tree edit similarity of the whole document |
| `tsed_tripart` | HTML | `tsed` on the HTML, CSS and JS separately, weighted by size |

`bbox`, `precision`, `recall` and `fullframe` take `--sim ssim` (default), `clip` or `dreamsim`, and align frames with DTW (`methods/dtw.py`) unless you pass `--without-dtw`.

A render with no animated pixels scores 0 on `temporal_fidelity` and is marked `skip_reason: no_animated_pixels_pred`.
