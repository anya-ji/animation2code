# Recording

Records `full.html` files to `animation.mp4` with Puppeteer on a deterministic virtual clock (1024×768, 30 fps, up to 8 s).

## Setup

Requires Node.js 18+ and ffmpeg.

```bash
npm install
pip install -r requirements.txt
```

## Run

```bash
# All test examples in anim2code_data/, written to output/run1/
node record.js --js-inject

# One example, or a custom run name and parallelism
node record.js <id> --js-inject
node record.js --output run2 --concurrency 4 --js-inject

# Any folder of <id>/full.html, writing videos next to them
node record.js --html-dir <dir> --js-inject
```

`--js-inject` mocks JavaScript time and `Math.random`, which makes runs of JavaScript animations repeatable. Add `--rewrite` to re-record finished examples or `--fps N` to change the frame rate.

## Quality checks

```bash
# Flag videos with no motion, written to static_check/static_videos.json
RUN_DIR=output/run1 python static_check/check_static.py
RUN_DIR=output/run1 python static_check/dashboard.py    # http://localhost:8080

# Compare two independent runs, written to determinism_check/determinism_comparison.csv
python determinism_check/check_deterministic.py --runs output/run1 output/run2
python determinism_check/dashboard.py                   # http://localhost:5500
```
