# Recording Pipeline

Records `full.html` animation files into `animation.mp4` using Puppeteer with a deterministic virtual clock.

## Setup

```bash
npm install
```

Requires Node.js, ffmpeg, and a Chromium installation accessible by Puppeteer.

## Usage

```bash
# Record all examples in dataset/
node record.js

# Record a single example
node record.js codepen-ABCD-1234

# Custom output folder and concurrency
node record.js --output run1 --concurrency 4

# Re-record all (ignore resume state)
node record.js --rewrite

# Enable deterministic JS time mocking
node record.js --js-inject
```

Input: `dataset/<example>/full.html`
Output: `output/<run_name>/<example>/animation.mp4`

Default: `1024×768`, 30 fps, up to 8s duration.

## Quality checks

### Static detection

```bash
python static_check/check_static.py --run-dir output/run1
python static_check/dashboard.py --run-dir output/run1  # web UI
```

Flags videos where no pixels change between frames.

### Determinism check

Run two independent recordings, then compare:

```bash
node record.js --output run1
node record.js --output run2
python determinism_check/check_deterministic.py --run1 output/run1 --run2 output/run2
python determinism_check/dashboard.py           # web UI
python determinism_check/rerun_false.py         # re-record non-deterministic examples
```
