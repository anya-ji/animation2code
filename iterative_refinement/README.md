# Iterative Refinement

An iterative loop that generates animation code from a video, records it, visually critiques it against the ground truth, and refines the code — repeating for N rounds.

## Pipeline

```
ground truth video
       │
       ▼
  [generator]  →  full.html  →  [recorder]  →  iter_0/animation.mp4
                                                       │
       ┌───────────────────────────────────────────────┘
       ▼
  [critic]  (gt video + iter_N video)  →  feedback.txt
       │
       ▼
  [editor]  (feedback + current code)  →  full.html
       │
       ▼
  [recorder]  →  iter_N+1/animation.mp4  →  repeat
```

Each stage uses a configurable model (`gemini` or `qwen3vl`).

## Setup

```bash
pip install google-genai transformers torch qwen-vl-utils python-dotenv pyyaml tqdm
```

Set `GOOGLE_API_KEY` in `.env` if using Gemini.

For the recorder, install Node.js dependencies:
```bash
npm install
```

## Usage

```bash
# Baseline pipeline
python run.py --ids codepen-ABKVZX-AbRrvV
python run.py --start 0 --end 20

# Experiment variants
python run_experiment.py baseline --ids codepen-ABKVZX-AbRrvV
python run_experiment.py describe_first --ids codepen-ABKVZX-AbRrvV
python run_experiment.py dual_critic --ids codepen-ABKVZX-AbRrvV
python run_experiment.py caption_bridge --ids codepen-ABKVZX-AbRrvV

# Run all four variants on the same examples
python run_experiment.py all --ids codepen-ABKVZX-AbRrvV
```

## Variants

| Name | Description |
|------|-------------|
| `baseline` | Standard generate → critique → edit loop |
| `describe_first` | Describe animation in natural language before generating code (ChartIR-inspired) |
| `dual_critic` | Separate visual critic + code critic, merge feedback (METAL-inspired) |
| `caption_bridge` | Caption both videos as text, compare captions, edit from diff (PhyT2V-inspired) |

## Configuration

Copy `configs/config.yaml` (Gemini) or `configs/config_qwen.yaml` (Qwen3-VL) and edit:

| Key | Description |
|-----|-------------|
| `max_iterations` | Refinement rounds after iter_0 |
| `generator.model` / `critic.model` / `editor.model` | `gemini` or `qwen3vl` |
| `recorder.fps` / `recorder.duration` | Recording parameters |

## Visualizer

```bash
cd visualizer && node server.cjs
# open http://localhost:8765
```
