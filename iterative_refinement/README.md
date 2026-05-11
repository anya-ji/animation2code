# Iterative Refinement

An iterative loop that generates animation code from a video, records it, visually critiques it against the ground truth, and refines the code — repeating for N rounds.

## Pipelines

Each stage uses a configurable model (`gemini` or `qwen3vl`).

### baseline

```
gt video
   │
   ▼
[generator] ──→ full.html ──→ [recorder] ──→ iter_0/animation.mp4
                                                      │
     ┌────────────────────────────────────────────────┘
     ▼
[critic]  (gt video + iter_N video) ──→ feedback.txt
     │
     ▼
[editor]  (current code + feedback) ──→ full.html
     │
     ▼
[recorder] ──→ iter_N+1/animation.mp4  →  repeat
```

### describe_first (ChartIR-based)

iter_0 decouples visual understanding from code generation:

```
gt video
   │
   ├──→ [generator/describe] ──→ description.txt
   │                                    │
   └──────────────────────────────┐     │
                                  ▼     ▼
                           [generator/code]  (video + description) ──→ full.html
                                  │
                                  ▼
                            [recorder] ──→ iter_0/animation.mp4
```

Refinement uses the original description as context for the critic:

```
[critic]  (gt video + iter_N video + original description) ──→ diff feedback
     │
     ▼
[editor]  (current code + diff feedback + description) ──→ full.html
     │
     ▼
[recorder] ──→ iter_N+1/animation.mp4  →  repeat
```

### dual_critic (METAL-based)

iter_0 is the same as baseline. Refinement runs two critic passes:

```
[visual critic]  (gt video + iter_N video) ──→ visual_feedback.txt
     │
     ▼
[code critic]   (visual feedback + current code, text only) ──→ code_feedback.txt
     │
     ▼
[editor]  (current code + visual feedback + code feedback) ──→ full.html
     │
     ▼
[recorder] ──→ iter_N+1/animation.mp4  →  repeat
```

### caption_bridge (PhyT2V-based)

Target is captioned once before iter_0. Refinement compares captions as text (no video in the compare step):

```
gt video ──→ [captioner] ──→ target_caption.txt  (once, reused every iteration)
```

iter_0 is the same as baseline. Refinement:

```
iter_N video ──→ [captioner] ──→ generated_caption.txt
                                          │
              target_caption ─────────────┤
                                          ▼
                              [compare]  (text only) ──→ feedback.txt
                                          │
                                          ▼
                              [editor]  (current code + feedback + target caption)
                                          │
                                          ▼
                              [recorder] ──→ iter_N+1/animation.mp4  →  repeat
```

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
| `describe_first` | Describe animation in natural language before generating code (ChartIR-based) |
| `dual_critic` | Separate visual critic + code critic, merge feedback (METAL-based) |
| `caption_bridge` | Caption both videos as text, compare captions, edit from diff (PhyT2V-based) |

## Configuration

Copy `configs/config.yaml` (Gemini) or `configs/config_qwen.yaml` (Qwen3-VL) and edit:

| Key | Description |
|-----|-------------|
| `max_iterations` | Refinement rounds after iter_0 |
| `generator.model` / `critic.model` / `editor.model` | `gemini` or `qwen3vl` |
| `recorder.fps` / `recorder.duration` | Recording parameters |

