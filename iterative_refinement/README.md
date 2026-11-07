# Iterative Refinement

Generates code from a video, records it, critiques the result against the target and edits the code, for a set number of rounds.

| Variant | What it does |
|---|---|
| `baseline` | Generate, critique, edit |
| `describe_first` | Describe the animation first, then generate and critique against that description (ChartIR) |
| `dual_critic` | Visual critic, then code critic, then edit (METAL) |
| `caption_bridge` | Caption both videos and compare the captions as text (PhyT2V) |

## Setup

```bash
pip install -r requirements.txt
npm install
```

Put `GOOGLE_API_KEY` in `.env` to use Gemini. `configs/config_qwen.yaml` runs Qwen3-VL locally and needs no key.

## Run

```bash
python run_experiment.py baseline --ids <id>
python run_experiment.py dual_critic --config configs/config_qwen.yaml --start 0 --end 20
python run_experiment.py all --ids <id> --max-iterations 3
```

Each run writes `output/<variant>_<timestamp>/<id>/iter_N/` with `full.html` and `animation.mp4`. Pass `--run-dir` to resume a run or start from an existing `iter_0`.

Each stage (`generator`, `critic`, `editor`) picks its model in the config. Videos are recorded at 12 fps during refinement, so re-record the final HTML at 30 fps with `baselines/record.py` before scoring.
