# Baselines

Zero-shot video-to-code inference on the test set, using the settings from the paper (2 fps input).

## Setup

```bash
pip install -r requirements.txt
cd ../recording && npm install && cd ../baselines
```

Put API keys in `.env` at the repo root: `GOOGLE_API_KEY`, `OPENAI_API_KEY` and `OPENROUTER_API_KEY`.

## Run

Choose a config: `gemini`, `gpt`, `claude`, `llama` or `qwen3vl`. Qwen3-VL runs locally on a GPU; the others call an API.

```bash
# Inference and recording
python pipeline.py --config config/gemini.yaml

# Inference only, on a subset
python run.py --config config/gemini.yaml --start 0 --end 10
python run.py --config config/gemini.yaml --ids <id> <id>

# Resume a run
python run.py --config config/gemini.yaml --output-dir <run_dir>

# Record an existing run at 30 fps
python record.py --input <run_dir>
```

Each run writes `<model>/output/<timestamp>_2fps/<id>/` with `full.html`, `_raw_output.txt` and `animation.mp4`, or `_error.txt` on failure.

To change the input rate, edit `fps` in the config (`image_sampling.fps` for GPT, Claude and LLaMA).
