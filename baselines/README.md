# Baselines

Inference pipelines for all baseline models on the Animation2Code test set.

## Models

| File | Model | Input |
|------|-------|-------|
| `models/gemini.py` | Google Gemini (API) | video or frames |
| `models/gpt.py` | OpenAI GPT (API) | frames |
| `models/openrouter.py` | OpenRouter (API) | frames |
| `models/qwen3vl.py` | Qwen3-VL-8B (local GPU) | video or frames |
| `models/molmo.py` | Molmo (local GPU) | video |

## Setup

```bash
pip install pyyaml tqdm google-genai openai python-dotenv torch transformers accelerate qwen-vl-utils av
```

Set API keys in `.env`:
```
GEMINI_API_KEY=...
OPENAI_API_KEY=...
OPENROUTER_API_KEY=...
```

For recording, install Node.js dependencies in the `recording/` directory:
```bash
cd recording && npm install
```

## Running inference

```bash
# From baselines/
python run.py --config config/gemini.yaml
python run.py --config config/qwen3vl.yaml

# Subset
python run.py --config config/gemini.yaml --start 0 --end 10
python run.py --config config/gemini.yaml --ids codepen-xxx codepen-yyy

# Resume an existing run
python run.py --config config/gemini.yaml --output-dir gemini/output/20260318_123456

# Multi-GPU for Qwen
CUDA_VISIBLE_DEVICES=0,1 python run.py --config config/qwen3vl.yaml
```

Output per run: `<model>/output/<timestamp>/` with `_config.yaml`, `_run.log`, and per-example `full.html` + `_raw_output.txt`.

## Inference + recording in one step

```bash
python pipeline.py --config config/gemini.yaml
```

## Recording only (for existing HTML outputs)

```bash
python record.py --input gemini/output/20260318_123456 \
    --recorder ../recording/record.js \
    --fps 30 --concurrency 4
```
