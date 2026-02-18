# AI Video Generator (Runnable)

This project provides a runnable Python script that generates **high-quality AI videos from a text prompt** using Hugging Face Diffusers video models.

## 1) Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## 2) Run

```bash
python video_ai.py \
  --prompt "A cinematic tracking shot of a futuristic city at sunrise, ultra-detailed, 4k" \
  --model-id THUDM/CogVideoX-2b \
  --width 720 \
  --height 480 \
  --num-frames 49 \
  --num-inference-steps 50 \
  --guidance-scale 7.5 \
  --fps 8 \
  --output outputs/city.mp4
```

## Notes for high quality

- Use a **GPU** (NVIDIA CUDA) for best results and speed.
- Increase `--num-inference-steps` for better quality (but slower).
- Improve prompt detail (camera type, lighting, motion, style keywords).
- For stronger visual quality, test larger models when your hardware can support them.

## Optional: gated/private model access

If a model requires auth, set your token first:

```bash
export HUGGING_FACE_HUB_TOKEN=your_token_here
```

## Helpful alternatives

Try other model IDs with `--model-id`, for example:

- `THUDM/CogVideoX-2b`
- `THUDM/CogVideoX-5b` (heavier, potentially higher quality)

