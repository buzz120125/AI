#!/usr/bin/env python3
"""Generate high-quality videos from text prompts using Diffusers.

Example:
  python video_ai.py \
    --prompt "cinematic drone shot over neon cyberpunk city at night, volumetric fog" \
    --model-id THUDM/CogVideoX-2b \
    --num-frames 49 \
    --fps 8 \
    --output outputs/cyberpunk.mp4
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI video generation from a text prompt (Diffusers + Hugging Face models)."
    )
    parser.add_argument("--prompt", required=True, help="Text prompt describing the video.")
    parser.add_argument(
        "--negative-prompt",
        default="blurry, low quality, artifacts, distorted faces, flicker",
        help="What to avoid in the generated video.",
    )
    parser.add_argument(
        "--model-id",
        default="THUDM/CogVideoX-2b",
        help="Hugging Face model ID for a text-to-video diffusion model.",
    )
    parser.add_argument("--output", default="outputs/generated.mp4", help="Output MP4 file path.")
    parser.add_argument("--width", type=int, default=720, help="Video width.")
    parser.add_argument("--height", type=int, default=480, help="Video height.")
    parser.add_argument("--num-frames", type=int, default=49, help="Number of video frames.")
    parser.add_argument("--fps", type=int, default=8, help="Frames per second in output video.")
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="Diffusion denoising steps (higher = usually better quality, slower).",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="Prompt guidance scale (higher = stronger prompt adherence).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="Enable model CPU offload to reduce VRAM usage (slower).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # For private/gated models, set: export HUGGING_FACE_HUB_TOKEN=...
    token = os.getenv("HUGGING_FACE_HUB_TOKEN")

    print(f"Loading model '{args.model_id}' on {device} ({dtype})...")
    pipe = DiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        token=token,
    )

    if args.cpu_offload and device == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    generator = torch.Generator(device=device).manual_seed(args.seed)

    print("Generating video... this can take several minutes.")
    result = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    )

    frames = result.frames[0]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(output_path), fps=args.fps)

    print(f"Done. Video saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
