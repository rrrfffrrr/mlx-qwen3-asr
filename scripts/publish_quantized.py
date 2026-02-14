#!/usr/bin/env python3
"""Convert Qwen3-ASR weights and publish quantized artifacts to HuggingFace.

Example:
    python scripts/publish_quantized.py \
      --source-model Qwen/Qwen3-ASR-0.6B \
      --repo-id moona3k/mlx-qwen3-asr-0.6b-4bit \
      --bits 4
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

from huggingface_hub import HfApi


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert and upload quantized MLX Qwen3-ASR model to HuggingFace"
    )
    parser.add_argument(
        "--source-model",
        default="Qwen/Qwen3-ASR-0.6B",
        help="Source HuggingFace model ID or local model path",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Target HuggingFace repo ID, e.g. moona3k/mlx-qwen3-asr-0.6b-4bit",
    )
    parser.add_argument(
        "--bits",
        type=int,
        choices=[4, 8],
        required=True,
        help="Quantization bits",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=64,
        help="Quantization group size",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Intermediate conversion dtype",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the target repo as private",
    )
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("Set HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) before publishing.")

    project_root = Path(__file__).resolve().parents[1]
    convert_script = project_root / "scripts" / "convert.py"

    with tempfile.TemporaryDirectory(prefix="mlx-qwen3-asr-quant-") as tmpdir:
        out_dir = Path(tmpdir) / "model"

        _run(
            [
                "python",
                str(convert_script),
                "--model",
                args.source_model,
                "--output-dir",
                str(out_dir),
                "--quantize",
                str(args.bits),
                "--group-size",
                str(args.group_size),
                "--dtype",
                args.dtype,
            ],
            cwd=project_root,
        )

        readme = out_dir / "README.md"
        readme.write_text(
            (
                f"# {args.repo_id}\n\n"
                f"Quantized MLX conversion of `{args.source_model}`.\n\n"
                f"- bits: {args.bits}\n"
                f"- group_size: {args.group_size}\n"
                f"- dtype: {args.dtype}\n"
            ),
            encoding="utf-8",
        )

        api = HfApi(token=token)
        api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)
        api.upload_folder(
            folder_path=str(out_dir),
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=f"Add {args.bits}-bit MLX quantized weights from {args.source_model}",
        )
        print(f"Uploaded quantized model to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
