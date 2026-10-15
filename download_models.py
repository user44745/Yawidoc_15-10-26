#!/usr/bin/env python
"""
download_models.py — one-time download of the models used by the two
translator characters ("pattani_thai_translator", "thai_pattani_translator").

  1. Whisper large-v3-turbo (CTranslate2) -> ./models/whisper
     speech recognition for Pattani Malay speech (transcribed as Malay)
  2. biodatlab/distill-whisper-th-large-v3, converted to CTranslate2
     -> ./models/whisper/distill-whisper-th-large-v3-ct2
     speech recognition for Thai speech
  3. facebook/nllb-200-distilled-600M -> ./models/nllb-200-distilled-600M
     translation (run_server.py sets HF_HOME=./models)

Run:  uv run python download_models.py
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent
# Match run_server.py: models and tokenizer caches live under ./models
os.environ.setdefault("HF_HOME", str(ROOT / "models"))

# Windows consoles default to a legacy codepage (cp874) that can't encode
# the status marks printed here — force UTF-8 so they print cleanly.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from huggingface_hub import snapshot_download  # noqa: E402


def main() -> None:
    whisper_dir = ROOT / "models" / "whisper"

    print("== 1/3: Whisper large-v3-turbo (CT2) ->", whisper_dir)
    from faster_whisper import download_model

    turbo_dir = Path(download_model("large-v3-turbo", cache_dir=str(whisper_dir)))
    if not (turbo_dir / "model.bin").exists():
        print(f"❌ No model.bin found in {turbo_dir}", file=sys.stderr)
        sys.exit(1)

    thai_dir = whisper_dir / "distill-whisper-th-large-v3-ct2"
    print("== 2/3: biodatlab/distill-whisper-th-large-v3 -> CT2 ->", thai_dir)
    if (thai_dir / "model.bin").exists():
        print("   already converted, skipping")
    else:
        # The model is published in transformers format only; faster-whisper
        # needs CTranslate2. This downloads about 3.2 GB and writes 1.6 GB.
        from ctranslate2.converters import TransformersConverter

        TransformersConverter(
            "biodatlab/distill-whisper-th-large-v3",
            copy_files=["tokenizer.json", "preprocessor_config.json"],
        ).convert(str(thai_dir), quantization="float16", force=True)
    if not (thai_dir / "model.bin").exists():
        print(f"❌ No model.bin found in {thai_dir}", file=sys.stderr)
        sys.exit(1)

    print("== 3/3: facebook/nllb-200-distilled-600M -> ./models/nllb-200-distilled-600M")
    nllb_dir = ROOT / "models" / "nllb-200-distilled-600M"
    # NOTE: downloaded as local_dir (plain files) because the default HF cache
    # layout needs OS symlinks, which are unavailable on Windows without
    # Developer Mode / admin privileges.
    snapshot_download(
        repo_id="facebook/nllb-200-distilled-600M",
        local_dir=str(nllb_dir),
        allow_patterns=["*.json", "model.safetensors", "sentencepiece.bpe.model"],
    )
    if not (nllb_dir / "model.safetensors").exists():
        # repo ships only .bin weights -> fetch those instead
        snapshot_download(
            repo_id="facebook/nllb-200-distilled-600M",
            local_dir=str(nllb_dir),
            allow_patterns=["pytorch_model.bin"],
        )
    if not any(
        (nllb_dir / w).exists()
        for w in ("model.safetensors", "pytorch_model.bin")
    ):
        print(f"❌ No model weights found in {nllb_dir}", file=sys.stderr)
        sys.exit(1)

    print("✅ Model download complete.")


if __name__ == "__main__":
    main()
