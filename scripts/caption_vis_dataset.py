#!/usr/bin/env python3
"""
Caption the visible-light images (vi/) of a dataset in English with BLIP/BLIP2;
each same-named ir/ pair shares the same caption at training time.

Expected layout (consistent with sd_dual/data_pair.py and the configs):
  <data_root>/
    vi/   <- the VL model is only run on these RGB images
    ir/   <- same-name files pair with vi/; captions are not duplicated here

Default output: <data_root>/vi_captions.json with a "captions" dict
  { "filename.png": "english caption", ... } for configs' data.captions_json.

Dependencies: transformers, torch, Pillow (see requirements.txt).
The first run downloads BLIP weights from Hugging Face; cache them beforehand
or set a mirror when offline.

Usage:
  python scripts/caption_vis_dataset.py --data-root /path/to/M3FD_Detection
  python scripts/caption_vis_dataset.py --data-root /path/to/M3FD_Detection --batch-size 8 --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tqdm import tqdm

try:
    from transformers import (
        AutoProcessor,
        Blip2ForConditionalGeneration,
        BlipForConditionalGeneration,
        BlipProcessor,
    )
except ImportError as e:  # pragma: no cover
    raise ImportError("transformers is required: pip install transformers") from e

from PIL import Image


def list_vi_files(vi_dir: Path) -> List[str]:
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    return sorted(
        p.name for p in vi_dir.iterdir() if p.is_file() and p.suffix.lower() in exts
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Dataset root (must contain vi/ and ir/, paired by name)",
    )
    ap.add_argument(
        "--vis-dir-name",
        type=str,
        default="vi",
        help="Visible-light dir name (default vi; FLIR=RGB, LLVIP=visible)",
    )
    ap.add_argument(
        "--ir-dir-name",
        type=str,
        default="ir",
        help="Infrared dir name (default ir; FLIR=IR, LLVIP=infrared)",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="",
        help="Output JSON path; default <data-root>/vi_captions.json",
    )
    ap.add_argument(
        "--model",
        type=str,
        default="Salesforce/blip2-opt-2.7b",
        help="Captioning model id or local directory",
    )
    ap.add_argument(
        "--model-type",
        type=str,
        default="auto",
        choices=["auto", "blip", "blip2"],
        help="Model type; auto tries BLIP2 first and falls back to BLIP",
    )
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=35)
    ap.add_argument(
        "--local-files-only",
        action="store_true",
        help="Offline: only use local cache / local model directories",
    )
    ap.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow transformers to run custom code from the model repo if needed",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="If the output JSON exists, only add missing filenames; keep existing captions",
    )
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    vi_dir = data_root / args.vis_dir_name
    ir_dir = data_root / args.ir_dir_name
    if not vi_dir.is_dir():
        raise FileNotFoundError(f"visible dir missing: {vi_dir}")
    if not ir_dir.is_dir():
        raise FileNotFoundError(f"infrared dir missing: {ir_dir}")

    # Only vis images are captioned; ir name pairing is not enforced here
    # (id_regex datasets like FLIR have different vis/ir filenames).
    names = list_vi_files(vi_dir)
    if not names:
        raise RuntimeError(f"no visible images found: {vi_dir}")

    out_path = Path(args.out) if args.out else data_root / "vi_captions.json"
    existing_caps: Dict[str, str] = {}
    if args.resume and out_path.is_file():
        with open(out_path, "r", encoding="utf-8") as f:
            blob = json.load(f)
        if isinstance(blob, dict) and "captions" in blob:
            existing_caps = {str(k): str(v) for k, v in blob["captions"].items()}
        elif isinstance(blob, dict):
            existing_caps = {
                str(k): str(v)
                for k, v in blob.items()
                if not str(k).startswith("_") and str(k) not in {"meta", "captions"}
            }

    import torch

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("CUDA unavailable, using CPU (slow).")

    proc_kw: Dict[str, object] = {}
    if args.local_files_only:
        proc_kw["local_files_only"] = True
    if args.trust_remote_code:
        proc_kw["trust_remote_code"] = True

    # Load processor/model from a local path or HF id.
    if args.model_type == "blip":
        processor = BlipProcessor.from_pretrained(args.model, **proc_kw)
        model = BlipForConditionalGeneration.from_pretrained(args.model, **proc_kw)
    elif args.model_type == "blip2":
        processor = AutoProcessor.from_pretrained(args.model, **proc_kw)
        model = Blip2ForConditionalGeneration.from_pretrained(args.model, **proc_kw)
    else:
        # auto: try BLIP2 first, fallback to BLIP.
        try:
            processor = AutoProcessor.from_pretrained(args.model, **proc_kw)
            model = Blip2ForConditionalGeneration.from_pretrained(args.model, **proc_kw)
            print("[caption] model_type=auto -> loaded as BLIP2")
        except Exception as e:
            print(f"[caption] BLIP2 load failed, fallback to BLIP: {e}")
            processor = BlipProcessor.from_pretrained(args.model, **proc_kw)
            model = BlipForConditionalGeneration.from_pretrained(args.model, **proc_kw)

    model = model.to(device)
    model.eval()

    captions: Dict[str, str] = dict(existing_caps)
    to_do = [n for n in names if n not in captions]
    if not to_do:
        print("All files already captioned; nothing to do.")
        return

    print(f"{len(names)} pairs total; {len(to_do)} visible images to caption")

    bs = max(1, int(args.batch_size))
    with torch.no_grad():
        for start in tqdm(range(0, len(to_do), bs), desc="BLIP caption"):
            batch_names = to_do[start : start + bs]
            images = [
                Image.open(vi_dir / n).convert("RGB") for n in batch_names
            ]
            inputs = processor(
                images=images, return_tensors="pt", padding=True
            ).to(device)
            gen = model.generate(
                **inputs, max_length=int(args.max_length)
            )
            texts = processor.batch_decode(gen, skip_special_tokens=True)
            for n, t in zip(batch_names, texts):
                captions[n] = t.strip()

    out_obj = {
        "_meta": {
            "model": args.model,
            "data_root": str(data_root),
            "vi_dir": args.vis_dir_name,
            "ir_dir": args.ir_dir_name,
            "note": "captions generated from visible vi/ images only; paired ir/ reuses the same caption at training time",
        },
        "captions": captions,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)
    print(f"Wrote {out_path} ({len(captions)} captions)")
    print("Set in configs:  data.captions_json: \"" + str(out_path) + "\"")


if __name__ == "__main__":
    main()
