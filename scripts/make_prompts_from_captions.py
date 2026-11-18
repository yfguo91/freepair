#!/usr/bin/env python3
"""
Export a prompt text file (one prompt per line) from a captions JSON, for
text-conditioned batch sampling.

Supports:
- JSON of the form {"captions": {...}} or a plain {filename: caption} map
- Deduplication
- Reproducible random sampling of N prompts
- Minimum-length filtering and prefix/suffix concatenation

Example:
  python scripts/make_prompts_from_captions.py \
    --captions-json /path/to/M3FD_Detection/vi_captions.json \
    --out prompts.txt \
    --num 256 --dedup --shuffle --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List


def load_caption_map(path: Path) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "captions" in obj and isinstance(obj["captions"], dict):
        return {str(k): str(v) for k, v in obj["captions"].items()}
    if isinstance(obj, dict):
        out: Dict[str, str] = {}
        for k, v in obj.items():
            sk = str(k)
            if sk.startswith("_") or sk in {"meta", "_meta", "captions"}:
                continue
            out[sk] = str(v)
        return out
    raise ValueError("captions_json must be an object: {'captions': {...}} or {filename: caption}.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions-json", type=str, required=True, help="Input captions JSON path")
    ap.add_argument("--out", type=str, default="prompts.txt", help="Output prompts text path")
    ap.add_argument("--num", type=int, default=0, help="Number of prompts to export; <=0 means all")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle before export")
    ap.add_argument("--dedup", action="store_true", help="Dedup by text (case/whitespace insensitive)")
    ap.add_argument("--min-chars", type=int, default=3, help="Filter out captions shorter than this")
    ap.add_argument("--prefix", type=str, default="", help="Prefix prepended to every prompt")
    ap.add_argument("--suffix", type=str, default="", help="Suffix appended to every prompt")
    args = ap.parse_args()

    random.seed(args.seed)
    cap_map = load_caption_map(Path(args.captions_json))
    prompts: List[str] = []
    for _name, cap in cap_map.items():
        txt = cap.strip()
        if len(txt) < int(args.min_chars):
            continue
        if args.prefix:
            txt = f"{args.prefix}{txt}"
        if args.suffix:
            txt = f"{txt}{args.suffix}"
        prompts.append(txt)

    if not prompts:
        raise RuntimeError("No valid captions found after filtering.")

    if args.dedup:
        seen = set()
        uniq: List[str] = []
        for p in prompts:
            k = p.strip().lower()
            if k in seen:
                continue
            seen.add(k)
            uniq.append(p)
        prompts = uniq

    if args.shuffle:
        random.shuffle(prompts)

    if int(args.num) > 0:
        prompts = prompts[: int(args.num)]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(p + "\n")

    print(f"Saved {len(prompts)} prompts to {out_path}")


if __name__ == "__main__":
    main()

