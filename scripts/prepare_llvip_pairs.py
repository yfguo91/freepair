#!/usr/bin/env python3
"""Flatten LLVIP visible/infrared (train+test) into paired flat directories.

Expects the official layout after unzip:
  <llvip_root>/visible/{train,test}/*.jpg
  <llvip_root>/infrared/{train,test}/*.jpg

Default output (symlinks):
  <out_root>/visible/*.jpg
  <out_root>/infrared/*.jpg

Same basename must exist in both modalities; train+test are merged (no split).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def _link_or_copy(src: Path, dst: Path, use_symlink: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if use_symlink:
        os.symlink(src.resolve(), dst, target_is_directory=False)
    else:
        shutil.copy2(src, dst)


def main() -> int:
    p = argparse.ArgumentParser(description="Prepare flat LLVIP pair directories.")
    p.add_argument(
        "--llvip-root",
        type=Path,
        default=Path("data/LLVIP"),
        help="Unzipped LLVIP root (contains visible/ and infrared/).",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=Path("data/LLVIP_pair"),
        help="Output root with visible/ and infrared/ flat folders.",
    )
    p.add_argument("--copy", action="store_true", help="Copy files instead of symlink.")
    args = p.parse_args()

    root = args.llvip_root.resolve()
    out = args.out_root.resolve()
    use_symlink = not args.copy

    subdirs = ("train", "test")
    vis_out = out / "visible"
    ir_out = out / "infrared"

    vis_src_dirs = [root / "visible" / s for s in subdirs]
    ir_src_dirs = [root / "infrared" / s for s in subdirs]

    for d in vis_src_dirs + ir_src_dirs:
        if not d.is_dir():
            print(f"ERROR: missing directory: {d}", file=sys.stderr)
            return 1

    n_vis = n_ir = 0
    dup_warn = 0

    for split, vdir, idir in zip(subdirs, vis_src_dirs, ir_src_dirs):
        for src in sorted(vdir.glob("*.jpg")) + sorted(vdir.glob("*.JPG")):
            dst = vis_out / src.name
            if dst.exists() or dst.is_symlink():
                dup_warn += 1
                continue
            _link_or_copy(src, dst, use_symlink)
            n_vis += 1
        for src in sorted(idir.glob("*.jpg")) + sorted(idir.glob("*.JPG")):
            dst = ir_out / src.name
            if dst.exists() or dst.is_symlink():
                dup_warn += 1
                continue
            _link_or_copy(src, dst, use_symlink)
            n_ir += 1

    vis_names = {p.name for p in vis_out.iterdir() if p.is_file() or p.is_symlink()}
    ir_names = {p.name for p in ir_out.iterdir() if p.is_file() or p.is_symlink()}
    paired = sorted(vis_names & ir_names)
    missing_vis = len(ir_names - vis_names)
    missing_ir = len(vis_names - ir_names)

    print(f"Linked/copied visible: {n_vis}, infrared: {n_ir} (new files in this run)")
    print(f"Flat visible count: {len(vis_names)}, infrared: {len(ir_names)}, paired: {len(paired)}")
    if dup_warn:
        print(f"Skipped duplicates (same filename in train+test or re-run): {dup_warn}")
    if missing_vis or missing_ir:
        print(
            f"WARNING: unpaired filenames — only in IR: {missing_vis}, only in VIS: {missing_ir}",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
