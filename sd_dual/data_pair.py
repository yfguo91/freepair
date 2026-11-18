"""Paired VIS/IR datasets for the shared-latent pipeline (same-name or regex-id pairing)."""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def _load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _load_gray(path: Path) -> Image.Image:
    return Image.open(path).convert("L")


class M3FDPairDatasetSD(Dataset):
    """
    Same pairing as `data.dataset.M3FDPairDataset`, but IR is returned as **3×H×W** in [-1,1]
    (repeat grayscale) for SD VAE.
    """

    def __init__(
        self,
        root: Union[str, Path],
        img_size: int = 512,
        max_samples: Optional[int] = None,
        default_caption: str = "a paired rgb and thermal image",
        captions_json: Optional[Union[str, Path]] = None,
        ir_channels: int = 3,
        vis_dir_name: str = "vi",
        ir_dir_name: str = "ir",
        pair_mode: str = "same_name",
        id_regex_vis: str = r"FLIR_(\d+)",
        id_regex_ir: str = r"FLIR_(\d+)",
    ) -> None:
        super().__init__()
        self.root = Path(root)
        vi_dir = self.root / str(vis_dir_name)
        ir_dir = self.root / str(ir_dir_name)
        if not vi_dir.is_dir() or not ir_dir.is_dir():
            raise FileNotFoundError(f"Need {vi_dir} and {ir_dir}")
        self.vi_dir = vi_dir
        self.ir_dir = ir_dir
        self.pair_mode = str(pair_mode).lower().strip()
        self.id_regex_vis = str(id_regex_vis)
        self.id_regex_ir = str(id_regex_ir)

        allowed_suffix = {".png", ".jpg", ".jpeg", ".bmp"}
        vis_files = sorted([p for p in vi_dir.iterdir() if p.is_file() and p.suffix.lower() in allowed_suffix])
        ir_files = sorted([p for p in ir_dir.iterdir() if p.is_file() and p.suffix.lower() in allowed_suffix])

        if self.pair_mode == "same_name":
            vis_map = {p.name: p for p in vis_files}
            ir_map = {p.name: p for p in ir_files}
            names = sorted([n for n in vis_map.keys() if n in ir_map])
            if max_samples is not None:
                names = names[: int(max_samples)]
            self.pairs: List[Tuple[str, str, str]] = [(n, n, n) for n in names]
        elif self.pair_mode == "id_regex":
            rv = re.compile(self.id_regex_vis)
            ri = re.compile(self.id_regex_ir)
            vis_id: Dict[str, Path] = {}
            ir_id: Dict[str, Path] = {}
            for p in vis_files:
                m = rv.search(p.name)
                if m:
                    vis_id[str(m.group(1))] = p
            for p in ir_files:
                m = ri.search(p.name)
                if m:
                    ir_id[str(m.group(1))] = p
            common_ids = sorted([k for k in vis_id.keys() if k in ir_id])
            if max_samples is not None:
                common_ids = common_ids[: int(max_samples)]
            self.pairs = [
                (vis_id[k].name, ir_id[k].name, f"FLIR_{k}") for k in common_ids
            ]
        else:
            raise ValueError("pair_mode must be 'same_name' or 'id_regex'")

        if not self.pairs:
            raise RuntimeError(
                f"No paired samples found under {vi_dir} and {ir_dir} with pair_mode={self.pair_mode}"
            )
        self.names: List[str] = [p[2] for p in self.pairs]
        self.default_caption = default_caption
        if int(ir_channels) not in (1, 3):
            raise ValueError("ir_channels must be 1 or 3")
        self.ir_channels = int(ir_channels)
        self.captions: Optional[Dict[str, str]] = None
        if captions_json is not None:
            path = Path(captions_json)
            if not path.is_file():
                raise FileNotFoundError(f"captions_json not found: {path}")
            with open(path, "r", encoding="utf-8") as f:
                raw: Any = json.load(f)
            if isinstance(raw, dict) and "captions" in raw and isinstance(
                raw["captions"], dict
            ):
                self.captions = {str(k): str(v) for k, v in raw["captions"].items()}
            elif isinstance(raw, dict):
                self.captions = {}
                for k, v in raw.items():
                    sk = str(k)
                    if sk.startswith("_") or sk in {"meta", "captions"}:
                        continue
                    self.captions[sk] = str(v)
            else:
                raise ValueError("captions_json must be a JSON object (filename -> caption).")
            missing = [n for n in self.names if (n not in self.captions)]
            if missing:
                warnings.warn(
                    f"{len(missing)} samples have no caption in JSON; using default_caption. "
                    f"Example missing: {missing[:3]}"
                )

        self.tf_vis = transforms.Compose(
            [
                transforms.Resize(
                    (img_size, img_size),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.tf_ir = transforms.Compose(
            [
                transforms.Resize(
                    (img_size, img_size),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
        vis_name, ir_name, sample_name = self.pairs[idx]
        vi = _load_rgb(self.vi_dir / vis_name)
        ir = _load_gray(self.ir_dir / ir_name)
        x_vis = self.tf_vis(vi)
        x_ir_1 = self.tf_ir(ir)
        if self.ir_channels == 1:
            x_ir = x_ir_1
        else:
            x_ir = x_ir_1.repeat(3, 1, 1)
        if self.captions is not None:
            cap = self.captions.get(sample_name, self.captions.get(vis_name, self.default_caption))
        else:
            cap = self.default_caption
        return x_vis, x_ir, sample_name, cap
