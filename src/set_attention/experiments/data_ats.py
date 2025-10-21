from __future__ import annotations
from dataclasses import dataclass
from typing import List
import torch
from torch.utils.data import Dataset, DataLoader


def _read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


@dataclass
class SummarizationConfig:
    src_path: str
    sum_path: str
    max_len: int = 128
    batch_size: int = 16


class SummarizationDataset(Dataset):
    def __init__(self, cfg: SummarizationConfig):
        self.src = _read_lines(cfg.src_path)
        self.sum = _read_lines(cfg.sum_path)
        assert len(self.src) == len(self.sum)
        self.max_len = cfg.max_len

    def __len__(self):
        return len(self.src)

    def __getitem__(self, i: int):
        return self.src[i], self.sum[i]


def make_summarization_loader(cfg: SummarizationConfig) -> DataLoader:
    ds = SummarizationDataset(cfg)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

