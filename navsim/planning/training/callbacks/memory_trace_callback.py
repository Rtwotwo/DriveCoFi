import csv
import os
import time
from pathlib import Path
from typing import Dict, Optional

import pytorch_lightning as pl
import torch

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None


def _read_proc_mem_kb() -> Dict[str, int]:
    """Read current process RSS/VMS in KB from /proc as a psutil-free fallback."""
    mem = {"rss_kb": -1, "vms_kb": -1}
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    mem["rss_kb"] = int(line.split()[1])
                elif line.startswith("VmSize:"):
                    mem["vms_kb"] = int(line.split()[1])
    except Exception:
        pass
    return mem


class MemoryTraceCallback(pl.Callback):
    """Trace host/GPU memory over time for leak localization."""

    def __init__(self, interval_steps: int = 10, output_subdir: str = "memory_trace") -> None:
        super().__init__()
        self.interval_steps = max(1, int(interval_steps))
        self.output_subdir = output_subdir
        self._csv_path: Optional[Path] = None
        self._writer: Optional[csv.DictWriter] = None
        self._fp = None
        self._pid = os.getpid()
        self._rank = int(os.getenv("RANK", "0"))
        self._local_rank = int(os.getenv("LOCAL_RANK", "0"))

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: Optional[str] = None) -> None:
        if self._writer is not None:
            return

        root_dir = Path(trainer.default_root_dir)
        out_dir = root_dir / self.output_subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = out_dir / f"rank{self._rank}_pid{self._pid}.csv"

        self._fp = open(self._csv_path, "w", newline="", encoding="utf-8")
        fieldnames = [
            "timestamp",
            "epoch",
            "global_step",
            "batch_idx",
            "stage",
            "pid",
            "rank",
            "local_rank",
            "rss_mb",
            "vms_mb",
            "children_rss_mb",
            "cuda_device",
            "cuda_allocated_mb",
            "cuda_reserved_mb",
            "cuda_max_allocated_mb",
        ]
        self._writer = csv.DictWriter(self._fp, fieldnames=fieldnames)
        self._writer.writeheader()
        self._fp.flush()

    def teardown(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: Optional[str] = None) -> None:
        if self._fp is not None:
            self._fp.close()
            self._fp = None

    def _current_mem(self) -> Dict[str, float]:
        rss_mb = -1.0
        vms_mb = -1.0
        children_rss_mb = 0.0

        if psutil is not None:
            proc = psutil.Process(self._pid)
            mem = proc.memory_info()
            rss_mb = float(mem.rss) / (1024.0 * 1024.0)
            vms_mb = float(mem.vms) / (1024.0 * 1024.0)
            for child in proc.children(recursive=True):
                try:
                    children_rss_mb += float(child.memory_info().rss) / (1024.0 * 1024.0)
                except Exception:
                    continue
        else:
            proc_mem = _read_proc_mem_kb()
            if proc_mem["rss_kb"] >= 0:
                rss_mb = float(proc_mem["rss_kb"]) / 1024.0
            if proc_mem["vms_kb"] >= 0:
                vms_mb = float(proc_mem["vms_kb"]) / 1024.0

        cuda_device = -1
        cuda_allocated_mb = 0.0
        cuda_reserved_mb = 0.0
        cuda_max_allocated_mb = 0.0
        if torch.cuda.is_available():
            cuda_device = torch.cuda.current_device()
            cuda_allocated_mb = float(torch.cuda.memory_allocated(cuda_device)) / (1024.0 * 1024.0)
            cuda_reserved_mb = float(torch.cuda.memory_reserved(cuda_device)) / (1024.0 * 1024.0)
            cuda_max_allocated_mb = float(torch.cuda.max_memory_allocated(cuda_device)) / (1024.0 * 1024.0)

        return {
            "rss_mb": rss_mb,
            "vms_mb": vms_mb,
            "children_rss_mb": children_rss_mb,
            "cuda_device": cuda_device,
            "cuda_allocated_mb": cuda_allocated_mb,
            "cuda_reserved_mb": cuda_reserved_mb,
            "cuda_max_allocated_mb": cuda_max_allocated_mb,
        }

    def _log_point(self, trainer: pl.Trainer, batch_idx: int, stage: str, force: bool = False) -> None:
        if self._writer is None:
            return
        if (not force) and (trainer.global_step % self.interval_steps != 0):
            return

        mem = self._current_mem()
        row = {
            "timestamp": f"{time.time():.6f}",
            "epoch": int(trainer.current_epoch),
            "global_step": int(trainer.global_step),
            "batch_idx": int(batch_idx),
            "stage": stage,
            "pid": self._pid,
            "rank": self._rank,
            "local_rank": self._local_rank,
            **mem,
        }
        self._writer.writerow(row)
        self._fp.flush()

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._log_point(trainer, batch_idx=-1, stage="fit_start", force=True)

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch,
        batch_idx: int,
    ) -> None:
        self._log_point(trainer, batch_idx=batch_idx, stage="batch_start")

    def on_before_backward(self, trainer: pl.Trainer, pl_module: pl.LightningModule, loss: torch.Tensor) -> None:
        self._log_point(trainer, batch_idx=-1, stage="before_backward")

    def on_after_backward(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._log_point(trainer, batch_idx=-1, stage="after_backward")

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        self._log_point(trainer, batch_idx=batch_idx, stage="batch_end")

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._log_point(trainer, batch_idx=-1, stage="fit_end", force=True)
