from __future__ import annotations

from typing import Any, Dict

import torch


def resolve_device(raw_cfg: Dict[str, Any]) -> torch.device:
    """Resolve runtime device with GPU preference by default.

    Priority when device=auto:
    1) CUDA GPU
    2) MPS
    3) CPU
    """
    device_setting = str(raw_cfg.get("device", "auto")).lower()
    runtime_cfg = raw_cfg.get("runtime", {})
    gpu_id = int(runtime_cfg.get("gpu_id", 0))

    if device_setting == "auto":
        if torch.cuda.is_available():
            return torch.device(f"cuda:{gpu_id}")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if device_setting.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(device_setting)
        return torch.device("cpu")

    return torch.device(device_setting)


def device_summary(device: torch.device) -> str:
    if device.type == "cuda":
        idx = 0 if device.index is None else int(device.index)
        name = torch.cuda.get_device_name(idx)
        return f"{device} ({name})"
    return str(device)
