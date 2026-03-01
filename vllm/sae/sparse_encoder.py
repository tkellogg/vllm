# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU-side SAE encoder that returns sparse indices/values on CPU."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import torch


@dataclass
class SparseSAE:
    encoder_weight: torch.Tensor
    encoder_bias: torch.Tensor
    decoder_weight: torch.Tensor
    decoder_bias: torch.Tensor
    threshold: torch.Tensor
    k: int = 0

    def to(self, device: torch.device, dtype: torch.dtype) -> "SparseSAE":
        self.encoder_weight = self.encoder_weight.to(device=device, dtype=dtype)
        self.encoder_bias = self.encoder_bias.to(device=device, dtype=dtype)
        self.decoder_weight = self.decoder_weight.to(device=device, dtype=dtype)
        self.decoder_bias = self.decoder_bias.to(device=device, dtype=dtype)
        self.threshold = self.threshold.to(device=device, dtype=dtype)
        return self

    def encode(self, activations: torch.Tensor) -> torch.Tensor:
        centered = activations - self.decoder_bias
        pre_act = centered @ self.encoder_weight.t() + self.encoder_bias
        return torch.where(pre_act > self.threshold, pre_act, torch.zeros_like(pre_act))


def _resolve_sae_paths(
    repo: str,
    subdir: str,
    cache_dir: str | None,
    local_root: str | None,
) -> tuple[Path, Path, Path]:
    if local_root or (subdir and Path(subdir).is_absolute()):
        if subdir and Path(subdir).is_absolute():
            base = Path(subdir)
        else:
            base = Path(local_root) if local_root else None
        if base is None:
            raise ValueError("sae_local_root must be set when sae_subdir is not absolute.")
        if subdir and not Path(subdir).is_absolute():
            base = base / subdir
        config_path = base / "cfg.json"
        weights_path = base / "sae_weights.safetensors"
        sparsity_path = base / "sparsity.safetensors"
        return config_path, weights_path, sparsity_path

    if not repo:
        raise ValueError("sae_repo must be set when sae_local_root is not provided.")

    from huggingface_hub import hf_hub_download

    config_path = Path(
        hf_hub_download(
            repo,
            f"{subdir}/cfg.json",
            cache_dir=cache_dir,
        )
    )
    weights_path = Path(
        hf_hub_download(
            repo,
            f"{subdir}/sae_weights.safetensors",
            cache_dir=cache_dir,
        )
    )
    sparsity_path = Path(
        hf_hub_download(
            repo,
            f"{subdir}/sparsity.safetensors",
            cache_dir=cache_dir,
        )
    )
    return config_path, weights_path, sparsity_path


def _load_json(path: Path) -> dict[str, Any]:
    import json

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_sae_from_env() -> tuple[SparseSAE, int]:
    repo = (os.environ.get("VLLM_SAE_REPO") or "").strip()
    subdir = (os.environ.get("VLLM_SAE_SUBDIR") or "").strip()
    cache_dir = (os.environ.get("VLLM_SAE_CACHE_DIR") or "").strip() or None
    local_root = (os.environ.get("VLLM_SAE_LOCAL_ROOT") or "").strip() or None
    layer_override = os.environ.get("VLLM_SAE_LAYER_INDEX")

    if not subdir:
        raise ValueError("VLLM_SAE_SUBDIR must be set to enable SAE.")

    config_path, weights_path, sparsity_path = _resolve_sae_paths(
        repo, subdir, cache_dir, local_root
    )
    config = _load_json(config_path)

    runner_path = config_path.with_name("runner_cfg.json")
    runner_cfg = _load_json(runner_path) if runner_path.exists() else None

    try:
        from safetensors.torch import load_file as safetensors_load_file
    except Exception as exc:
        raise ImportError("safetensors is required to load SAE weights") from exc

    weights = safetensors_load_file(str(weights_path))
    sparsity = safetensors_load_file(str(sparsity_path))

    def _get_first(state: dict[str, torch.Tensor], keys: list[str]) -> torch.Tensor | None:
        for key in keys:
            if key in state:
                return state[key]
        return None

    W_enc = _get_first(weights, ["encoder.weight", "W_enc", "w_enc"])
    W_dec = _get_first(weights, ["decoder.weight", "W_dec", "w_dec"])
    b_dec = _get_first(weights, ["b_dec", "decoder.bias", "bias"])
    b_enc = _get_first(weights, ["encoder.bias", "b_enc"])

    if W_enc is None or W_dec is None or b_dec is None:
        raise ValueError(
            "Missing SAE weights. Required keys: W_enc/encoder.weight, W_dec/decoder.weight, b_dec."
        )

    if b_dec.ndim != 1:
        raise ValueError(f"Expected b_dec to be 1D, got shape {tuple(b_dec.shape)}")

    d_in = b_dec.shape[0]
    if W_enc.ndim != 2:
        raise ValueError(f"Expected W_enc to be 2D, got shape {tuple(W_enc.shape)}")
    if W_enc.shape[1] != d_in and W_enc.shape[0] == d_in:
        W_enc = W_enc.T
    if W_enc.shape[1] != d_in:
        raise ValueError(
            f"W_enc shape {tuple(W_enc.shape)} is incompatible with b_dec shape {tuple(b_dec.shape)}"
        )

    if W_dec.ndim != 2:
        raise ValueError(f"Expected W_dec to be 2D, got shape {tuple(W_dec.shape)}")
    if W_dec.shape[0] != d_in and W_dec.shape[1] == d_in:
        W_dec = W_dec.T

    d_sae = W_enc.shape[0]
    if b_enc is None:
        b_enc = torch.zeros((d_sae,), dtype=W_enc.dtype)

    threshold = _get_first(weights, ["threshold"])
    if threshold is None:
        threshold = _get_first(sparsity, ["threshold"])
    if threshold is None:
        threshold = _get_first(weights, ["log_threshold"])
        if threshold is not None:
            threshold = threshold.exp()
    if threshold is None:
        threshold = _get_first(sparsity, ["log_threshold"])
        if threshold is not None:
            threshold = threshold.exp()
    if threshold is None:
        threshold = torch.zeros((d_sae,), dtype=W_enc.dtype)
    if threshold.ndim == 0:
        threshold = torch.full((d_sae,), threshold.item(), dtype=threshold.dtype)

    k_value = _get_first(weights, ["k", "topk_threshold"])
    if k_value is None:
        k_value = _get_first(sparsity, ["k", "topk_threshold"])
    if k_value is None:
        k_value = 0

    if isinstance(k_value, torch.Tensor):
        k_value = int(k_value.flatten()[0].item()) if k_value.numel() else 0

    sae = SparseSAE(
        encoder_weight=W_enc,
        encoder_bias=b_enc,
        decoder_weight=W_dec,
        decoder_bias=b_dec,
        threshold=threshold,
        k=int(k_value),
    )

    layer = None
    if runner_cfg:
        layer = runner_cfg.get("trainer", {}).get("layer")
    if layer is None:
        layer = config.get("trainer", {}).get("layer")
    if layer_override is not None:
        layer = int(layer_override)
    if layer is None:
        raise ValueError("Unable to determine SAE layer index. Set VLLM_SAE_LAYER_INDEX.")

    return sae, int(layer)


def encode_sparse(
    sae: SparseSAE,
    hidden_states: torch.Tensor,
    *,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    encoded = sae.encode(hidden_states)
    if threshold > 0:
        mask = encoded > threshold
    else:
        mask = encoded > 0
    nz = torch.nonzero(mask, as_tuple=False)
    if nz.numel() == 0:
        empty = torch.empty((0,), device="cpu", dtype=torch.int32)
        return empty, empty, torch.empty((0,), device="cpu", dtype=torch.float16)
    token_idx = nz[:, 0]
    feat_idx = nz[:, 1]
    values = encoded[nz[:, 0], nz[:, 1]]
    return (
        token_idx.to(torch.int32).cpu(),
        feat_idx.to(torch.int32).cpu(),
        values.to(torch.float16).cpu(),
    )
