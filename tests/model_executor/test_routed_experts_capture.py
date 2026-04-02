# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import types

import numpy as np
import pytest
import torch
import torch.nn as nn

from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter

pytestmark = pytest.mark.cpu_test


class DummyRouter(BaseRouter):
    @property
    def routing_method_type(self) -> RoutingMethodType:
        return RoutingMethodType.FUSED_TOPK

    def _compute_routing(self, hidden_states, router_logits, indices_type):
        topk_ids = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
        topk_weights = torch.ones_like(topk_ids, dtype=torch.float32)
        return topk_weights, topk_ids

    def _apply_eplb_mapping(self, topk_ids: torch.Tensor) -> torch.Tensor:
        # Make mapping observable without requiring CUDA EPLB path.
        return topk_ids + 10


def _make_router() -> DummyRouter:
    return DummyRouter(
        top_k=2,
        global_num_experts=16,
        eplb_state=EplbLayerState(),
        enable_eplb=False,
        indices_type_getter=None,
    )


def test_base_router_capture_pre_eplb_mapping():
    router = _make_router()
    captured = []

    def capture_fn(ids):
        captured.append(ids.clone())

    router.set_capture_fn(capture_fn)
    topk_weights, topk_ids = router.select_experts(
        hidden_states=torch.empty(1),
        router_logits=torch.empty(1),
    )

    assert topk_weights.shape == topk_ids.shape
    assert len(captured) == 1
    assert torch.equal(captured[0], torch.tensor([[1, 2], [3, 4]]))
    assert torch.equal(topk_ids, torch.tensor([[11, 12], [13, 14]]))


def test_base_router_capture_with_eplb_enabled():
    router = _make_router()
    router.enable_eplb = True
    router.eplb_state.expert_load_view = torch.zeros(32, dtype=torch.int64)
    router.eplb_state.logical_to_physical_map = torch.arange(32).view(32, 1)
    router.eplb_state.logical_replica_count = torch.ones(32, dtype=torch.int64)

    captured = []

    def capture_fn(ids):
        captured.append(ids.clone())

    router.set_capture_fn(capture_fn)
    _, topk_ids = router.select_experts(
        hidden_states=torch.empty(1),
        router_logits=torch.empty(1),
    )

    assert len(captured) == 1
    # Capture should see logical ids pre-EPLB mapping.
    assert torch.equal(captured[0], torch.tensor([[1, 2], [3, 4]]))
    # Our DummyRouter mapping adds +10.
    assert torch.equal(topk_ids, torch.tensor([[11, 12], [13, 14]]))


def test_gpu_model_runner_binds_router_capture(monkeypatch):
    from vllm.v1.worker import gpu_model_runner as gmr

    class DummyFusedMoE:
        def __init__(self):
            self.layer_id = 7
            self.router = _make_router()

    class DummyCapturer:
        def __init__(self):
            self.calls = []

        def capture(self, layer_id, topk_ids):
            self.calls.append((layer_id, topk_ids))

    dummy_module = DummyFusedMoE()

    # Patch the runtime import inside _bind_routed_experts_capturer.
    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer

    monkeypatch.setattr(fused_moe_layer, "FusedMoE", DummyFusedMoE)

    dummy_self = types.SimpleNamespace(
        compilation_config=types.SimpleNamespace(
            static_forward_context={"dummy": dummy_module}
        )
    )

    capturer = DummyCapturer()
    gmr.GPUModelRunner._bind_routed_experts_capturer(dummy_self, capturer)

    assert dummy_module.router.capture_fn is not None
    dummy_module.router.capture_fn(torch.tensor([[5, 6]]))

    assert len(capturer.calls) == 1
    layer_id, topk_ids = capturer.calls[0]
    assert layer_id == 7
    assert torch.equal(topk_ids, torch.tensor([[5, 6]]))


def test_gpu_model_runner_binding_stage(monkeypatch):
    from vllm.v1.worker import gpu_model_runner as gmr

    class DummyFusedMoE:
        def __init__(self):
            self.layer_id = 11
            self.router = _make_router()

    class DummyCapturer:
        def __init__(self):
            self.calls = []

        def capture(self, layer_id, topk_ids):
            self.calls.append((layer_id, topk_ids))

    dummy_module = DummyFusedMoE()

    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer

    monkeypatch.setattr(fused_moe_layer, "FusedMoE", DummyFusedMoE)

    dummy_self = types.SimpleNamespace(
        compilation_config=types.SimpleNamespace(
            static_forward_context={"dummy": dummy_module}
        )
    )

    # Before binding, no capture hook.
    assert dummy_module.router.capture_fn is None

    capturer = DummyCapturer()
    gmr.GPUModelRunner._bind_routed_experts_capturer(dummy_self, capturer)

    # After binding, hook should exist and be callable.
    assert callable(dummy_module.router.capture_fn)
    dummy_module.router.capture_fn(torch.tensor([[9, 10]]))
    assert len(capturer.calls) == 1


class DummySae:
    def to(self, device, dtype):
        return self


def _make_dummy_granitemoe(monkeypatch):
    import vllm.model_executor.models.granitemoe as granitemoe_module

    class DummyGate(nn.Module):
        def __init__(self, hidden_size, num_experts, **kwargs):
            super().__init__()
            self.num_experts = num_experts

        def forward(self, hidden_states):
            logits = torch.arange(
                hidden_states.shape[0] * self.num_experts, dtype=torch.float32
            ).view(hidden_states.shape[0], self.num_experts)
            return logits, None

    class DummyExperts(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.last_router_logits = None

        def forward(self, hidden_states, router_logits):
            self.last_router_logits = router_logits
            return hidden_states

    monkeypatch.setattr(granitemoe_module, "ReplicatedLinear", DummyGate)
    monkeypatch.setattr(granitemoe_module, "FusedMoE", DummyExperts)
    return granitemoe_module.GraniteMoeMoE(
        num_experts=72,
        top_k=10,
        hidden_size=4,
        intermediate_size=8,
    )


def test_granitemoe_moe_capture_routing_disabled(monkeypatch):
    moe = _make_dummy_granitemoe(monkeypatch)

    _ = moe(torch.randn(2, 4))

    assert moe.capture_routing is False
    assert moe._last_router_logits is None


def test_granitemoe_moe_capture_routing_enabled(monkeypatch):
    moe = _make_dummy_granitemoe(monkeypatch)
    moe.capture_routing = True

    _ = moe(torch.randn(2, 4))

    expected = torch.arange(2 * 72, dtype=torch.float32).view(2, 72)
    assert moe._last_router_logits is not None
    assert torch.equal(moe._last_router_logits, expected)
    assert not moe._last_router_logits.requires_grad


def test_gpu_model_runner_collect_routing_logits(monkeypatch):
    from vllm.v1.worker import gpu_model_runner as gmr

    monkeypatch.setattr(gmr, "_ROUTING_CAPTURE", True)
    layer0 = types.SimpleNamespace(
        _last_router_logits=torch.arange(2 * 72, dtype=torch.float32).view(2, 72)
    )
    layer3 = types.SimpleNamespace(
        _last_router_logits=(
            torch.arange(2 * 72, dtype=torch.float32).view(2, 72) + 1000
        )
    )
    dummy_self = types.SimpleNamespace(
        _routing_capture_layers=[(0, layer0), (3, layer3)],
        _routing_capture_num_layers=4,
    )

    routing = gmr.GPUModelRunner._collect_routing_logits(dummy_self)

    assert routing is not None
    assert routing.token_index.shape == (40,)
    assert routing.layer_index.shape == (40,)
    assert routing.expert_index.shape == (40,)
    assert routing.gate_value.shape == (40,)
    assert routing.num_tokens == 2
    assert routing.num_layers == 4
    assert routing.num_experts == 72
    assert routing.top_k == 10
    assert layer0._last_router_logits is None
    assert layer3._last_router_logits is None
    assert set(np.unique(routing.layer_index).tolist()) == {0, 3}

    mask = (routing.layer_index == 0) & (routing.token_index == 0)
    np.testing.assert_array_equal(
        routing.expert_index[mask], np.arange(71, 61, -1, dtype=np.int8)
    )
    np.testing.assert_allclose(
        routing.gate_value[mask].astype(np.float32).sum(),
        1.0,
        rtol=1e-3,
        atol=1e-3,
    )


def test_gpu_model_runner_ensure_sae_loaded_enables_routing_capture(monkeypatch):
    from vllm.sae import sparse_encoder
    from vllm.v1.worker import gpu_model_runner as gmr

    monkeypatch.setattr(gmr, "_ROUTING_CAPTURE", True)
    monkeypatch.setattr(
        sparse_encoder, "load_sae_from_env", lambda: (DummySae(), 1)
    )

    layers = [
        types.SimpleNamespace(
            block_sparse_moe=types.SimpleNamespace(
                capture_routing=False, _last_router_logits="stale"
            )
        )
        for _ in range(2)
    ]
    dummy_self = types.SimpleNamespace(
        _sae_enabled=True,
        _sae=None,
        _sae_block_tokens=512,
        _routing_capture_layers=[],
        _routing_capture_num_layers=0,
        device=torch.device("cpu"),
        model=types.SimpleNamespace(model=types.SimpleNamespace(layers=layers)),
        model_config=types.SimpleNamespace(sae_max_layer=1, dtype=torch.float32),
    )

    gmr.GPUModelRunner._ensure_sae_loaded(dummy_self)

    assert isinstance(dummy_self._sae, DummySae)
    assert dummy_self._routing_capture_num_layers == 2
    assert len(dummy_self._routing_capture_layers) == 2
    assert all(layer.block_sparse_moe.capture_routing for layer in layers)
    assert all(layer.block_sparse_moe._last_router_logits is None for layer in layers)


def test_gpu_model_runner_collect_routing_logits_returns_none_when_disabled(
    monkeypatch,
):
    from vllm.v1.worker import gpu_model_runner as gmr

    monkeypatch.setattr(gmr, "_ROUTING_CAPTURE", False)
    layer = types.SimpleNamespace(_last_router_logits=torch.zeros(1, 72))
    dummy_self = types.SimpleNamespace(
        _routing_capture_layers=[(0, layer)],
        _routing_capture_num_layers=1,
    )

    assert gmr.GPUModelRunner._collect_routing_logits(dummy_self) is None


def test_gpu_model_runner_ensure_sae_loaded_skips_routing_when_disabled(monkeypatch):
    from vllm.sae import sparse_encoder
    from vllm.v1.worker import gpu_model_runner as gmr

    monkeypatch.setattr(gmr, "_ROUTING_CAPTURE", False)
    monkeypatch.setattr(
        sparse_encoder, "load_sae_from_env", lambda: (DummySae(), 1)
    )

    layers = [
        types.SimpleNamespace(
            block_sparse_moe=types.SimpleNamespace(
                capture_routing=False, _last_router_logits=None
            )
        )
        for _ in range(2)
    ]
    dummy_self = types.SimpleNamespace(
        _sae_enabled=True,
        _sae=None,
        _sae_block_tokens=512,
        _routing_capture_layers=[],
        _routing_capture_num_layers=0,
        device=torch.device("cpu"),
        model=types.SimpleNamespace(model=types.SimpleNamespace(layers=layers)),
        model_config=types.SimpleNamespace(sae_max_layer=1, dtype=torch.float32),
    )

    gmr.GPUModelRunner._ensure_sae_loaded(dummy_self)

    assert isinstance(dummy_self._sae, DummySae)
    assert dummy_self._routing_capture_layers == []
    assert dummy_self._routing_capture_num_layers == 2
    assert all(not layer.block_sparse_moe.capture_routing for layer in layers)
