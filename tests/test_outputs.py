# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest

from vllm.outputs import RequestOutput
from vllm.v1.outputs import RoutingSparseOutput, SaeSparseOutput

pytestmark = pytest.mark.cpu_test


def test_request_output_forward_compatible():
    output = RequestOutput(
        request_id="test_request_id",
        prompt="test prompt",
        prompt_token_ids=[1, 2, 3],
        prompt_logprobs=None,
        outputs=[],
        finished=False,
        example_arg_added_in_new_version="some_value",
    )
    assert output is not None


def test_routing_sparse_output_shapes_and_sae_default():
    routing = RoutingSparseOutput(
        token_index=np.array([0, 0, 1], dtype=np.int32),
        layer_index=np.array([0, 1, 1], dtype=np.int16),
        expert_index=np.array([3, 7, 11], dtype=np.int8),
        gate_value=np.array([0.5, 0.3, 0.2], dtype=np.float16),
        num_tokens=2,
        num_layers=2,
    )

    assert routing.token_index.shape == (3,)
    assert routing.layer_index.shape == (3,)
    assert routing.expert_index.shape == (3,)
    assert routing.gate_value.shape == (3,)
    assert routing.token_index.dtype == np.int32
    assert routing.layer_index.dtype == np.int16
    assert routing.expert_index.dtype == np.int8
    assert routing.gate_value.dtype == np.float16
    assert routing.num_tokens == 2
    assert routing.num_layers == 2
    assert routing.num_experts == 72
    assert routing.top_k == 10

    sae = SaeSparseOutput(
        token_index=np.array([0], dtype=np.int32),
        feature_index=np.array([1], dtype=np.int16),
        value=np.array([1.0], dtype=np.float16),
        num_tokens=1,
    )
    assert sae.routing is None

    sae_with_routing = SaeSparseOutput(
        token_index=np.array([0], dtype=np.int32),
        feature_index=np.array([1], dtype=np.int16),
        value=np.array([1.0], dtype=np.float16),
        num_tokens=1,
        routing=routing,
    )
    assert sae_with_routing.routing is routing
