import importlib.util
from types import SimpleNamespace

import pytest

from invarirank.modeling import SpanInfo


pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")


def test_block_mask_prevents_cross_candidate_attention():
    import torch

    from invarirank.modeling import build_attention_mask

    cfg = SimpleNamespace(attention_mask="block", span_causal=True)
    span = SpanInfo(span_start=0, span_end=3, candidate_spans=[(3, 5), (5, 7)])
    attn_2d = torch.ones((1, 7), dtype=torch.long)
    mask = build_attention_mask(attn_2d, span, cfg, torch.float32)
    allowed = mask[0, 0] == 0

    assert allowed[3, 0]
    assert allowed[3, 3]
    assert not allowed[3, 5]
    assert not allowed[5, 3]
