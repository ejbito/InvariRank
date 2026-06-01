import importlib.util

import pytest

from invarirank.modeling import SpanInfo


pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")


def test_shared_positions_restart_candidate_frame():
    import torch

    from invarirank.modeling import make_shared_position_ids

    input_ids = torch.ones((1, 8), dtype=torch.long)
    span = SpanInfo(span_start=0, span_end=3, candidate_spans=[(3, 5), (5, 8)])
    pos = make_shared_position_ids(input_ids, span)[0].tolist()
    assert pos[0:3] == [0, 1, 2]
    assert pos[3:5] == [3, 4]
    assert pos[5:8] == [3, 4, 5]
