"""Position-scope mapping and row-edit application for streaming interventions.

Pure-logic tests (no model, no lens files): _chunk_local_indices maps global
template positions onto extend() chunks; _apply_edits must touch exactly the
selected rows, preserve dtype, and compose same-layer edits in list order.
"""
import mlx.core as mx
import numpy as np

from jlens_qwen.model import LayerEdit, _apply_edits, _chunk_local_indices


# ----- _chunk_local_indices -----

def test_explicit_positions_split_across_chunks():
    positions = [2, 5, 9]
    # chunk 1 covers [0, 4): hits 2
    assert _chunk_local_indices(positions, None, 0, 4) == [2]
    # chunk 2 covers [4, 8): hits 5 -> local 1
    assert _chunk_local_indices(positions, None, 4, 4) == [1]
    # chunk 3 covers [8, 12): hits 9 -> local 1
    assert _chunk_local_indices(positions, None, 8, 4) == [1]


def test_from_pos_before_inside_after_chunk():
    # from_pos before the chunk: every row selected
    assert _chunk_local_indices(None, 3, 10, 4) == [0, 1, 2, 3]
    # from_pos inside the chunk: suffix selected
    assert _chunk_local_indices(None, 12, 10, 4) == [2, 3]
    # from_pos after the chunk: nothing
    assert _chunk_local_indices(None, 99, 10, 4) == []


def test_union_of_scopes():
    # explicit position 1 plus from_pos 3, chunk [0, 5)
    assert _chunk_local_indices([1], 3, 0, 5) == [1, 3, 4]


def test_single_token_chunks_generation_regime():
    # generation extends one token at a time: start=N, n=1
    assert _chunk_local_indices(None, 7, 7, 1) == [0]
    assert _chunk_local_indices(None, 7, 6, 1) == []
    assert _chunk_local_indices([7], None, 7, 1) == [0]
    assert _chunk_local_indices([7], None, 8, 1) == []


def test_out_of_range_and_empty():
    assert _chunk_local_indices([100], None, 0, 8) == []
    assert _chunk_local_indices(None, None, 0, 8) == []
    assert _chunk_local_indices([], None, 0, 8) == []


def test_duplicates_deduped_and_sorted():
    assert _chunk_local_indices([3, 1, 3], 2, 0, 5) == [1, 2, 3, 4]


# ----- _apply_edits -----

def _rand_hidden(n=8, d=16):
    mx.random.seed(7)
    return mx.random.normal((1, n, d)).astype(mx.bfloat16)


def test_only_selected_rows_change():
    hidden = _rand_hidden()
    before = np.array(hidden.astype(mx.float32))
    edit = LayerEdit(layer=0, fn=lambda h: h + 1.0, positions=(2, 5), from_pos=None)
    out = _apply_edits(hidden, [edit], start=0)
    mx.eval(out)
    after = np.array(out.astype(mx.float32))
    for r in range(8):
        if r in (2, 5):
            assert np.allclose(after[0, r], before[0, r] + 1.0, atol=1e-2)
        else:
            assert np.array_equal(after[0, r], before[0, r])
    assert out.dtype == mx.bfloat16


def test_composition_order_within_layer():
    hidden = mx.zeros((1, 4, 8)).astype(mx.bfloat16)
    e1 = LayerEdit(layer=0, fn=lambda h: h + 3.0, positions=(1,), from_pos=None)
    e2 = LayerEdit(layer=0, fn=lambda h: h * 2.0, positions=(1,), from_pos=None)
    out = _apply_edits(hidden, [e1, e2], start=0)
    mx.eval(out)
    # (0 + 3) * 2 = 6 — list order, not commutative order.
    assert np.allclose(np.array(out.astype(mx.float32))[0, 1], 6.0)


def test_start_offset_maps_global_scope():
    hidden = _rand_hidden(n=3)
    before = np.array(hidden.astype(mx.float32))
    # global position 11 lands at local row 1 when the chunk starts at 10
    edit = LayerEdit(layer=0, fn=lambda h: h * 0.0, positions=(11,), from_pos=None)
    out = _apply_edits(hidden, [edit], start=10)
    mx.eval(out)
    after = np.array(out.astype(mx.float32))
    assert np.allclose(after[0, 1], 0.0)
    assert np.array_equal(after[0, 0], before[0, 0])
    assert np.array_equal(after[0, 2], before[0, 2])


def test_miss_is_identity():
    hidden = _rand_hidden()
    before = np.array(hidden.astype(mx.float32))
    edit = LayerEdit(layer=0, fn=lambda h: h + 99.0, positions=(50,), from_pos=None)
    out = _apply_edits(hidden, [edit], start=0)
    mx.eval(out)
    assert np.array_equal(np.array(out.astype(mx.float32)), before)


def test_batched_equals_per_row():
    hidden = _rand_hidden()
    fn = lambda h: h * 2.0 + 0.5
    batched = _apply_edits(
        mx.array(hidden), [LayerEdit(layer=0, fn=fn, positions=None, from_pos=0)], start=0
    )
    per_row = mx.array(hidden)
    for r in range(8):
        per_row = _apply_edits(
            per_row, [LayerEdit(layer=0, fn=fn, positions=(r,), from_pos=None)], start=0
        )
    mx.eval(batched, per_row)
    assert np.allclose(
        np.array(batched.astype(mx.float32)),
        np.array(per_row.astype(mx.float32)),
    )
