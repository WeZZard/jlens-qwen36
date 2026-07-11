"""Numeric equivalence of the server-path intervention ops.

Runs against small synthetic weights (no 27B model, no lens files):
- unembed_rows == row slice of the full dequantized W_U
- j_lens_vectors_lite == rows of W_U @ J (the j_lens_vectors definition)
- patch_swap_rows == patch_swap applied per row; orthogonal part preserved
- ablate_rows zeroes the span{V} lens coordinates
- compile_edits produces closures matching the underlying ops
"""
import mlx.core as mx
import numpy as np
import pytest

from jlens_qwen.interventions import (
    ablate_rows,
    compile_edits,
    gram_inv,
    j_lens_vectors_lite,
    make_swap_basis,
    patch_swap,
    patch_swap_rows,
    unembed_rows,
)
from jlens_qwen.lens import JacobianLens

D = 64
VOCAB = 32


class StubQuantizedHead:
    """Duck-types QuantizedLinear for unembed_rows: subscriptable weights."""

    def __init__(self, w: mx.array, group_size: int = 32, bits: int = 4):
        self.weight, self.scales, self.biases = mx.quantize(
            w, group_size=group_size, bits=bits
        )
        self.group_size = group_size
        self.bits = bits

    def __getitem__(self, key):
        return getattr(self, key)

    def dense(self) -> mx.array:
        return mx.dequantize(
            self.weight, self.scales, self.biases,
            group_size=self.group_size, bits=self.bits,
        ).astype(mx.float32)


class StubModel:
    def __init__(self, n_layers: int = 4):
        mx.random.seed(11)
        self._lm_head = StubQuantizedHead(mx.random.normal((VOCAB, D)))
        self.n_layers = n_layers


@pytest.fixture(scope="module")
def model():
    return StubModel()


@pytest.fixture(scope="module")
def lens():
    rng = np.random.default_rng(3)
    jac = {
        l: rng.standard_normal((D, D)).astype(np.float32) * 0.2
        for l in range(3)  # layers 0..2 fitted; layer 3 = final, unfitted
    }
    return JacobianLens(jac, n_prompts=1, d_model=D)


def test_unembed_rows_matches_full_dequantize(model):
    full = model._lm_head.dense()
    ids = [0, 5, 31]
    rows = unembed_rows(model, ids)
    mx.eval(rows)
    assert rows.shape == (3, D)
    assert np.allclose(np.array(rows), np.array(full[mx.array(ids)]), atol=1e-6)


def test_j_lens_vectors_lite_matches_definition(model, lens):
    full = np.array(model._lm_head.dense())
    for layer in (0, 2):
        J = lens.jacobians[layer]
        expected = full @ J  # V = W_U @ J, rows are v_t
        got = j_lens_vectors_lite(lens, model, layer, [1, 7])
        mx.eval(got)
        # fp16 matvec against fp32 reference: loose-but-tight-enough tol.
        assert np.allclose(np.array(got), expected[[1, 7]], rtol=2e-2, atol=2e-2)


def test_j_lens_vectors_lite_final_layer_identity(model, lens):
    # The final (unfitted) layer reads out through the logit lens: v_t = W_U[t].
    got = j_lens_vectors_lite(lens, model, model.n_layers - 1, [4])
    mx.eval(got)
    full = np.array(model._lm_head.dense())
    assert np.allclose(np.array(got)[0], full[4], atol=1e-6)


def test_j_lens_vectors_lite_unfitted_middle_layer_raises(model, lens):
    with pytest.raises(KeyError):
        j_lens_vectors_lite(lens, model, model.n_layers - 2 + 10, [4])


def test_patch_swap_rows_matches_per_row():
    mx.random.seed(21)
    v_s = mx.random.normal((D,))
    v_t = mx.random.normal((D,))
    h = mx.random.normal((5, D))
    V, inv = make_swap_basis(v_s, v_t)
    batched = patch_swap_rows(h, V, inv, alpha=1.0)
    mx.eval(batched)
    for r in range(5):
        single = patch_swap(h[r], v_s, v_t, alpha=1.0)
        mx.eval(single)
        assert np.allclose(np.array(batched[r]), np.array(single), atol=1e-4)


def test_patch_swap_rows_swaps_coords_and_preserves_orthogonal():
    mx.random.seed(22)
    v_s = mx.random.normal((D,))
    v_t = mx.random.normal((D,))
    h = mx.random.normal((3, D))
    V, inv = make_swap_basis(v_s, v_t)
    out = patch_swap_rows(h, V, inv, alpha=1.0)
    co_in = np.array(mx.matmul(inv, mx.matmul(V, h.T)))
    co_out = np.array(mx.matmul(inv, mx.matmul(V, out.T)))
    # Coordinates exchanged...
    assert np.allclose(co_out[0], co_in[1], atol=1e-3)
    assert np.allclose(co_out[1], co_in[0], atol=1e-3)
    # ...and the delta lies entirely in span{v_s, v_t}: projecting the
    # residual of (out - h) off the span leaves ~nothing.
    delta = np.array(out - h)
    Vn = np.array(V)
    coeff = np.linalg.lstsq(Vn.T, delta.T, rcond=None)[0]
    recon = (Vn.T @ coeff).T
    assert np.allclose(delta, recon, atol=1e-3)


def test_ablate_rows_zeroes_span_coordinates():
    mx.random.seed(23)
    V = mx.random.normal((4, D))
    G_inv = gram_inv(V)
    h = mx.random.normal((6, D)) * 3.0
    out = ablate_rows(h, V, G_inv, alpha=1.0)
    co_out = np.array(mx.matmul(G_inv, mx.matmul(V, out.T)))
    assert np.abs(co_out).max() < 1e-3
    # Idempotent (up to ridge): a second pass changes ~nothing.
    out2 = ablate_rows(out, V, G_inv, alpha=1.0)
    assert np.allclose(np.array(out), np.array(out2), atol=1e-3)


def test_compile_edits_steer(model, lens):
    edits = compile_edits(
        lens, model, mode="steer", layers=[0, model.n_layers - 1],
        token_id=3, alpha=5.0, from_pos=0,
    )
    assert [e.layer for e in edits] == [0, model.n_layers - 1]
    h = mx.zeros((2, D))
    full = np.array(model._lm_head.dense())
    # Fitted layer: alpha * (W_U[3] @ J_0); final layer: alpha * W_U[3].
    for e, expected in zip(edits, (full[3] @ lens.jacobians[0], full[3])):
        out = e.fn(h)
        mx.eval(out)
        assert np.allclose(np.array(out[0]), 5.0 * expected, rtol=3e-2, atol=3e-2)
        assert e.from_pos == 0 and e.positions is None


def test_compile_edits_swap_and_ablate_smoke(model, lens):
    swap = compile_edits(
        lens, model, mode="swap", layers=[1], token_id=2, target_id=9,
        alpha=1.0, positions=[4],
    )
    assert swap[0].positions == (4,)
    h = mx.random.normal((3, D))
    mx.eval(swap[0].fn(h))

    abl = compile_edits(
        lens, model, mode="ablate", layers=[1],
        ablate_token_ids=[1, 2, 3], from_pos=7,
    )
    out = abl[0].fn(h)
    mx.eval(out)
    assert out.shape == (3, D)


def test_compile_edits_error_branches(model, lens):
    with pytest.raises(ValueError):
        compile_edits(lens, model, mode="steer", layers=[0], from_pos=0)
    with pytest.raises(ValueError):
        compile_edits(lens, model, mode="swap", layers=[0], token_id=1, from_pos=0)
    with pytest.raises(ValueError):
        compile_edits(lens, model, mode="ablate", layers=[0], from_pos=0)
    with pytest.raises(ValueError):
        compile_edits(lens, model, mode="clamp", layers=[0], token_id=1, from_pos=0)
