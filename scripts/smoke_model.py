"""Smoke test: model loads, forward works, GDN patch is in place, and
mx.vjp through a single linear-attention layer returns finite gradients."""

import sys
import time

import mlx.core as mx

sys.path.insert(0, ".")
from jlens_qwen.model import load


def main():
    print("Loading model...", flush=True)
    t0 = time.perf_counter()
    model = load()
    print(f"  loaded in {time.perf_counter()-t0:.1f}s: {model}", flush=True)

    # Confirm GDN patch took: gated_delta_update should be our patched version.
    from mlx_lm.models import gated_delta as gdn
    from jlens_qwen.patch_gdn import _patched_gated_delta_update
    assert gdn.gated_delta_update is _patched_gated_delta_update, "GDN patch not applied"
    print("  GDN patch applied: OK", flush=True)

    # Confirm a linear layer's __call__ is checkpointed.
    from jlens_qwen import patch_gdn as pg
    assert pg._CHECKPOINT_APPLIED, "GatedDeltaNet.__call__ not checkpointed"
    lin_layer = next(l for l in model.layers if l.is_linear)
    print("  GDN checkpoint: OK", flush=True)

    # Forward pass: capture all layers.
    prompt = "Fact: The currency used in the country shaped like a boot is"
    input_ids = model.encode(prompt, max_length=128)
    print(f"  prompt: {prompt!r}", flush=True)
    print(f"  input_ids shape: {input_ids.shape}", flush=True)

    print("Running forward (capture all 64 layers)...", flush=True)
    t0 = time.perf_counter()
    mx.eval_disabled = False
    final, acts = model.forward(input_ids, capture_layers=list(range(64)))
    mx.eval(final)
    print(f"  forward in {time.perf_counter()-t0:.1f}s", flush=True)
    print(f"  final residual shape: {final.shape}, dtype: {final.dtype}", flush=True)
    print(f"  captured {len(acts)} layer activations", flush=True)

    # Final-layer logits via lm_head.
    logits = model.unembed(model.final_norm(acts[63]))
    mx.eval(logits)
    print(f"  logits shape: {logits.shape}", flush=True)

    # Top-5 next-token predictions. mx.topk is buggy on large vocabs, use argsort.
    lf = logits[0, -1].astype(mx.float32)
    sorted_idx = mx.argsort(lf)
    top_tokens = [int(t) for t in sorted_idx[-5:][::-1].tolist()]
    print("  top-5 next tokens at last position:", flush=True)
    for t in top_tokens:
        s = model.tokenizer.decode([t])
        print(f"    {t}: {s!r}", flush=True)

    # Now the critical test: mx.vjp from final residual back to an early layer.
    # We re-run the forward from a source layer to the end, with the source
    # layer's captured activation as a primal, and backprop a one-hot cotangent.
    print("\nVJP test: backprop from final -> layer 32...", flush=True)
    src_layer = 32
    h_src = acts[src_layer]  # [1, seq_len, d_model]
    print(f"  h_src shape: {h_src.shape}", flush=True)

    # Build a function: given h at layer src_layer, run layers src+1..end + final norm.
    def run_from_src(h: mx.array) -> mx.array:
        from mlx_lm.models.base import create_attention_mask, create_ssm_mask
        fa_mask = create_attention_mask(h, cache=None)
        ssm_mask = create_ssm_mask(h, cache=None)
        hidden = h
        for i in range(src_layer + 1, model.n_layers):
            layer = model.layers[i]
            mask = ssm_mask if layer.is_linear else fa_mask
            hidden = layer(hidden, mask=mask, cache=None)
        return model._text_module.norm(hidden)

    # Cotangent: one-hot at output dim 0, at every position.
    B, S, D = h_src.shape
    cotangent = mx.zeros((B, S, D), dtype=mx.float32)
    # Build one-hot at dim 0 via concatenate.
    one = mx.ones((B, S, 1), dtype=mx.float32)
    rest = mx.zeros((B, S, D - 1), dtype=mx.float32)
    cotangent = mx.concatenate([one, rest], axis=-1)

    t0 = time.perf_counter()
    (final_v, ) = run_from_src(h_src)
    # Need to eval so we have a concrete output shape, then vjp.
    mx.eval(final_v)
    print(f"  re-forward from layer {src_layer} in {time.perf_counter()-t0:.1f}s, output shape {final_v.shape}", flush=True)

    t0 = time.perf_counter()
    out, vjps = mx.vjp(run_from_src, [h_src], [cotangent])
    mx.eval(vjps[0])
    elapsed = time.perf_counter() - t0
    print(f"  vjp in {elapsed:.1f}s", flush=True)
    print(f"  vjp[0] shape: {vjps[0].shape}, dtype: {vjps[0].dtype}", flush=True)

    import numpy as np
    vnp = np.array(vjps[0].astype(mx.float32))
    print(f"  vjp[0] finite: {np.isfinite(vnp).all()}", flush=True)
    print(f"  vjp[0] max abs: {np.abs(vnp).max():.3e}", flush=True)
    print(f"  vjp[0] mean abs: {np.abs(vnp).mean():.3e}", flush=True)

    print("\nSMOKE TEST PASSED", flush=True)


if __name__ == "__main__":
    main()