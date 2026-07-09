"""Lens selection enforcement (serve.resolve_lens_path).

The server must never silently fall back to lens-less mode — the failure
that shipped a one-column workspace band. Every branch of the resolver
is pinned here; these tests are pure logic (no model, milliseconds).

    uv run python -m pytest tests/test_lens_resolution.py -v
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jlens_qwen.serve import LensResolutionError, resolve_lens_path


def _touch(path):
    path.write_bytes(b"x")
    return str(path)


def test_explicit_path_exists(tmp_path):
    p = _touch(tmp_path / "my_lens.npz")
    assert resolve_lens_path(p, str(tmp_path)) == p


def test_explicit_path_missing_refuses(tmp_path):
    _touch(tmp_path / "other.npz")
    with pytest.raises(LensResolutionError) as e:
        resolve_lens_path(str(tmp_path / "typo.npz"), str(tmp_path))
    msg = str(e.value)
    assert "typo.npz" in msg          # names the wrong path
    assert "other.npz" in msg         # lists what IS available
    assert "JLENS_PATH=none" in msg   # offers the explicit lens-less option


@pytest.mark.parametrize("token", ["none", "NONE", "logit", " none "])
def test_explicit_lensless_mode(tmp_path, token):
    assert resolve_lens_path(token, str(tmp_path)) is None


def test_default_lens_npz_wins(tmp_path):
    default = _touch(tmp_path / "lens.npz")
    _touch(tmp_path / "another.npz")  # ambiguity is irrelevant when default exists
    assert resolve_lens_path(None, str(tmp_path)) == default


def test_single_candidate_autoselected(tmp_path, capsys):
    only = _touch(tmp_path / "full_depth.npz")
    assert resolve_lens_path(None, str(tmp_path)) == only
    assert "auto-selected" in capsys.readouterr().out  # loud, not silent


def test_multiple_candidates_refuse_to_guess(tmp_path):
    a = _touch(tmp_path / "a.npz")
    b = _touch(tmp_path / "b.npz")
    with pytest.raises(LensResolutionError) as e:
        resolve_lens_path(None, str(tmp_path))
    msg = str(e.value)
    assert a in msg and b in msg
    assert "JLENS_PATH=none" in msg


def test_empty_dir_refuses_with_instructions(tmp_path):
    with pytest.raises(LensResolutionError) as e:
        resolve_lens_path(None, str(tmp_path))
    msg = str(e.value)
    assert "run_fit" in msg or "README" in msg
    assert "JLENS_PATH=none" in msg


def test_non_npz_files_are_not_candidates(tmp_path):
    _touch(tmp_path / "checkpoint.npy")
    _touch(tmp_path / "fit.log")
    with pytest.raises(LensResolutionError):
        resolve_lens_path(None, str(tmp_path))
