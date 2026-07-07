"""Compatibility shim for mlx-lm 0.31.3 + transformers v5.

Two fixes applied at import time (this file is auto-loaded by Python via
the sitecustomize mechanism because it sits at the project root, which is
on ``sys.path`` when running via ``uv run``):

1. **AutoTokenizer.register string acceptance.** mlx-lm 0.31.3 calls
   ``AutoTokenizer.register("NewlineTokenizer", ...)`` passing a *string*
   as the config class. Transformers v5 changed ``register`` to require a
   real config *class* (it checks ``key.__module__``), which breaks
   mlx-lm at import time. We restore the pre-v5 string-accepting
   behavior.

2. **Suppress the "PyTorch was not found" advisory.** transformers v5
   prints this warning at import when torch isn't installed, but mlx-lm
   doesn't need torch (it uses MLX). We swallow just that one warning
   during the transformers import so the CLI output stays clean.

If a future mlx-lm release fixes either issue upstream, the
corresponding block becomes a no-op and can be removed.
"""

import contextlib
import io
import sys

# Fix 2: swallow the "PyTorch was not found" advisory during transformers
# import. We redirect stderr to capture it, then filter the line out.
_real_stderr = sys.stderr
class _FilterStream(io.TextIOBase):
    def __init__(self, underlying):
        self._underlying = underlying
        self._buf = ""
    def write(self, s):
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if "PyTorch was not found" not in line:
                self._underlying.write(line + "\n")
        return len(s)
    def flush(self):
        if self._buf and "PyTorch was not found" not in self._buf:
            self._underlying.write(self._buf)
        self._buf = ""
    def isatty(self):
        return self._underlying.isatty()
    @property
    def encoding(self):
        return getattr(self._underlying, "encoding", "utf-8")

sys.stderr = _FilterStream(_real_stderr)
try:
    import transformers
finally:
    sys.stderr = _real_stderr
    # Flush any pending output in the filter stream
    getattr(sys.stderr, "flush", lambda: None)()

from transformers.models.auto.tokenization_auto import (
    AutoTokenizer,
    TOKENIZER_MAPPING,
)

# Fix 1: restore string-accepting AutoTokenizer.register.
_orig_register = AutoTokenizer.register


def _patched_register(name, tokenizer_class=None, fast_tokenizer_class=None, exist_ok=False):
    if isinstance(name, str):
        candidate = tokenizer_class or fast_tokenizer_class
        if candidate is not None:
            try:
                TOKENIZER_MAPPING._extra_content[name] = candidate
            except AttributeError:
                return _orig_register(
                    name, tokenizer_class, fast_tokenizer_class, exist_ok=exist_ok
                )
        if fast_tokenizer_class is not None:
            from transformers.models.auto.tokenization_auto import (
                REGISTERED_TOKENIZER_CLASSES,
                REGISTERED_FAST_ALIASES,
            )
            registered_name = (
                candidate.__name__ if candidate else fast_tokenizer_class.__name__
            )
            REGISTERED_TOKENIZER_CLASSES[registered_name] = fast_tokenizer_class
            if tokenizer_class is not None:
                REGISTERED_FAST_ALIASES[tokenizer_class.__name__] = fast_tokenizer_class
        return
    return _orig_register(name, tokenizer_class, fast_tokenizer_class, exist_ok=exist_ok)


AutoTokenizer.register = _patched_register