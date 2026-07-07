"""Compatibility shim: make transformers v5 AutoTokenizer.register accept
string config_class names (mlx_lm 0.31.3 still passes strings).

Loaded automatically by Python via the sitecustomize mechanism (this dir is
on sys.path because it's the project root and uv runs from here).
"""

import transformers
from transformers.models.auto.tokenization_auto import AutoTokenizer, TOKENIZER_MAPPING

_orig_register = AutoTokenizer.register


def _patched_register(name, tokenizer_class=None, fast_tokenizer_class=None, exist_ok=False):
    if isinstance(name, str):
        # mlx_lm passes a string like "NewlineTokenizer" with no config class.
        # Register it under a synthetic config key in _extra_content so
        # __contains__ and __getattr__ don't crash, but skip the
        # `key.__module__` check that expects a real config class.
        from transformers.models.auto.auto_factory import _LazyAutoMapping

        # Stuff it into _extra_content with the string as the key; the
        # AutoTokenizer lookup path uses string keys too, so lookups still
        # work. The internal "is it a transformers native" check is skipped
        # because we never reach the branch that touches key.__module__.
        candidate = tokenizer_class or fast_tokenizer_class
        if candidate is not None:
            try:
                TOKENIZER_MAPPING._extra_content[name] = candidate
            except AttributeError:
                TOKENIZER_MAPPING.register(name, candidate, exist_ok=exist_ok)
        if fast_tokenizer_class is not None:
            from transformers.models.auto.tokenization_auto import (
                REGISTERED_TOKENIZER_CLASSES,
                REGISTERED_FAST_ALIASES,
            )
            REGISTERED_TOKENIZER_CLASSES[candidate.__name__ if candidate else fast_tokenizer_class.__name__] = fast_tokenizer_class
            if tokenizer_class is not None:
                REGISTERED_FAST_ALIASES[tokenizer_class.__name__] = fast_tokenizer_class
        return
    return _orig_register(name, tokenizer_class, fast_tokenizer_class, exist_ok=exist_ok)


AutoTokenizer.register = _patched_register