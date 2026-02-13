from __future__ import annotations

import importlib
import os
import pkgutil
from collections.abc import Callable, Iterator
from typing import Any

from numba.core.registry import CPUDispatcher

import nbux

_LAYER_RAW = (os.getenv("NUMBA_TEST_LAYER") or "all").strip().lower()
if _LAYER_RAW in {"", "all"}:
    NUMBA_TEST_LAYER = "all"
elif _LAYER_RAW in {"python", "py"}:
    NUMBA_TEST_LAYER = "py"
elif _LAYER_RAW in {"jit", "numba"}:
    NUMBA_TEST_LAYER = "jit"
else:
    NUMBA_TEST_LAYER = "all"

_CACHE_RAW = (os.getenv("NUMBA_TEST_CACHE") or "false").strip().lower()
NUMBA_TEST_CACHE = _CACHE_RAW == "true"

RUN_PY = NUMBA_TEST_LAYER in {"all", "py"}
RUN_JIT = NUMBA_TEST_LAYER in {"all", "jit"}


def _py_first_runner(func: Callable[..., Any]) -> Callable[..., Any]:
    """Run py_func first, then fall back to dispatcher if py path raises."""

    def _runner(*args: Any, **kwargs: Any) -> Any:
        try:
            return func.py_func(*args, **kwargs)
        except Exception:
            return func(*args, **kwargs)

    return _runner


def iter_function_layers(func: Callable[..., Any]) -> Iterator[tuple[str, Callable[..., Any]]]:
    """Yield active callable layers for a function under NUMBA_TEST_LAYER."""
    if hasattr(func, "py_func"):
        if RUN_PY:
            yield "py", _py_first_runner(func)
        if RUN_JIT:
            yield "jit", func
        return

    # Register-jitable and normal python callables run once.
    if RUN_PY or not RUN_JIT:
        yield "py", func
    elif RUN_JIT:
        yield "jit", func


def _iter_nbux_modules() -> Iterator[Any]:
    """Iterate imported nbux modules for dispatcher cache operations."""
    yield nbux
    for mod_info in pkgutil.walk_packages(nbux.__path__, prefix=f"{nbux.__name__}."):
        yield importlib.import_module(mod_info.name)


def reset_nbux_numba_cache() -> int:
    """Reset numba dispatcher in-memory caches for nbux modules."""
    count = 0
    for module in _iter_nbux_modules():
        for obj in vars(module).values():
            if isinstance(obj, CPUDispatcher):
                clear = getattr(obj, "_clear", None)
                if callable(clear):
                    clear()
                    count += 1
    return count
