from __future__ import annotations

import math as mt
from pathlib import Path
from typing import Any

import numba as nb
import numpy as np

import nbux
from tests.numba_layers import iter_function_layers

OUTPUT_DIR = Path(__file__).parent / "linesearch_output"


def _f1(x: float) -> float:
    return (x - 0.15) * (x - 1.35) * (x + 0.4) + 0.04 * mt.sin(9.0 * x)


def _f2(x: float) -> float:
    t = 3.0 * (x - 0.9)
    return (x - 0.9) + 0.2 * mt.tanh(t) + 0.05 * mt.sin(7.0 * x)


def _g2(x: float) -> float:
    t = 3.0 * (x - 0.9)
    c = mt.cosh(t)
    sech2 = 1.0 / (c * c)
    return 1.0 + 0.6 * sech2 + 0.35 * mt.cos(7.0 * x)


def _c2(x: float) -> float:
    t = 3.0 * (x - 0.9)
    c = mt.cosh(t)
    sech2 = 1.0 / (c * c)
    return -3.6 * mt.tanh(t) * sech2 - 2.45 * mt.sin(7.0 * x)


@nb.njit
def _f1_jit(x: float) -> float:
    return (x - 0.15) * (x - 1.35) * (x + 0.4) + 0.04 * mt.sin(9.0 * x)


@nb.njit
def _f2_jit(x: float) -> float:
    t = 3.0 * (x - 0.9)
    return (x - 0.9) + 0.2 * mt.tanh(t) + 0.05 * mt.sin(7.0 * x)


@nb.njit
def _g2_jit(x: float) -> float:
    t = 3.0 * (x - 0.9)
    c = mt.cosh(t)
    sech2 = 1.0 / (c * c)
    return 1.0 + 0.6 * sech2 + 0.35 * mt.cos(7.0 * x)


@nb.njit
def _c2_jit(x: float) -> float:
    t = 3.0 * (x - 0.9)
    c = mt.cosh(t)
    sech2 = 1.0 / (c * c)
    return -3.6 * mt.tanh(t) * sech2 - 2.45 * mt.sin(7.0 * x)


def _record_f1_py(x: float, xs: np.ndarray, ys: np.ndarray, idx: np.ndarray) -> float:
    i = int(idx[0])
    y = _f1(x)
    xs[i] = x
    ys[i] = y
    idx[0] = i + 1
    return y


@nb.njit
def _record_f1_jit(x: float, xs: np.ndarray, ys: np.ndarray, idx: np.ndarray) -> float:
    i = idx[0]
    y = _f1_jit(x)
    xs[i] = x
    ys[i] = y
    idx[0] = i + 1
    return y


def _record_f2_py(x: float, xs: np.ndarray, ys: np.ndarray, idx: np.ndarray) -> float:
    i = int(idx[0])
    y = _f2(x)
    xs[i] = x
    ys[i] = y
    idx[0] = i + 1
    return y


@nb.njit
def _record_f2_jit(x: float, xs: np.ndarray, ys: np.ndarray, idx: np.ndarray) -> float:
    i = idx[0]
    y = _f2_jit(x)
    xs[i] = x
    ys[i] = y
    idx[0] = i + 1
    return y


def _fmt_float(v: float) -> str:
    if np.isnan(v):
        return "nan"
    if np.isinf(v):
        return "-inf" if v < 0 else "inf"
    return f"{v:.12e}"


def _build_markdown(
    test_name: str,
    layer: str,
    descriptor: str,
    xs: np.ndarray,
    ys: np.ndarray,
    root: float,
    lo: float,
    hi: float,
    status: int,
    residual: float,
) -> str:
    abs_vals = np.abs(ys)
    log_abs = np.full(abs_vals.shape, -np.inf, dtype=np.float64)
    nz = abs_vals > 0.0
    log_abs[nz] = np.log10(abs_vals[nz])
    step_dx = np.full(xs.shape, np.nan, dtype=np.float64)
    if xs.size > 1:
        step_dx[1:] = np.abs(np.diff(xs))

    lines = [
        f"# {test_name} ({layer})",
        "",
        f"- Descriptor: {descriptor}",
        f"- Function evaluations tracked: {xs.size}",
        f"- Root estimate: {_fmt_float(root)}",
        f"- Final bracket: [{_fmt_float(lo)}, {_fmt_float(hi)}]",
        f"- Return status: {status}",
        f"- |f(root)|: {_fmt_float(abs(residual))}",
        "",
        "| eval | x | f(x) | log10(abs(f(x))) | step_dx |",
        "|---:|---:|---:|---:|---:|",
    ]
    for i in range(xs.size):
        lines.append(
            f"| {i} | {_fmt_float(xs[i])} | {_fmt_float(ys[i])} | {_fmt_float(log_abs[i])} | {_fmt_float(step_dx[i])} |"
        )
    return "\n".join(lines) + "\n"


def _run_and_write_report(
    *,
    test_name: str,
    algo_dispatcher: Any,
    bracket: tuple[float, float],
    kwargs: dict[str, Any],
    f_record_py: Any,
    f_record_jit: Any,
    f_eval: Any,
    g_py: Any = None,
    g_jit: Any = None,
    c_py: Any = None,
    c_jit: Any = None,
    descriptor: str,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lo0, hi0 = bracket
    max_rows = 512

    for layer, layer_fn in iter_function_layers(algo_dispatcher):
        xs = np.empty(max_rows, dtype=np.float64)
        ys = np.empty(max_rows, dtype=np.float64)
        idx = np.zeros(1, dtype=np.int64)
        f_op = (f_record_py, xs, ys, idx) if layer == "py" else (f_record_jit, xs, ys, idx)

        if g_py is None:
            root, lo, hi, status = layer_fn(f_op, lo0, hi0, **kwargs)
        elif c_py is None:
            g_op = g_py if layer == "py" else g_jit
            root, lo, hi, status = layer_fn(f_op, g_op, lo0, hi0, **kwargs)
        else:
            g_op = g_py if layer == "py" else g_jit
            c_op = c_py if layer == "py" else c_jit
            root, lo, hi, status = layer_fn(f_op, g_op, c_op, lo0, hi0, **kwargs)

        n = int(idx[0])
        assert n > 2
        xs_used = xs[:n].copy()
        ys_used = ys[:n].copy()
        assert np.isfinite(root)
        assert min(lo, hi) <= root <= max(lo, hi)
        assert status in (0, 1, 2)
        residual = float(f_eval(float(root)))
        assert abs(residual) < 1e-6

        md = _build_markdown(
            test_name=test_name,
            layer=layer,
            descriptor=descriptor,
            xs=xs_used,
            ys=ys_used,
            root=float(root),
            lo=float(lo),
            hi=float(hi),
            status=int(status),
            residual=residual,
        )
        out = OUTPUT_DIR / f"{test_name}_{layer}.md"
        out.write_text(md, encoding="utf-8")
        assert out.exists()


def test_signedroot_secant_convergence_report() -> None:
    """Record and report secant signed-root convergence with nontrivial polynomial-sine structure."""
    _run_and_write_report(
        test_name="test_signedroot_secant_convergence_report",
        algo_dispatcher=nbux.algo.signedroot_secant,
        bracket=(0.2, 1.8),
        kwargs=dict(br_rate=0.45, max_iters=70, sign=1, eager=True, fallb=True),
        f_record_py=_record_f1_py,
        f_record_jit=_record_f1_jit,
        f_eval=_f1,
        descriptor="f(x)=(x-0.15)(x-1.35)(x+0.4)+0.04*sin(9x), signed secant with fallback",
    )


def test_signedroot_quadinterp_convergence_report() -> None:
    """Record and report quadratic interpolation signed-root convergence on the same difficult surface."""
    _run_and_write_report(
        test_name="test_signedroot_quadinterp_convergence_report",
        algo_dispatcher=nbux.algo.signedroot_quadinterp,
        bracket=(0.2, 1.8),
        kwargs=dict(br_rate=0.45, max_iters=70, sign=1, eager=True),
        f_record_py=_record_f1_py,
        f_record_jit=_record_f1_jit,
        f_eval=_f1,
        descriptor="f(x)=(x-0.15)(x-1.35)(x+0.4)+0.04*sin(9x), signed quadratic interpolation",
    )


def test_signedroot_newton_convergence_report() -> None:
    """Record and report signed Newton convergence for a tanh-sine mixed nonlinearity."""
    _run_and_write_report(
        test_name="test_signedroot_newton_convergence_report",
        algo_dispatcher=nbux.algo.signedroot_newton,
        bracket=(0.0, 1.8),
        kwargs=dict(br_rate=0.45, max_iters=70, sign=1, eager=True),
        f_record_py=_record_f2_py,
        f_record_jit=_record_f2_jit,
        f_eval=_f2,
        g_py=_g2,
        g_jit=_g2_jit,
        descriptor="f(x)=(x-0.9)+0.2*tanh(3(x-0.9))+0.05*sin(7x), bracketed signed Newton",
    )


def test_signseeking_halley_convergence_report() -> None:
    """Record and report signseeking Halley convergence using first and second derivative operators."""
    _run_and_write_report(
        test_name="test_signseeking_halley_convergence_report",
        algo_dispatcher=nbux.algo.signseeking_halley,
        bracket=(0.0, 1.8),
        kwargs=dict(br_rate=0.45, max_iters=70, sign=1, eager=True),
        f_record_py=_record_f2_py,
        f_record_jit=_record_f2_jit,
        f_eval=_f2,
        g_py=_g2,
        g_jit=_g2_jit,
        c_py=_c2,
        c_jit=_c2_jit,
        descriptor="f(x)=(x-0.9)+0.2*tanh(3(x-0.9))+0.05*sin(7x), signseeking Halley with curvature",
    )
