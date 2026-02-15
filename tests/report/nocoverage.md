# No-Coverage Notes

This file documents intentional `# pragma: no cover` usage.

## `nbux/rng.py`

- `_place_gauss_pl`
- `_2place_gauss`
- `_place_gauss_pl1`
- `place_uniform_pl`
- `_place_rademacher_pl`

Reason:
- These are parallel (`_pl`) RNG kernels (`@nbu.jtpc`) with environment-dependent runtime behavior and thread backend differences.
- `_2place_gauss` and `_place_gauss_pl1` are parallel-path helper wrappers retained for experimentation/examples around forced parallel selection.
- In this stage we are validating the stable Python/sync paths first, while leaving parallel-path coverage for a later dedicated pass.

## `nbux/algo/_dev.py`

- `lars1_constraintsolve_dev`

Reason:
- This is a development/experimental solver path and is intentionally excluded from default coverage while stage work focuses on stable public interfaces.
- A smoke test remains in `tests/unit/test_nbux_utils_algo_extra.py`, but it is marked skipped by default and can be enabled later when dev-path validation is resumed.

## `nbux/utils.py`

- `compiletime_parallelswitch`
- `stack_empty_impl`
- `stack_empty_`
- `_stack_empty`
- `nb_val_ptr`
- `nb_ptr_val`
- `buffer_nelems_andp`
- `_type_ref`
- `_if_val_cast`
- `_op_call`
- `_op_call_args`
- `_op_args`
- `_force_const`
- `_l_1_0`
- `_l_1_1`
- `_l_1_2`
- `_l_12_0`
- `_l_21_0`
- `_l_12_d`

Reason:
- These are intrinsic/overload/literal-dispatch helpers that primarily execute at Numba typing/lowering time, not as normal Python runtime paths.
- Several return pointer-backed values or request literal typing (`nb.literally`) and are not reliably attributable under Python-layer coverage.

## `nbux/_sort.py`

- `_sqleq_arg_`

Reason:
- This is an overload selector for `_sqleq_arg` unsafe modes. Its effective behavior is selected at compile-time and not consistently traceable in Python-layer coverage.

## `nbux/op/_misc.py`

- `mmul_cself` (BLAS_PACK stub branch)
- `cholesky_fsolve_inplace` (BLAS_PACK stub branch)
- `potrs` (BLAS_PACK stub branch)

Reason:
- These stubs are only active when `BLAS_PACK` is enabled. In the current environment the non-BLAS implementation branch is active and tested.
