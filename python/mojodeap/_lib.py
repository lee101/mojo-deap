"""ctypes bridge to the Mojo variation kernels."""

from __future__ import annotations

import ctypes
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJODEAP_LIB") or os.path.join(
    ROOT, "dist", "libmojo-deap.so"
)

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mdeap_check_permutations": ([I, I, I, I, I], I),
    "mdeap_cx_uniform_f64": ([I, I, I, I], None),
    "mdeap_cx_uniform_i64": ([I, I, I, I], None),
    "mdeap_cx_uniform_u8": ([I, I, I, I], None),
    "mdeap_cx_pmx": ([I, I, I, I, I, I, I], None),
    "mdeap_cx_upmx": ([I, I, I, I, I, I], None),
    "mdeap_cx_ordered": ([I, I, I, I, I, I, I], None),
    "mdeap_cx_blend": ([I, I, I, I, F], None),
    "mdeap_cx_sbx": ([I, I, I, I, F], None),
    "mdeap_cx_sbx_bounded": ([I, I, I, I, I, I, I, I, F], None),
    "mdeap_mut_gaussian": ([I, I, I, I], None),
    "mdeap_mut_polynomial": ([I, I, I, I, I, I, F, I, F], I),
    "mdeap_mut_polynomial_scalar_bounds": ([I, I, I, I, F, I, F, F, F], I),
    "mdeap_mut_shuffle": ([I, I, I], None),
    "mdeap_mut_flip": ([I, I, I], None),
    "mdeap_mut_uniform_int": ([I, I, I, I], None),
    "mdeap_mut_es": ([I, I, I, I, I, I], None),
}

_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        if not os.path.exists(LIB):
            raise RuntimeError(
                f"Mojo library not found at {LIB}; run `pixi run build` first"
            )
        _library = ctypes.CDLL(LIB)
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_library, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _library


def addr(array: np.ndarray) -> int:
    address = int(array.ctypes.data)
    if array.size and address == 0:
        raise ValueError("non-empty NumPy buffer has a null data pointer")
    return address


def f64(values, *, copy: bool = False) -> np.ndarray:
    if copy:
        return np.array(values, dtype=np.float64, order="C", copy=True)
    return np.ascontiguousarray(values, dtype=np.float64)


def i64(values, *, copy: bool = False) -> np.ndarray:
    if copy:
        return np.array(values, dtype=np.int64, order="C", copy=True)
    return np.ascontiguousarray(values, dtype=np.int64)


def u8(values, *, copy: bool = False) -> np.ndarray:
    if copy:
        return np.array(values, dtype=np.uint8, order="C", copy=True)
    return np.ascontiguousarray(values, dtype=np.uint8)
