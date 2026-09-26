"""DEAP-compatible crossover and mutation operators.

The public functions intentionally use DEAP's names, signatures, in-place
semantics, return tuples, and module-level :mod:`random` source.
"""

from __future__ import annotations

import math
import random
import warnings
from collections.abc import Sequence
from itertools import repeat

import numpy as np

from ._lib import addr, f64, i64, lib, u8


def _copy_back(target, values: np.ndarray) -> None:
    if isinstance(target, np.ndarray):
        target[: len(values)] = values
    else:
        for index, value in enumerate(values):
            target[index] = value.item()


def _float_work(values, size: int | None = None):
    n = len(values) if size is None else size
    if (
        isinstance(values, np.ndarray)
        and values.dtype == np.float64
        and values.ndim == 1
        and values.flags.c_contiguous
        and values.flags.writeable
    ):
        return values[:n], None
    array = f64(values[:n], copy=True)
    return array, lambda: _copy_back(values, array)


def _int_work(values, size: int | None = None):
    n = len(values) if size is None else size
    if (
        isinstance(values, np.ndarray)
        and values.dtype == np.int64
        and values.ndim == 1
        and values.flags.c_contiguous
        and values.flags.writeable
    ):
        return values[:n], None
    if isinstance(values, np.ndarray) and np.issubdtype(
        values.dtype, np.unsignedinteger
    ):
        maximum = values[:n].max(initial=0)
        if maximum > np.iinfo(np.int64).max:
            raise OverflowError("integer gene cannot be represented as int64")
    array = i64(values[:n], copy=True)
    return array, lambda: _copy_back(values, array)


def _finish(*callbacks) -> None:
    for callback in callbacks:
        if callback is not None:
            callback()


def _validate_index_permutations(first: np.ndarray, second: np.ndarray) -> None:
    """Keep native scratch indexing inside [0, n) for DEAP permutations."""
    size = len(first)
    expected = np.arange(size, dtype=np.int64)
    if not np.array_equal(np.sort(first), expected) or not np.array_equal(
        np.sort(second), expected
    ):
        raise ValueError(
            "permutation crossover requires each individual to contain "
            "every integer in range(len(shorter_individual)) exactly once"
        )


def _parameter(value, size: int, name: str):
    if not isinstance(value, Sequence):
        return np.full(size, value, dtype=np.float64)
    if len(value) < size:
        raise IndexError(
            f"{name} must be at least the size of individual: {len(value)} < {size}"
        )
    return f64(value[:size], copy=True)


def _bounded_parameter(value, size: int, name: str, shorter: bool = False):
    if not isinstance(value, Sequence):
        return np.full(size, value, dtype=np.float64)
    if len(value) < size:
        subject = "the shorter individual" if shorter else "individual"
        raise IndexError(
            f"{name} must be at least the size of {subject}: {len(value)} < {size}"
        )
    return f64(value[:size], copy=True)


def cxOnePoint(ind1, ind2):
    size = min(len(ind1), len(ind2))
    cxpoint = random.randint(1, size - 1)
    ind1[cxpoint:], ind2[cxpoint:] = ind2[cxpoint:], ind1[cxpoint:]
    return ind1, ind2


def cxTwoPoint(ind1, ind2):
    size = min(len(ind1), len(ind2))
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1
    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] = (
        ind2[cxpoint1:cxpoint2],
        ind1[cxpoint1:cxpoint2],
    )
    return ind1, ind2


def cxTwoPoints(ind1, ind2):
    warnings.warn(
        "tools.cxTwoPoints has been renamed. Use cxTwoPoint instead.",
        FutureWarning,
    )
    return cxTwoPoint(ind1, ind2)


def cxUniform(ind1, ind2, indpb):
    size = min(len(ind1), len(ind2))
    mask = np.fromiter(
        (random.random() < indpb for _ in range(size)),
        dtype=np.uint8,
        count=size,
    )
    if (
        isinstance(ind1, np.ndarray)
        and isinstance(ind2, np.ndarray)
        and ind1.dtype == ind2.dtype
        and ind1.ndim == ind2.ndim == 1
        and ind1.flags.c_contiguous
        and ind2.flags.c_contiguous
        and ind1.flags.writeable
        and ind2.flags.writeable
    ):
        suffix = {
            np.dtype(np.float64): "f64",
            np.dtype(np.int64): "i64",
            np.dtype(np.uint8): "u8",
            np.dtype(np.bool_): "u8",
        }.get(ind1.dtype)
        if suffix is not None and size:
            getattr(lib(), f"mdeap_cx_uniform_{suffix}")(
                addr(ind1), addr(ind2), addr(mask), size
            )
            return ind1, ind2
    for index in np.flatnonzero(mask):
        ind1[index], ind2[index] = ind2[index], ind1[index]
    return ind1, ind2


def cxPartialyMatched(ind1, ind2):
    size = min(len(ind1), len(ind2))
    first, finish_first = _int_work(ind1, size)
    second, finish_second = _int_work(ind2, size)
    positions1 = np.empty(size, dtype=np.int64)
    positions2 = np.empty(size, dtype=np.int64)
    _validate_index_permutations(first, second)
    cxpoint1 = random.randint(0, size)
    cxpoint2 = random.randint(0, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1
    lib().mdeap_cx_pmx(
        addr(first),
        addr(second),
        addr(positions1),
        addr(positions2),
        size,
        cxpoint1,
        cxpoint2,
    )
    _finish(finish_first, finish_second)
    return ind1, ind2


def cxUniformPartialyMatched(ind1, ind2, indpb):
    size = min(len(ind1), len(ind2))
    first, finish_first = _int_work(ind1, size)
    second, finish_second = _int_work(ind2, size)
    positions1 = np.empty(size, dtype=np.int64)
    positions2 = np.empty(size, dtype=np.int64)
    _validate_index_permutations(first, second)
    mask = np.fromiter(
        (random.random() < indpb for _ in range(size)),
        dtype=np.uint8,
        count=size,
    )
    lib().mdeap_cx_upmx(
        addr(first),
        addr(second),
        addr(positions1),
        addr(positions2),
        addr(mask),
        size,
    )
    _finish(finish_first, finish_second)
    return ind1, ind2


def cxOrdered(ind1, ind2):
    size = min(len(ind1), len(ind2))
    left, right = sorted(random.sample(range(size), 2))
    first, finish_first = _int_work(ind1, size)
    second, finish_second = _int_work(ind2, size)
    holes1 = np.empty(size, dtype=np.uint8)
    holes2 = np.empty(size, dtype=np.uint8)
    _validate_index_permutations(first, second)
    lib().mdeap_cx_ordered(
        addr(first),
        addr(second),
        addr(holes1),
        addr(holes2),
        size,
        left,
        right,
    )
    _finish(finish_first, finish_second)
    return ind1, ind2


def cxBlend(ind1, ind2, alpha):
    size = min(len(ind1), len(ind2))
    first, finish_first = _float_work(ind1, size)
    second, finish_second = _float_work(ind2, size)
    random_values = np.fromiter(
        (random.random() for _ in range(size)), dtype=np.float64, count=size
    )
    if size:
        lib().mdeap_cx_blend(
            addr(first), addr(second), addr(random_values), size, alpha
        )
    _finish(finish_first, finish_second)
    return ind1, ind2


def cxSimulatedBinary(ind1, ind2, eta):
    size = min(len(ind1), len(ind2))
    first, finish_first = _float_work(ind1, size)
    second, finish_second = _float_work(ind2, size)
    random_values = np.fromiter(
        (random.random() for _ in range(size)), dtype=np.float64, count=size
    )
    if size:
        lib().mdeap_cx_sbx(
            addr(first), addr(second), addr(random_values), size, eta
        )
    _finish(finish_first, finish_second)
    return ind1, ind2


def cxSimulatedBinaryBounded(ind1, ind2, eta, low, up):
    size = min(len(ind1), len(ind2))
    first, finish_first = _float_work(ind1, size)
    second, finish_second = _float_work(ind2, size)
    lower = _bounded_parameter(low, size, "low", shorter=True)
    upper = _bounded_parameter(up, size, "up", shorter=True)
    active = np.zeros(size, dtype=np.uint8)
    swaps = np.zeros(size, dtype=np.uint8)
    random_values = np.zeros(size, dtype=np.float64)
    for index in range(size):
        if random.random() <= 0.5 and abs(first[index] - second[index]) > 1e-14:
            active[index] = 1
            random_values[index] = random.random()
            swaps[index] = random.random() <= 0.5
    if size:
        lib().mdeap_cx_sbx_bounded(
            addr(first),
            addr(second),
            addr(lower),
            addr(upper),
            addr(random_values),
            addr(active),
            addr(swaps),
            size,
            eta,
        )
    _finish(finish_first, finish_second)
    return ind1, ind2


def cxMessyOnePoint(ind1, ind2):
    cxpoint1 = random.randint(0, len(ind1))
    cxpoint2 = random.randint(0, len(ind2))
    ind1[cxpoint1:], ind2[cxpoint2:] = ind2[cxpoint2:], ind1[cxpoint1:]
    return ind1, ind2


def cxESBlend(ind1, ind2, alpha):
    size = min(len(ind1), len(ind2), len(ind1.strategy), len(ind2.strategy))
    first, finish_first = _float_work(ind1, size)
    second, finish_second = _float_work(ind2, size)
    strategy1, finish_strategy1 = _float_work(ind1.strategy, size)
    strategy2, finish_strategy2 = _float_work(ind2.strategy, size)
    values_random = np.empty(size, dtype=np.float64)
    strategy_random = np.empty(size, dtype=np.float64)
    for index in range(size):
        values_random[index] = random.random()
        strategy_random[index] = random.random()
    lib().mdeap_cx_blend(
        addr(first), addr(second), addr(values_random), size, alpha
    )
    lib().mdeap_cx_blend(
        addr(strategy1),
        addr(strategy2),
        addr(strategy_random),
        size,
        alpha,
    )
    _finish(
        finish_first, finish_second, finish_strategy1, finish_strategy2
    )
    return ind1, ind2


def cxESTwoPoint(ind1, ind2):
    size = min(len(ind1), len(ind2))
    point1 = random.randint(1, size)
    point2 = random.randint(1, size - 1)
    if point2 >= point1:
        point2 += 1
    else:
        point1, point2 = point2, point1
    ind1[point1:point2], ind2[point1:point2] = (
        ind2[point1:point2],
        ind1[point1:point2],
    )
    ind1.strategy[point1:point2], ind2.strategy[point1:point2] = (
        ind2.strategy[point1:point2],
        ind1.strategy[point1:point2],
    )
    return ind1, ind2


def cxESTwoPoints(ind1, ind2):
    return cxESTwoPoint(ind1, ind2)


def mutGaussian(individual, mu, sigma, indpb):
    size = len(individual)
    means = _parameter(mu, size, "mu")
    deviations = _parameter(sigma, size, "sigma")
    values, finish = _float_work(individual)
    active = np.zeros(size, dtype=np.uint8)
    delta = np.zeros(size, dtype=np.float64)
    for index in range(size):
        if random.random() < indpb:
            active[index] = 1
            delta[index] = random.gauss(means[index], deviations[index])
    if size:
        lib().mdeap_mut_gaussian(addr(values), addr(delta), addr(active), size)
    _finish(finish)
    return (individual,)


def mutPolynomialBounded(individual, eta, low, up, indpb, device="cpu"):
    if device not in ("cpu", "gpu"):
        raise ValueError("device must be 'cpu' or 'gpu'")
    size = len(individual)
    scalar_bounds = not isinstance(low, Sequence) and not isinstance(up, Sequence)
    if not scalar_bounds:
        lower = _bounded_parameter(low, size, "low")
        upper = _bounded_parameter(up, size, "up")
    values, finish = _float_work(individual)
    if not size:
        _finish(finish)
        return (individual,)
    version, internal_state, gaussian = random.getstate()
    mt_state = np.asarray(internal_state[:-1], dtype=np.uint32)
    random_values = np.empty(size, dtype=np.float64)
    next_index = lib().mdeap_fill_polynomial_random(
        addr(random_values),
        addr(mt_state),
        size,
        indpb,
        internal_state[-1],
    )
    random.setstate(
        (version, tuple(map(int, mt_state)) + (next_index,), gaussian)
    )
    active_indices = np.flatnonzero(random_values >= 0)
    if active_indices.size:
        if scalar_bounds:
            device_bytes = (
                values.nbytes + random_values.nbytes + active_indices.nbytes
            )
            # `device="gpu"` is retained for API compatibility. This toolchain
            # ships no GPU host API, so the kernel runs serially on the CPU;
            # see README, "CPU-only by necessity".
            if device == "gpu":
                if device_bytes >= 2_000_000_000:
                    raise MemoryError(
                        "GPU polynomial mutation buffers would exceed 2 GB"
                    )
                status = lib().mdeap_mut_polynomial_scalar_bounds_gpu(
                        addr(values),
                        addr(random_values),
                        addr(active_indices),
                        size,
                        active_indices.size,
                        eta,
                        low,
                        up,
                    )
                if status != 1:
                    raise RuntimeError(
                        "GPU polynomial mutation failed or has insufficient "
                        "free device memory"
                    )
            else:
                lib().mdeap_mut_polynomial_scalar_bounds(
                    addr(values),
                    addr(random_values),
                    addr(active_indices),
                    active_indices.size,
                    eta,
                    low,
                    up,
                )
        else:
            lib().mdeap_mut_polynomial(
                addr(values),
                addr(lower),
                addr(upper),
                addr(random_values),
                addr(active_indices),
                active_indices.size,
                eta,
            )
    _finish(finish)
    return (individual,)


def mutShuffleIndexes(individual, indpb):
    size = len(individual)
    values, finish = _int_work(individual)
    targets = np.full(size, -1, dtype=np.int64)
    for index in range(size):
        if random.random() < indpb:
            target = random.randint(0, size - 2)
            if target >= index:
                target += 1
            targets[index] = target
    if size:
        lib().mdeap_mut_shuffle(addr(values), addr(targets), size)
    _finish(finish)
    return (individual,)


def mutFlipBit(individual, indpb):
    size = len(individual)
    active = np.fromiter(
        (random.random() < indpb for _ in range(size)),
        dtype=np.uint8,
        count=size,
    )
    if (
        isinstance(individual, np.ndarray)
        and individual.dtype in (np.dtype(np.bool_), np.dtype(np.uint8))
        and individual.ndim == 1
        and individual.flags.c_contiguous
        and individual.flags.writeable
    ):
        if size:
            lib().mdeap_mut_flip(addr(individual), addr(active), size)
    else:
        for index in np.flatnonzero(active):
            individual[index] = type(individual[index])(not individual[index])
    return (individual,)


def mutUniformInt(individual, low, up, indpb):
    size = len(individual)
    if not isinstance(low, Sequence):
        lower = repeat(low, size)
    elif len(low) < size:
        raise IndexError(
            f"low must be at least the size of individual: {len(low)} < {size}"
        )
    else:
        lower = low
    if not isinstance(up, Sequence):
        upper = repeat(up, size)
    elif len(up) < size:
        raise IndexError(
            f"up must be at least the size of individual: {len(up)} < {size}"
        )
    else:
        upper = up
    values, finish = _int_work(individual)
    active = np.zeros(size, dtype=np.uint8)
    replacements = np.zeros(size, dtype=np.int64)
    for index, lo, hi in zip(range(size), lower, upper):
        if random.random() < indpb:
            active[index] = 1
            replacements[index] = random.randint(lo, hi)
    if size:
        lib().mdeap_mut_uniform_int(
            addr(values), addr(replacements), addr(active), size
        )
    _finish(finish)
    return (individual,)


def mutInversion(individual):
    size = len(individual)
    if size == 0:
        return (individual,)
    index_one = random.randrange(size)
    index_two = random.randrange(size)
    start = min(index_one, index_two)
    end = max(index_one, index_two)
    individual[start:end] = individual[start:end][::-1]
    return (individual,)


def mutESLogNormal(individual, c, indpb):
    size = len(individual)
    t = c / math.sqrt(2.0 * math.sqrt(size))
    t0 = c / math.sqrt(2.0 * size)
    common = t0 * random.gauss(0, 1)
    values, finish_values = _float_work(individual)
    strategy, finish_strategy = _float_work(individual.strategy)
    active = np.zeros(size, dtype=np.uint8)
    factors = np.zeros(size, dtype=np.float64)
    normals = np.zeros(size, dtype=np.float64)
    for index in range(size):
        if random.random() < indpb:
            active[index] = 1
            factors[index] = math.exp(common + t * random.gauss(0, 1))
            normals[index] = random.gauss(0, 1)
    lib().mdeap_mut_es(
        addr(values),
        addr(strategy),
        addr(factors),
        addr(normals),
        addr(active),
        size,
    )
    _finish(finish_values, finish_strategy)
    return (individual,)


def _batch_pair(first, second):
    a = np.asarray(first)
    b = np.asarray(second)
    if a.shape != b.shape:
        raise ValueError("paired batches must have the same shape")
    if (
        a.dtype != np.float64
        or b.dtype != np.float64
        or not a.flags.c_contiguous
        or not b.flags.c_contiguous
        or not a.flags.writeable
        or not b.flags.writeable
    ):
        raise TypeError("batch arrays must be writable C-contiguous float64")
    return a, b


def cxBlendBatch(ind1, ind2, alpha, rng=None):
    """Apply ``cxBlend`` elementwise to two same-shaped float64 batches."""
    first, second = _batch_pair(ind1, ind2)
    generator = np.random.default_rng() if rng is None else rng
    random_values = np.ascontiguousarray(
        generator.random(first.size), dtype=np.float64
    )
    if first.size:
        lib().mdeap_cx_blend(
            addr(first), addr(second), addr(random_values), first.size, alpha
        )
    return first, second


def cxSimulatedBinaryBatch(ind1, ind2, eta, rng=None):
    """Apply ``cxSimulatedBinary`` elementwise to float64 batches."""
    first, second = _batch_pair(ind1, ind2)
    generator = np.random.default_rng() if rng is None else rng
    random_values = np.ascontiguousarray(
        generator.random(first.size), dtype=np.float64
    )
    if first.size:
        lib().mdeap_cx_sbx(
            addr(first), addr(second), addr(random_values), first.size, eta
        )
    return first, second


def mutGaussianBatch(population, mu, sigma, indpb, rng=None):
    """Apply independent Gaussian mutation to a C-contiguous float64 batch."""
    values = np.asarray(population)
    if (
        values.dtype != np.float64
        or not values.flags.c_contiguous
        or not values.flags.writeable
    ):
        raise TypeError("population must be writable C-contiguous float64")
    generator = np.random.default_rng() if rng is None else rng
    active = np.ascontiguousarray(
        generator.random(values.size) < indpb, dtype=np.uint8
    )
    delta = np.ascontiguousarray(
        generator.normal(mu, sigma, values.size), dtype=np.float64
    )
    if values.size:
        lib().mdeap_mut_gaussian(
            addr(values), addr(delta), addr(active), values.size
        )
    return (values,)


__all__ = [
    "cxOnePoint",
    "cxTwoPoint",
    "cxTwoPoints",
    "cxUniform",
    "cxPartialyMatched",
    "cxUniformPartialyMatched",
    "cxOrdered",
    "cxBlend",
    "cxSimulatedBinary",
    "cxSimulatedBinaryBounded",
    "cxMessyOnePoint",
    "cxESBlend",
    "cxESTwoPoint",
    "cxESTwoPoints",
    "mutGaussian",
    "mutPolynomialBounded",
    "mutShuffleIndexes",
    "mutFlipBit",
    "mutUniformInt",
    "mutInversion",
    "mutESLogNormal",
    "cxBlendBatch",
    "cxSimulatedBinaryBatch",
    "mutGaussianBatch",
]
