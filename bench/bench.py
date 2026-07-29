"""Honest operator benchmarks against upstream DEAP."""

from __future__ import annotations

import math
import os
import platform
import random
import sys
import time

import numpy as np

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"
    ),
)

from deap import tools as deap_tools  # noqa: E402
from mojodeap import tools  # noqa: E402


def timeit(function, repeat=3):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


CASES = []


def benchmark(name):
    def decorate(builder):
        CASES.append((name, builder))
        return builder

    return decorate


@benchmark("PMX permutation crossover, 500k genes")
def pmx():
    first = np.arange(500_000, dtype=np.int64)
    second = first[::-1].copy()

    def mojo():
        a, b = first.copy(), second.copy()
        random.seed(10)
        tools.cxPartialyMatched(a, b)

    def upstream():
        a, b = first.copy(), second.copy()
        random.seed(10)
        deap_tools.cxPartialyMatched(a, b)

    return mojo, upstream


@benchmark("BLX-alpha batch, 2m genes")
def blend_batch():
    first = np.linspace(-2, 2, 2_000_000).reshape(2000, 1000)
    second = first[::-1].copy()

    def mojo():
        a, b = first.copy(), second.copy()
        tools.cxBlendBatch(a, b, 0.3, rng=np.random.default_rng(10))

    def upstream():
        a, b = first.copy(), second.copy()
        random.seed(10)
        for row_a, row_b in zip(a, b):
            deap_tools.cxBlend(row_a, row_b, 0.3)

    return mojo, upstream


@benchmark("SBX batch, 2m genes")
def sbx_batch():
    first = np.linspace(-2, 2, 2_000_000).reshape(2000, 1000)
    second = first[::-1].copy()

    def mojo():
        a, b = first.copy(), second.copy()
        tools.cxSimulatedBinaryBatch(
            a, b, 15.0, rng=np.random.default_rng(10)
        )

    def upstream():
        a, b = first.copy(), second.copy()
        random.seed(10)
        for row_a, row_b in zip(a, b):
            deap_tools.cxSimulatedBinary(row_a, row_b, 15.0)

    return mojo, upstream


@benchmark("Gaussian mutation batch, 2m genes")
def gaussian_batch():
    population = np.zeros((2000, 1000), dtype=np.float64)

    def mojo():
        values = population.copy()
        tools.mutGaussianBatch(
            values, 0.0, 1.0, 0.2, rng=np.random.default_rng(10)
        )

    def upstream():
        values = population.copy()
        random.seed(10)
        for individual in values:
            deap_tools.mutGaussian(individual, 0.0, 1.0, 0.2)

    return mojo, upstream


@benchmark("Polynomial mutation, 500k genes")
def polynomial():
    original = np.linspace(-0.9, 0.9, 500_000)

    def mojo():
        values = original.copy()
        random.seed(10)
        tools.mutPolynomialBounded(
            values,
            20.0,
            -1.0,
            1.0,
            0.2,
            device=os.environ.get("MOJODEAP_BENCH_DEVICE", "cpu"),
        )

    def upstream():
        values = original.copy()
        random.seed(10)
        deap_tools.mutPolynomialBounded(
            values, 20.0, -1.0, 1.0, 0.2
        )

    return mojo, upstream


def main():
    print(f"Machine: {cpu_name()} ({platform.system()} {platform.machine()})")
    print()
    print("| operator | mojo-deap | DEAP 1.4 | speedup |")
    print("| --- | ---: | ---: | ---: |")
    for name, builder in CASES:
        mojo, upstream = builder()
        if (
            name.startswith("Polynomial mutation")
            and os.environ.get("MOJODEAP_BENCH_DEVICE") == "gpu"
        ):
            name += ' [device="gpu"]'
        mojo()
        mojo_time = timeit(mojo)
        upstream_time = timeit(upstream)
        ratio = upstream_time / mojo_time
        label = f"{ratio:.2f}x" if ratio >= 1.0 else f"{1.0 / ratio:.2f}x slower"
        print(
            f"| {name} | {mojo_time * 1000:.1f} ms | "
            f"{upstream_time * 1000:.1f} ms | {label} |"
        )


if __name__ == "__main__":
    main()
