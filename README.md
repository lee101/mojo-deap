# mojo-deap

DEAP-compatible genetic algorithm variation operators whose per-gene work runs
in [Mojo](https://www.modular.com/mojo).

`mojodeap.tools` mirrors the names, signatures, in-place mutation, tuple return
values, and module-level Python `random` behavior of the covered
[`deap.tools`](https://deap.readthedocs.io/) API. Existing operator
registrations can therefore change their import without changing call sites:

```python
from mojodeap import tools
```

This project is intentionally focused on variation. It is not a reimplementation
of all of DEAP.

## Install and build

The repository pins its Mojo toolchain and Python dependencies through Pixi.

```bash
pixi install
pixi run build
```

Run the parity suite and benchmarks with:

```bash
pixi run test
pixi run bench
```

The build task creates `dist/libmojo-deap.so`. Run examples from the repository
through Pixi so that `python/` is on `PYTHONPATH`:

## Usage

The standard functions use Python's global `random` module just like DEAP, so
existing seeded runs remain reproducible:

```python
import random
import numpy as np
from mojodeap import tools

random.seed(42)
parent_a = np.linspace(-1.0, 1.0, 1000)
parent_b = np.linspace(1.0, -1.0, 1000)

child_a, child_b = tools.cxSimulatedBinary(parent_a, parent_b, eta=15.0)
(mutant,) = tools.mutPolynomialBounded(
    child_a, eta=20.0, low=-2.0, up=2.0, indpb=0.1
)
print(mutant.shape)
```

For population-sized contiguous arrays, batch forms amortize the FFI call and
use a NumPy generator:

```python
rng = np.random.default_rng(42)
left = rng.normal(size=(2000, 1000))
right = rng.normal(size=(2000, 1000))

tools.cxBlendBatch(left, right, alpha=0.25, rng=rng)
tools.mutGaussianBatch(left, mu=0.0, sigma=0.1, indpb=0.05, rng=rng)
```

Batch inputs must be writable, C-contiguous `float64` arrays.

`mutPolynomialBounded` also accepts `device="gpu"` for scalar bounds:

```python
tools.mutPolynomialBounded(
    child_a,
    eta=20.0,
    low=-2.0,
    up=2.0,
    indpb=0.1,
    device="gpu",
)
```

### CPU-only by necessity

**This port runs on the CPU, and every kernel is serial.** That is a property
of the pinned toolchain, not a choice:

- `parallelize` and the whole `std.parallelism` module no longer exist. There
  is no CPU-parallel primitive in the standard library to port the crossover
  and mutation loops onto, so those loops stay serial rather than growing a
  bespoke threading API.
- The GPU host API is gone as well: `DeviceContext` is not in `std.gpu.host`,
  `std.gpu`, or anywhere else, so `enqueue_create_buffer`, `enqueue_copy`,
  `enqueue_function`, and `synchronize` are unavailable.

Consequently `device="gpu"` is retained for API compatibility but runs the
same serial kernel as `device="cpu"`. It performs the same work and returns
the same values; it does not use a GPU. No replacement GPU API was invented to
paper over this. Vectorization survives: the polynomial kernels gather,
evaluate, and scatter four `float64` lanes at a time.

## Coverage

All public functions in DEAP's `tools.crossover` and `tools.mutation` modules
are exposed and covered by tests. Operators that benefit from contiguous
numeric work use Mojo kernels. Slice-based crossover and inversion stay in
Python because their work is dominated by Python-container operations.

| family | operators |
| --- | --- |
| Basic crossover | `cxOnePoint`, `cxTwoPoint`, `cxUniform` |
| Permutation crossover | `cxPartialyMatched`, `cxUniformPartialyMatched`, `cxOrdered` |
| Real-valued crossover | `cxBlend`, `cxSimulatedBinary`, `cxSimulatedBinaryBounded` |
| Messy and ES crossover | `cxMessyOnePoint`, `cxESBlend`, `cxESTwoPoint` |
| Mutation | `mutGaussian`, `mutPolynomialBounded`, `mutShuffleIndexes`, `mutFlipBit`, `mutUniformInt`, `mutInversion`, `mutESLogNormal` |
| Deprecated aliases | `cxTwoPoints`, `cxESTwoPoints` |
| Batch extensions | `cxBlendBatch`, `cxSimulatedBinaryBatch`, `mutGaussianBatch` |

The project does not cover DEAP selection algorithms, non-dominated sorting,
initializers, migration, decorators, statistics, evolutionary loops, creator,
or toolbox machinery. Use upstream DEAP for those pieces; `mojodeap` operator
functions can be registered in a DEAP toolbox.

Parity is tested against the installed upstream DEAP package. The tests compare
outputs and the next RNG value after each seeded call. Discrete operators match
exactly. Floating-point parity checks use explicit tolerances because Mojo and
CPython can round fractional powers differently. Tests also cover NumPy fast
paths, vector bounds, SIMD remainders, empty buffers, invalid permutation
values, dtype overflow, batch formulas, and the `device="gpu"` entry point.

## Benchmarks

Measured by the final `pixi run bench` publication-gate run on this machine:
Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz, Linux x86-64. Times are the best of
three warmed runs and include input copies and random-number generation.

| operator | mojo-deap | DEAP 1.4 | speedup |
| --- | ---: | ---: | ---: |
| PMX permutation crossover, 500k genes | 67.6 ms | 1158.3 ms | 17.13x |
| BLX-alpha batch, 2m genes | 56.5 ms | 4888.4 ms | 86.59x |
| SBX batch, 2m genes | 239.3 ms | 10110.3 ms | 42.25x |
| Gaussian mutation batch, 2m genes | 118.1 ms | 1302.4 ms | 11.02x |
| Polynomial mutation, 500k genes | 52.8 ms | 474.7 ms | 9.00x |

The batch comparisons apply upstream DEAP once per population row, which is
the normal DEAP usage. The batch API uses NumPy's generator while upstream uses
Python's `random`; the reported gain therefore includes both bulk random
generation and compiled per-gene arithmetic. Run the benchmark on the target
machine rather than assuming these ratios transfer unchanged.

## How it works

Python owns every input, output, random-value, mask, and scratch array. The
ctypes layer validates native dtypes, contiguity, lengths, writable status, and
non-null buffers before passing addresses across a C ABI into one Mojo
compilation unit. Each NumPy owner remains live for the full synchronous call.
CPU exports reconstruct mutable typed pointers from `Int` addresses and never
retain them. There is no device path, so there are no device buffers.

Continuous values are row-major `float64`, permutation and integer values are
`int64`, and Boolean masks are one-byte `uint8`. Standard list inputs are
copied into contiguous working arrays and copied back in place. Writable
contiguous NumPy arrays of the native type cross without a data copy.

DEAP draws randomness inside its Python loops. To reproduce that behavior,
standard wrappers draw decisions and variates from the same module-level
`random` source in the same order, then pass those buffers to Mojo. Kernels
perform the crossover bookkeeping and numerical transforms. The explicitly
named batch APIs instead accept a NumPy generator and flatten arbitrary
same-shaped C-contiguous arrays into a single kernel call.

Polynomial mutation reads and writes the module generator's CPython MT19937
state around a bulk Mojo fill. Its decisions, variates, final generator state,
and numerical tolerance remain identical to the per-gene API contract while
avoiding hundreds of thousands of Python calls.
