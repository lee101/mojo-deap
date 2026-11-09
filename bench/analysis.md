# mojo-deap — ANALYZE phase: where the time goes

Measured (not estimated) hotspot map for the Mojo port of DEAP's
`tools.crossover` / `tools.mutation` operators. No algorithm was changed in
this phase; the next phases should act on the rankings below.

* Machine: Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz, Linux x86_64, 72 CPUs
  (`/proc/cpuinfo`, `os.cpu_count()` at run time).
* Toolchain: `mojo ==1.2.0.dev2026092605`, `max ==26.7.0.dev2026092605`
  (from `pixi.toml` / `MOJO_NOTES.md`).
* Python stack: NumPy 2.5.1, upstream DEAP 1.4.
* Date: measured 2026-11-09 (UTC) in this session.
* Flock-wrapped `pixi run bench` was run first and alone; no other benchmark
  ran alongside it. All timings below are best-of-k (k = 3 for
  `bench/bench.py`, k = 5–9 for the probe scripts) after warmup runs, on
  this machine.
* Input sizes use the sizes real callers pass: `n = 1000` genes for the
  single-individual API (classic DEAP individual length; `n = 200` spot
  check for the slow rows), `2_000_000` genes for the batch API
  (`2000 x 1000` population), `500_000` genes for PMX and polynomial
  mutation (the `bench/bench.py` sizes).
* FLOP counting convention: one scalar `+`/`*`/`-`/`/`/compare = 1 FLOP.
  `pow` (`beta ** power`, `xy ** (eta+1)`, `** -(eta+1)`) is counted as
  1 FLOP in the table so the columns stay simple, and the compute-bound
  verdict for SBX / polynomial is based on the *measured* kernel
  throughput (see kernel-only rows), not on that nominal count. A
  scalar `pow` costs tens of ns — two per-gene `pow` calls dominate those
  kernels regardless of the nominal F/B. Bytes touched = bytes read +
  bytes written by the kernel and its RNG / mask / scratch buffers
  (allocations counted, behaviour of this session). Bench-harness input
  copies (`first.copy()` etc.) are excluded from "bytes touched" and
  costed separately below.

## Ranked hotspots — batch and large sizes (flock-wrapped `pixi run bench`)

Times are the flock-wrapped `pixi run bench` run of 2026-11-09 (best of 3,
inputs copied and RNG generated inside the timer, exactly as in
`bench/bench.py`). Kernel-only rows are `ctypes` direct calls to
`dist/libmojo-deap.so` with pre-made buffers (no RNG, no copy).

| Rank | Function | Input size | Time (this port) | Time upstream DEAP 1.4 | vs upstream | Bytes touched | FLOPs | FLOPs / byte | Bound | Time sits in | Next phase |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| 1 | `cxSimulatedBinaryBatch` (`mdeap_cx_sbx`, SBX batch) | 2,000,000 genes | 119.3 ms | 2700.8 ms | 22.64x faster | 128,000,000 B (40N kernel + 8N RNG + copies excluded; see note) | 16,000,000 | 0.12 nominal | compute-bound in practice (kernel-only 88.59 ms for the same 2M genes = 0.90 GB/s vs 16.11 GB/s for blend — per-gene `pow` dominates) | mostly kernel; RNG buffer 9.22 ms of the total | simd (vectorised `pow`) + parallel |
| 2 | `mutGaussianBatch` (`mdeap_mut_gaussian`) | 2,000,000 genes | 54.8 ms | 542.2 ms | 9.90x faster | 66,000,000 B (17N kernel + 8N RNG mask/normal; copies excluded) | 2,000,000 (1 add / gene; the normal variate generation is the cost) | 0.03 | memory-bound | mostly Python/NumPy glue: `generator.normal(2M)` alone measured 34.75 ms + mask 10.67 ms; kernel-only 6.00 ms | parallel (none for SIMD/GPU — see verdict) |
| 3 | `cxBlendBatch` (`mdeap_cx_blend`, BLX batch) | 2,000,000 genes | 33.8 ms | 1832.6 ms | 54.25x faster | 128,000,000 B (40N kernel + 8N RNG; copies excluded) | 14,000,000 (7 / gene) | 0.11 | memory-bound | mostly glue at this size: kernel-only 4.97 ms (16.11 GB/s) vs 33.8 ms batch total; `generator.random(2M)` alone 9.22 ms + input copies 1.24 ms each | parallel |
| 4 | `mutPolynomialBounded` (`mdeap_mut_polynomial_scalar_bounds`) | 500,000 genes | 17.3 ms | 225.3 ms | 13.02x faster | 8,004,992 B (16N + MT19937 state 2496 B) | ~6,000,000 (2 `pow` per active gene × ~20% active; `pow` cost excluded from the count) | 0.75 incl. nominal `pow` | compute-bound in practice (kernel-only 13.70 ms; `pow` dominates) | mostly kernel (13.70 ms of 17.3 ms); glue is MT state extraction + `words` buffer 2496 B | simd (vectorised `pow`) + parallel |
| 5 | `cxPartialyMatched` (`mdeap_cx_pmx` + `mdeap_check_permutations`) | 500,000 genes | 11.6 ms | 492.1 ms | 42.45x faster | 26,000,000 B bench shape (a+b+positions+seen) | 2,000,000 (gather/scatter dominated, not FLOP) | 0.08 | memory-bound (random gather/scatter) | mostly kernel (2.74 ms PMX + 0.81 ms check kernel-only); copies 0.33 ms per 500k int64 array are part of bench time | none (sequential permutation dependency — no safe parallel; see verdict) |

Notes on bytes for ranks 1 and 3: the 128M B figure spreads the 40N
kernel footprint (`a` read + `b` read + `r` read + `a` write + `b` write)
plus the 8N RNG buffer read by the kernel and written by
`generator.random`; the two 16 MB input copies in `bench.py` are excluded
here and measured separately (1.24 ms per 2M `float64` copy at 26.8 GB/s).

## Ranked hotspots — single-individual API, n = 1000

Full-call time per call (RNG generation + wrapper + kernel), best of 9
after warmup, NumPy `float64`/`int64` fast path. Slower-than-upstream rows
are marked `SLOWER` and kept — they are the point of this phase.

| Rank | Function | Input size | Time (this port) | Time upstream | vs upstream | Bytes touched | FLOPs | FLOPs / byte | Bound | Time sits in | Next phase |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| S1 | `mutESLogNormal` | 1,000 | 0.4471 ms | 0.4637 ms | 1.04x faster | 33,000 B (values 8N + strategy 8N + active 1N + factors 8N + normals 8N) | 6,000 | 0.18 | memory-bound | Python glue: per-gene `random.gauss` loop at `tools.py:606` + `active`/`factors`/`normals` buffers | none |
| S2 | `cxESBlend` | 1,000 | 0.3859 ms | 2.1307 ms | 5.52x faster | 66,000 B (values + strategy, 2× blend buffers) | 7,000 | 0.11 | memory-bound | Python glue: per-gene random loop at `tools.py:337` + 2 kernel calls (values, strategy) | none |
| S3 | `mutGaussian` | 1,000 | 0.2949 ms | 0.2526 ms | **SLOWER** 0.86x | 33,000 B (means 8N + sigmas 8N + active 1N + delta 8N + values) | 1,000 | 0.03 | memory-bound | Python glue: per-gene `random.gauss` loop at `tools.py:386`; `np.full` for scalar `mu`/`sigma` = 16,000 B/call | none |
| S4 | `mutUniformInt` | 1,000 | 0.2764 ms | 0.2247 ms | **SLOWER** 0.81x | 17,000 B (values 8N + active 1N + replacements 8N) | 1,000 | 0.06 | memory-bound | Python glue: per-gene `random.randint` loop at `tools.py:557` | none |
| S5 | `mutShuffleIndexes` | 1,000 | 0.2408 ms | 0.3019 ms | 1.25x faster | 16,000 B (values 8N + targets 8N) | 1,000 | 0.06 | memory-bound | Python glue: per-gene `random.randint` loop at `tools.py:493` + `targets` 8,000 B | none |
| S6 | `mutPolynomialBounded` (scalar bounds) | 1,000 | 0.1973 ms | 0.4501 ms | 2.28x faster | 20,992 B (values 16N + MT state 2496 B + words 2496 B) | 12,000 | 0.57 | memory-bound nominal; `pow`-dominated on active genes | kernel + glue split: fused MT + mutation kernel; `weights` np.zeros check at `tools.py:411` only fires for `eta == -1`/vector bounds | none |
| S7 | `cxSimulatedBinary` | 1,000 | 0.1679 ms | 1.2739 ms | 7.59x faster | 40,000 B (a 8N + b 8N + rand 8N + writes 16N) | 8,000 | 0.20 | memory-bound nominal; `pow`-dominated kernel | glue: `np.fromiter(random.random() × 1000)` alone measured 0.100 ms (60% of 0.168 ms) + kernel | none |
| S8 | `cxUniformPartialyMatched` | 1,000 | 0.1610 ms | 0.4690 ms | 2.91x faster | 37,000 B (a+b 16N + positions 16N + seen 2N + mask 1N + writes) | 4,000 | 0.11 | memory-bound | glue: mask `np.fromiter` 0.100 ms of 0.161 ms; PMX kernel itself ~0.04 ms scale | none |
| S9 | `cxUniform` (float64 fast path) | 1,000 | 0.1446 ms | 0.2567 ms | 1.78x faster | 33,000 B (a 8N + b 8N + mask 1N + swaps 16N) | 1,000 | 0.03 | memory-bound | glue: mask `np.fromiter` 0.100 ms of 0.145 ms | none |
| S10 | `mutFlipBit` | 1,000 | 0.1374 ms | 0.1482 ms | 1.08x faster | 3,000 B (values 1N + active 1N + write 1N) | 1,000 | 0.33 | memory-bound | glue: mask `np.fromiter` 0.100 ms of 0.137 ms; kernel is a 1,000-element scan | none |
| S11 | `cxBlend` | 1,000 | 0.1237 ms | 0.8426 ms | 6.81x faster | 40,000 B (a 8N + b 8N + rand 8N + writes 16N) | 7,000 | 0.17 | memory-bound | glue: `np.fromiter` 0.100 ms (81% of 0.124 ms); kernel-only at n=1000 measured 7.30 µs | none |
| S12 | `cxSimulatedBinaryBounded` (scalar bounds) | 1,000 | 0.5834 ms | 2.2744 ms | 3.90x faster | 48,000 B (a 8N + b 8N + low/up 16N + rand 8N + active/swaps 2N + writes) | 20,000 | 0.42 | memory-bound nominal; 2× `pow` per active gene | glue: Python loop at `tools.py:296` filling `active`/`swaps`/`random_values` per gene (extra `random.random()` draws for `swap` gene) | none |
| S13 | `cxESTwoPoint` | 1,000 | 0.0923 ms | 0.0975 ms | 1.06x faster | 4,000 B (slice swap, ~N/3 mean segment) | 1,000 | 0.25 | memory-bound | 100% Python (slice swap both sides, no Mojo kernel) | none |
| S14 | `cxPartialyMatched` | 1,000 | 0.0411 ms | 0.8954 ms | 21.77x faster | 36,000 B (a+b 16N + positions 16N + seen 2N + segment writes) | 4,000 | 0.11 | memory-bound | kernel: dispatch ~5 µs + O(n) position build + O(segment) swaps; wrapper RNG is 2 `randint` calls only | none |
| S15 | `cxOrdered` | 1,000 | 0.0516 ms | 1.2399 ms | 24.03x faster | 18,000 B (a+b 16N + holes 2N) | 6,000 | 0.33 | memory-bound | kernel (3 O(n) passes + segment swap); wrapper RNG is `random.sample(range(n), 2)` | none |
| S16 | `cxTwoPoint` | 1,000 | 0.0142 ms | 0.0125 ms | **SLOWER** 0.88x | 4,000 B (slice swap) | 1,000 | 0.25 | memory-bound | 100% Python (both implementations are the same slice swap; 0.014 ms vs 0.013 ms is call overhead, not a kernel) | none |
| S17 | `cxOnePoint` | 1,000 | 0.0136 ms | 0.0134 ms | 1.00x (tie; 0.99x) | 4,000 B | 1,000 | 0.25 | memory-bound | 100% Python (identical slice swap) | none |
| S18 | `mutInversion` | 1,000 | 0.0135 ms | 0.0134 ms | 1.00x (tie; 0.99x) | 5,000 B (mean segment ~N/3, 2× copies) | 1,000 | 0.20 | memory-bound | 100% Python slice reversal both sides | none |

S12 is ranked by bytes/FLOPs, not by time; by time alone it (0.583 ms) is the
slowest single call. The full time ranking is S12 > S1 > S2 > S3 > S4 > S5 >
S6 > S7 > S8 > S9 > S10 > S11 > S13 > S15 > S14 > S16 > S17 ≈ S18.

## Small-size and list-input rows (overhead-bound, several SLOWER)

| Function | Input size | Time (this port) | Time upstream | vs upstream | Bytes touched | FLOPs | FLOPs / byte | Bound | Time sits in | Next phase |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| `cxBlend` | n = 10 | 0.023 ms | 0.021 ms | **SLOWER** 0.90x | 400 B | 70 | 0.17 | memory-bound | Python/ctypes glue: `np.fromiter` + wrapper + dispatch 4.99 µs at n=10; kernel µs | none |
| `mutFlipBit` | n = 10 | 0.018 ms | 0.010 ms | **SLOWER** 0.55x | 30 B | 10 | 0.33 | memory-bound | glue: mask generation dominates (kernel is a 10-element scan) | none |
| `mutFlipBit` | n = 100 | 0.043 ms | 0.032 ms | **SLOWER** 0.74x | 300 B | 100 | 0.33 | memory-bound | glue (`np.fromiter` ~0.01 ms at n=100 measured trend; dispatch 5.88 µs at n=100) | none |
| `cxBlend` | n = 100 | 0.040 ms | 0.196 ms | 4.86x faster | 4,000 B | 700 | 0.17 | memory-bound | glue (`np.fromiter` + dispatch) | none |
| `cxBlend` | n = 10,000 | 1.178 ms | 8.918 ms | 7.57x faster | 400,000 B | 70,000 | 0.17 | memory-bound | kernel + RNG; glue share small at 10k | none |
| `mutFlipBit` | n = 10,000 | 2.738 ms | 2.642 ms | **SLOWER** 0.97x (tie) | 30,000 B | 10,000 | 0.33 | memory-bound | glue: mask `np.fromiter(random.random×10000)` dominates; kernel scan is µs | none |
| `cxBlend` (python list inputs) | 1,000 | 1.381 ms | 0.263 ms | **SLOWER** 0.19x | 40,000 B + copy-in 16,000 B | 7,000 | 0.17 | memory-bound | Python glue: `_copy_back` elementwise loop measured 0.492 ms × 2 = 0.984 ms; `f64(list)` copy 0.040 ms × 2; kernel 7.30 µs | none |
| `mutGaussian` | n = 200 | 0.070 ms | 0.057 ms | **SLOWER** 0.82x | 6,600 B | 200 | 0.03 | memory-bound | Python glue (per-gene gauss loop); confirms SLOWER at 200 too | none |

`numpy` baselines measured in the same session (copy + RNG included):
`blend n=1000 0.0905 ms` (vs port wrapper 0.1237 ms, upstream 0.8426 ms);
`flip n=1000 0.0688 ms` (vs port 0.1374 ms, upstream 0.1482 ms);
`blend 2M 84.21 ms` (vs port kernel 4.97 ms, batch total 33.8 ms);
`SBX 2M 197.50 ms` (vs port kernel 88.59 ms, batch total 119.3 ms);
`Gauss 2M 59.27 ms` (vs port batch total 54.8 ms, kernel 6.00 ms).
The port beats vectorised NumPy on every large kernel; at n=1000 the
hand-rolled RNG fill (`np.fromiter` + `random.random()`) is slower than
NumPy's bulk PRNG (`np.random.random(1000)` measured 0.033 ms vs
`np.fromiter` 0.100 ms) — the single-API gap is RNG generation, not the
kernel.

## Kernel-only microbenchmarks (glue vs kernel split)

| Kernel | Input size | Kernel-only time | Bytes | FLOPs | FLOPs / byte | GB/s | Note |
|---|---:|---:|---:|---:|---:|---:|---|
| `mdeap_cx_blend` | 2,000,000 | 4.97 ms | 80,000,000 B | 14,000,000 | 0.17 | 16.11 | scalar loop; no SIMD in `src/operators.mojo:250` |
| `mdeap_cx_sbx` | 2,000,000 | 88.59 ms | 80,000,000 B | 16,000,000 nominal | 0.20 nominal | 0.90 | scalar loop with per-gene `pow` at `src/operators.mojo:275`; 18× slower than blend for the same traffic |
| `mdeap_mut_gaussian` | 2,000,000 | 6.00 ms | 34,000,000 B | 2,000,000 | 0.06 | 5.66 | masked add scan (`src/operators.mojo:345`) |
| `mdeap_cx_pmx` | 500,000 | 2.74 ms | ~28.7 MB incl. segment | — | — | 10.50 | O(n) + O(segment), gather/scatter |
| `mdeap_check_permutations` | 500,000 | 0.81 ms | ~18 MB | — | — | — | two O(n) validation passes |
| `mdeap_mut_polynomial_scalar_bounds` | 500,000 | 13.70 ms | 8,004,992 B | ~6,000,000 nominal | 0.75 nominal | 0.58 nominal | fused MT19937 + 2×`pow` per active gene; SIMD only in `_mt_refill` (`src/operators.mojo:62`), scalar in the mutation body |
| `mdeap_cx_blend` dispatch | n=0 / 10 / 100 / 1000 | 4.92 / 4.99 / 5.88 / 7.30 µs | — | — | — | — | ctypes + kernel dispatch; wrapper RNG dominates at n≤1000 |

RNG generation (measured): `np.fromiter((random.random()…)×1000) 0.100 ms`;
`1000× random.random()` loop `0.051 ms`; `np.random.random(1000) 0.033 ms`;
`np.random.random(2M) 9.22 ms`; `np.random.normal(2M) 34.75 ms`;
`mask (2M<0.2 → uint8) 10.67 ms`. Copies: `2M float64 1.24 ms (26.8 GB/s)`;
`500k int64 0.33 ms`.

## Flag: Python loops and per-call allocations / copies

**No wrapper re-enters `ctypes` per element.** Every `mojodeap.tools`
wrapper calls `lib()` once per operator and passes whole buffers
(`python/mojodeap/tools.py`; `cxESBlend` is the only wrapper calling the
kernel twice — once for values, once for strategy). `grep` for `lib()`
confirms one call site per wrapper; dispatch overhead is the 4.92–7.30 µs
figure above, not per-gene FFI.

The per-gene Python cost is *RNG generation + buffer stores*, not FFI:

| Python loop | Location | What it does per gene | Measured share |
|---|---|---|---|
| `np.fromiter((random.random() for _ in range(n)))` | `cxBlend` `tools.py:256`, `cxSimulatedBinary` `:276`, `cxUniform` `:143`, `cxUniformPartialyMatched` `:212`, `mutFlipBit` `:507` | 1 `random.random()` + store → fresh buffer | 0.100 ms per 1,000 genes = up to 81% of the wrapper total |
| `for index in range(n)` + `random.random()`/`gauss()`/`randint()` | `cxSimulatedBinaryBounded` `:296`, `cxESBlend` `:337`, `mutGaussian` `:386`, `mutShuffleIndexes` `:493`, `mutUniformInt` `:557`, `mutESLogNormal` `:606` | 1–3 Python-random draws + `active`/`delta`/`targets` stores per gene | entire wrapper time (S3/S4 slower-than-upstream rows) |
| `mutPolynomialBounded` pre-scan loop `:419` | runs only for `eta == -1` or vector-bounds aborts | `random.random()` per gene to find the abort | 0 in the scalar fast path measured |
| `_copy_back` element loop `tools.py:20` | **only for non-`float64`-1D / list inputs** (standard list path) | `target[i] = value.item()` per gene | 0.492 ms per 1,000 genes × 2 arrays = 0.984 ms — the list-input SLOWER row |

Per-call allocations / copies (per wrapper call, at n=1000; scale linearly):

| Wrapper | Buffers allocated per call | Bytes at n=1000 | Copies |
|---|---|---:|---|
| `cxOnePoint`/`cxTwoPoint`/`cxMessyOnePoint`/`mutInversion` | none | 0 | slice swap in place, no kernel |
| `cxBlend`/`cxSimulatedBinary` | `random_values` | 8,000 B | copy-in/out only for non-`float64` inputs (`_float_work`) |
| `cxUniform` (fast path) | `mask` | 1,000 B | none |
| `cxUniformPartialyMatched` | `mask` | 1,000 B | none |
| `cxPartialyMatched`/`cxOrdered` | `positions1/2` | 16,000 B | none |
| `cxPartialyMatched` (validation) | `seen1/2` | 2,000 B | — |
| `cxSimulatedBinaryBounded` | `active`, `swaps`, `random_values`, `low`, `up` | 2,000 + 8,000 + 16,000 B | `low`/`up` via `np.full` even for scalars |
| `mutGaussian` | `means`, `sigmas`, `active`, `delta` | 16,000 + 1,000 + 8,000 B | `np.full(size)` per scalar parameter |
| `mutPolynomialBounded` | `mt_state`, `words` (+ `zero_width` bool 1,000 B for `eta == −1`/vector bounds) | 2,496 + 2,496 B | none (scalar bounds) |
| `mutShuffleIndexes` | `targets` (`np.full`) | 8,000 B | none |
| `mutUniformInt` | `active`, `replacements` | 1,000 + 8,000 B | none |
| `mutESLogNormal` | `active`, `factors`, `normals` + saved RNG state | 1,000 + 16,000 B | none |
| Batch (`cxBlendBatch`, etc.) | `random_values` (`np.ascontiguousarray`) | 16,000,000 B at 2M | none (batch requires writable C-contiguous `float64`) |

Standard (non-batch) wrappers additionally copy in/out for any input that
is not already writable C-contiguous `float64`/`int64` 1-D NumPy:
`f64(list)` 0.040 ms per 1,000 genes measured, `_copy_back` 0.492 ms per
1,000 measured. On the NumPy fast path the input crosses without a data
copy (README "How it works") — the permutation wrappers still allocate
scratch (`positions`/`seen`/`holes`) on every call.

## Verdicts: next phase per hotspot (with the GPU numbers)

GPU rule applied (from `MOJO_NOTES.md`): a GPU path is only worth it above
roughly 2 flops per byte; the host API lives in `max.gpu.host` (not
`std.gpu`) but that doesn't help a memory-bound kernel. Every kernel here
is below that line, including the two `pow`-dominated ones:

* Highest nominal F/B in the repo: **polynomial 0.75** (incl. `pow` at
  1 FLOP). The *effective* FLOP count is higher, but the cost is scalar
  `pow` latency (≈0.58 GB/s nominal), not bandwidth — a GPU helps exactly
  when the bottleneck is bandwidth, which it is not. Verdict: **gpu: no**;
  use **simd** (vectorised `pow`, e.g. 4-lane like `_mt_refill` already
  does) + **parallel**.
* SBX batch: nominal F/B **0.20** (nominal), measured 0.90 GB/s. Same
  conclusion: **gpu: no** (F/B 0.20 ≪ 2); **simd + parallel**.
* Blend batch: F/B **0.11**, 16.11 GB/s single-thread — already
  bandwidth-efficient. **gpu: no** (0.11 ≪ 2); **parallel** (72 CPUs,
  `max.algorithm.parallelize` with an explicit worker count per
  `MOJO_NOTES.md` §5b) is the only scalable lever; SIMD would add ≤1.2×
  at 16 GB/s.
* Gaussian batch: F/B **0.03**; 34.75 ms of the 54.8 ms is NumPy
  `normal(2M)` generation. **gpu: no**; **parallel** kernels *and*
  vectorised RNG (NumPy already generates the variates in C; the wrapper
  is nearly free once the RNG is bulk — no Mojo change to the kernel
  helps the 34.75 ms).
* PMX / Ordered (F/B **0.08** / **0.33**): random gather/scatter over a
  permutation with a sequential swap chain (`cx_pmx` mutates `p1`/`p2`
  in place). **gpu: no** (0.08–0.33 ≪ 2); **parallel: no** (dependency
  chain — segment swaps are not independent). Verdict: **none**.
* All single-call `n=1000` wrappers (F/B 0.03–0.57): **gpu: no** (≪2);
  **parallel: no** (dispatch is 4.92–7.30 µs; thread launch would swamp
  a 0.012–0.583 ms call); **simd: no** (wrapper is 80% Python RNG fill;
  vectorising the kernel saves ≤7 µs). Verdict: **none** — the fix is
  bulk RNG (NumPy `random` + counter-based state, no Mojo change allowed
  in this phase by the task rules).

| Hotspot | F/B | Used next phase | Justification (number) |
|---|---:|---|---|
| SBX batch (rank 1) | 0.20 nominal, 0.90 GB/s measured | **simd + parallel**; gpu: no | `pow` latency, not bandwidth; 72 CPUs idle, kernel serial; F/B 0.20 ≪ 2 rules GPU out |
| Gaussian batch (rank 2) | 0.03 | **parallel**; gpu: no | 63% of time is `normal(2M)` RNG; kernel 6.00 ms could parallelise; F/B 0.03 ≪ 2 |
| Blend batch (rank 3) | 0.11 | **parallel**; gpu: no | 16.11 GB/s single core; parallel hits 72 cores, GPU would need >4 TB/s PCIe to win at F/B 0.11 |
| Polynomial (rank 4) | 0.75 nominal | **simd + parallel**; gpu: no | 13.70 ms kernel, 0.58 GB/s nominal; `pow` vectorisation 4× + parallel; F/B 0.75 still ≪ 2 |
| PMX (rank 5) | 0.08 | **none** | Sequential permutation chain; 2.74 ms + 0.81 ms already 42× upstream |
| Single `n=1000` wrappers (S1–S18) | 0.03–0.57 | **none** | Glue/RNG is 81–94% of wrapper time; kernel ≤7.30 µs; F/B ≪ 2 everywhere |
| Small `n=10/100` + list inputs | 0.17–0.33 | **none** | Overhead-bound: `np.fromiter` 0.010–0.100 ms + dispatch 4.92–5.88 µs; list `_copy_back` 0.492 ms/gene-set dominates |

## Slower-than-upstream: do not drop these rows

1. **`mutGaussian` n=1000 — 0.2949 ms vs 0.2526 ms (0.86x).** Stack-stable
   over 3 trials (0.80–0.86x) and at n=200 (0.82x). Cause: Python
   `random.gauss` loop `tools.py:386` + `means`/`sigmas` `np.full` (16,000 B).
   The kernel (`mdeap_mut_gaussian`) is 6.00 ms at 2M — the kernel is not
   the problem.
2. **`mutUniformInt` n=1000 — 0.2764 ms vs 0.2247 ms (0.81x).** Same pattern:
   Python `random.randint` loop `tools.py:557`.
3. **`cxTwoPoint` n=1000 — 0.0142 ms vs 0.0125 ms (0.88x)** and
   `cxOnePoint`/`mutInversion` ties (0.99x) — both sides are identical
   Python slice swaps; the gap is noise-level call overhead, not a kernel.
4. **Small n overhead rows**: `cxBlend n=10 0.90x`, `mutFlipBit n=10 0.55x`
   / `n=100 0.74x`, `mutFlipBit n=10000 0.97x` (mask fill scales with n
   while upstream does per-gene work in Python without the extra buffer).
5. **List-input standard API**: `cxBlend` on Python lists 1.381 ms vs
   upstream 0.263 ms — `_copy_back` 0.492 ms × 2 arrays is 71% of the
   port's time. Any GA caller passing lists (the common DEAP style)
   pays ~5× upstream on blend-class operators until the copy-back path
   is fixed.

All batch / 500k rows and the remaining single-call rows (cxBlend 6.81x,
cxSimulatedBinary 7.59x, cxPartialyMatched 21.77x, cxOrdered 24.03x,
cxESBlend 5.52x, polynomial 2.28x, etc.) are faster than upstream under
these measurements.

## Repro

```bash
pixi run bench          # flock-wrapped, canonical large-size table
pixi run python bench/bench.py   # same, without the flock wrapper
```

Probe scripts (`/tmp/measure2.py`, `/tmp/measure3.py`,
`/tmp/measure_kernel.py`, `/tmp/measure_batch.py`, `/tmp/measure_np.py`,
`/tmp/measure_final.py`, `/tmp/measure_ES.py`) were run with
`pixi run python` against the built `dist/libmojo-deap.so`; every number
in the tables above comes from one of those runs in this session.
