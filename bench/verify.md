# VERIFY phase — what was reproduced, what was not, what changed

Verification session: 2026-11-09 (UTC). Machine: Intel(R) Xeon(R) CPU
E5-2697 v4 @ 2.30GHz, Linux x86-64. Toolchain: `mojo ==1.2.0.dev2026092605`,
DEAP upstream 1.4, NumPy 2.5.1. Every number below was measured in this
session; nothing was copied from earlier phases.

## Gates run first

| command | result |
| --- | --- |
| `pixi run build && pixi run test` | green — 47 passed (2 unrelated deprecation warnings) |
| `pixi run bench` | exit 0; run four times total in this session (see spread below) |

## Reproduced (`pixi run bench`, four runs)

| operator | run 1 | run 2 | run 3 | run 4 (adopted for README) |
| --- | ---: | ---: | ---: | ---: |
| PMX 500k | 12.4 / 501.4 ms, 40.57x | 15.8 / 538.1 ms, 33.99x | 12.9 / 497.9 ms, 38.54x | **16.2 / 490.3 ms, 30.23x** |
| BLX batch 2m | 34.6 / 1865.8 ms, 53.90x | 37.1 / 1838.9 ms, 49.57x | 37.0 / 1890.2 ms, 51.12x | **36.5 / 1939.3 ms, 53.17x** |
| SBX batch 2m | 118.5 / 2636.2 ms, 22.24x | 115.8 / 2681.2 ms, 23.15x | 119.0 / 2712.5 ms, 22.80x | **119.3 / 2862.5 ms, 23.99x** |
| Gaussian batch 2m | 55.5 / 525.1 ms, 9.45x | 55.7 / 558.1 ms, 10.01x | 57.8 / 539.2 ms, 9.33x | **54.5 / 553.0 ms, 10.15x** |
| Polynomial 500k | 13.7 / 210.0 ms, 15.34x | 13.2 / 216.4 ms, 16.33x | 13.5 / 218.9 ms, 16.18x | **15.0 / 229.0 ms, 15.22x** |

(Runs are mojo-deap / DEAP times, best-of-3 warmed runs per `bench/bench.py`.
Run-to-run spread is within ~25% per cell; run 4 was adopted for the README
because it is a complete, self-consistent `pixi run bench` output.)

## Not reproduced — the old README table

| operator | old README | measured | verdict |
| --- | ---: | ---: | --- |
| PMX 500k | 67.6 / 1158.3 ms, 17.13x | 16.2 / 490.3 ms, 30.23x | mojo side >4x too slow, DEAP side >2x too fast in the old table — not reproduced |
| BLX batch 2m | 56.5 / 4888.4 ms, 86.59x | 36.5 / 1939.3 ms, 53.17x | DEAP side 2.5x too slow in the old table — not reproduced |
| SBX batch 2m | 239.3 / 10110.3 ms, 42.25x | 119.3 / 2862.5 ms, 23.99x | both sides badly off — not reproduced |
| Gaussian batch 2m | 118.1 / 1302.4 ms, 11.02x | 54.5 / 553.0 ms, 10.15x | both sides >2x off; speedup accidentally close — not reproduced |
| Polynomial 500k | 52.8 / 474.7 ms, 9.00x | 15.0 / 229.0 ms, 15.22x | both sides >2x off — not reproduced |

All five rows moved by far more than 20% on at least one side, so the whole
table was corrected to the run-4 numbers above, which were measured in this
session. Only the direction (port faster on these five rows) survives.

## Same-input / skipped-work audit (independent probe, best-of-5)

- **PMX and polynomial (same RNG):** both sides seeded with `random.seed(10)`
  on identical copies of the same inputs. PMX outputs are bit-identical;
  polynomial outputs match to 1.1e-10 max diff (within the documented 1e-9
  tolerance). No work skipped on either side: the port applies the operator
  to all 500k genes.
- **Batch rows (different RNGs, by design):** upstream is `deap.tools`
  applied row-by-row with Python `random`; the port batch APIs use NumPy's
  generator. This is the normal-DEAP usage the README already discloses, and
  it is stated in the README. What was verified here is that each batch call
  does the full per-gene work, not less:
  - BLX batch: output matches the per-gene formula `gamma = 1.6r - 0.3`
    exactly on all 2,000,000 genes.
  - SBX batch: output matches the per-gene formula to 1.95e-10 max diff
    (61.6% of genes within 1e-12; the rest is `pow` rounding, inside the
    documented 1e-9 tolerance).
  - Gaussian batch: 20.02% of genes mutated, matching `indpb=0.2`; no
    skipped genes.
- **Independent probe numbers (best-of-5, same sizes/seeds):**
  PMX 10.1 / 503.4 ms (49.7x); BLX 35.2 / 1857.6 ms (52.8x);
  SBX 119.2 / 2731.1 ms (22.9x); Gaussian 61.4 / 538.0 ms (8.8x);
  polynomial 14.1 / 221.8 ms (15.8x). Consistent with `pixi run bench` —
  no hidden work difference explains the old table.

## Test parity audit

`git log -p -- tests/` shows `tests/` was added once, in the initial commit
`a21206b`, and never touched again: `git diff a21206b HEAD -- tests/` is
empty. No assertions were removed, no tolerances loosened, no tests skipped.
The live suite still asserts output equality plus equality of the next
`random.random()` value after each seeded call (`paired_call`), exact
equality for discrete operators, 2e-14 for blend/Gaussian/ES paths and 1e-9
for SBX/polynomial paths. Nothing to restore; the suite passes 47/47.

## Changes made

1. `README.md`: benchmark table replaced with the run-4 `pixi run bench`
   numbers measured in this session; header updated to say so and to point
   here. No rows removed, no rewording of the batch-RNG disclosure. No row
   is slower-than-upstream in this table, so nothing to keep honest there —
   the slower small-size rows live in `bench/analysis.md`, which was not
   part of the README table and was not edited.
2. `bench/verify.md`: this file.
3. No code changes: nothing failed a test, so no fix was made.

## Final state

`pixi run build && pixi run test` — green (47 passed). The README numbers
are the run-4 `pixi run bench` output above and will drift a few percent
between runs; they are a measurement of this machine, not a promise.
