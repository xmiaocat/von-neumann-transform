# Benchmarks

`run_benchmarks.py` contains four studies covering the computational stages of
the forward and inverse transforms:

1. `operators`: overlap implementations with their natural preconditioners.
2. `preconditioners`: identity, banded-circulant, and dense-circulant
   preconditioners with Toeplitz-matmul and CG held fixed.
3. `projection`: direct, factorised, and FFT signal projection.
4. `reconstruction`: direct, factorised, and FFT signal reconstruction.

Transform construction and input generation happen before the measured region.
For solver studies, signal projection also happens before measurement. Setup and
solve time are reported separately and as a total. Their peak resident memory is
measured together because the operator and preconditioner remain resident while
the solver runs.

Each measurement uses a fresh subprocess. Timing and memory are collected in
separate workers so the memory-sampling thread cannot perturb recorded timings.

The version-controlled default experiment is defined in
`configs/default.toml`. Run it with:

```bash
uv run --group benchmark python benchmarks/run_benchmarks.py \
  --config benchmarks/configs/default.toml
```

The default configuration is used when `--config` is omitted. Command-line
values override the corresponding configuration values, which is convenient
for individual studies or smoke runs:

```text
TOML configuration < explicit command-line override
```

The configuration is required to contain every general run, transform, solver,
and memory setting. Each enabled study must also provide its case list. Missing
or unknown settings are reported before any benchmark workers are started.

```bash
uv run --group benchmark python benchmarks/run_benchmarks.py \
  --studies operators preconditioners

uv run --group benchmark python benchmarks/run_benchmarks.py \
  --studies projection reconstruction
```

For a quick smoke run:

```bash
uv run --group benchmark python benchmarks/run_benchmarks.py \
  --ks 8 --repeats 1
```

Results default to `benchmarks/results/benchmarks.csv`. Direct cases are skipped
above `k=32` by default because their storage grows as `N^2`, where `N = k^2`.
Pass `--direct-max-k 0` to remove that safeguard.

Alongside the CSV, the runner writes `benchmarks.resolved.json`. It contains the
fully resolved settings after command-line overrides, the source configuration
path, selected cases, and the concrete enum-based definition of every selected
case. Keep this file with the CSV when archiving or plotting a benchmark run.

## Study definitions

The operator study uses these pairings:

| Case | Overlap application | Preconditioner | Solver |
|---|---|---|---|
| `operator-direct` | dense matrix | none | direct |
| `operator-toeplitz-matmul` | dense block FFT (`matmul`) | dense circulant | CG |
| `operator-toeplitz-einsum` | dense block FFT (`einsum`) | dense circulant | CG |
| `operator-toeplitz-banded` | banded block FFT | banded circulant | CG |
| `operator-gaussian-stencil` | sparse local stencil | none | CG |

The preconditioner study holds `TOEPLITZ_MATMUL` and CG fixed, then compares
`NONE`, `CIRCULANT_BANDED`, and `CIRCULANT_DENSE`.

Solver rows include overlap-operator calls, preconditioner calls, the final
relative residual, and whether that residual meets the requested tolerance.
These should be considered alongside elapsed time: a more expensive
preconditioner can still reduce total cost by requiring fewer iterations.

Projection and reconstruction studies include all three `BasisMethod` values.
For `DIRECT`, evaluated basis construction is intentionally included in the
operation, matching the first real call on a new transform. Inputs are prepared
outside the measurement, and no basis cache is populated beforehand.

Memory is sampled at a configurable interval, so very brief temporary
allocations can fall between samples. Retained arrays and operators, normally
the dominant quantities here, are captured reliably.

## Reusing the measurement functions

`measurement.py` is independent of the transform being benchmarked. Its two
functions accept any zero-argument callable and return both the callable's
value and its measurement:

```python
from measurement import measure_peak_rss, measure_time

timing = measure_time(lambda: vnt.inverse_transform(q_nm))
memory = measure_peak_rss(lambda: vnt.inverse_transform(q_nm))
```

Use the same pattern for projection, basis construction, overlap setup, a
solver, or a callable that composes several phases. Prepare inputs before the
call when they should be excluded from the measurement. Run timing and memory
measurements separately because RSS sampling adds a small amount of overhead.

## Analysis and figures

After collecting measurements, run:

```bash
uv run --group benchmark python benchmarks/analyze_benchmarks.py
```

The analyzer reads `benchmarks/results/benchmarks.csv` and its neighboring
`benchmarks.resolved.json`. Use `--input` for a different CSV and `--output-dir`
for a different destination. By default, it writes to the ignored
`benchmarks/reports/` directory:

- `summary.csv`: median, quartiles, range, mean, and standard deviation for
  every method, size, and metric.
- `scaling.csv`: descriptive log-log slopes for total time and peak RSS
  increase when at least three sizes are available.
- `operators.svg`, `preconditioners.svg`, and `basis_methods.svg`.
- `validation.txt`: data completeness, convergence count, and the iteration
  counting convention.

The source benchmark CSV, summary, and figures all report memory in MiB.

The preconditioner figure includes CG iterations versus signal length. For the
current CG benchmark, `preconditioner_calls` equals the number of iterations:
SciPy's CG applies the preconditioner once per iteration with the default zero
initial guess. The stored `operator_calls` also equals that count in the
present data. These fields provide the total steps until convergence. Plotting
the residual after each step would require additional measurements.

Figures show medians with the 25th-to-75th percentile band. The analyzer checks
for missing or duplicate repetitions, invalid measurements, and non-converged
solves. Non-converged rows remain in the original CSV and are excluded from
summary statistics and figures.

All figure panels use logarithmic axes. The bottom horizontal axis shows
base-10 decades of signal length $N = k^2$. The top labels the same data
positions with the corresponding values of $k$. Vertical grid lines follow
the base-10 axis. The CG iteration panel also uses a logarithmic vertical axis.
The preconditioner legends name the actual methods as identity, circulant
banded, and circulant dense.
