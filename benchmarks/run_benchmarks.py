"""Benchmark the computational stages of the von Neumann transform.

Run through ``uv`` so the optional benchmark dependency is available::

    uv run --group benchmark python benchmarks/run_benchmarks.py

Every timing and memory measurement uses a fresh subprocess. Inputs and the
``VonNeumannTransform`` instance are prepared before the measured operation.
"""

import argparse
import csv
import gc
import json
import os
import subprocess
import sys
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

for variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(variable, "1")

import numpy as np
from measurement import measure_peak_rss, measure_time
from scipy.sparse.linalg import LinearOperator

from von_neumann_transform import (
    BasisMethod,
    MatVecMethod,
    PrecondMethod,
    SolverMethod,
    VonNeumannTransform,
)


@dataclass(frozen=True)
class Case:
    study: str
    basis: BasisMethod | None = None
    matvec: MatVecMethod | None = None
    preconditioner: PrecondMethod | None = None
    solver: SolverMethod | None = None


CASES = {
    "operator-direct": Case(
        "operators",
        matvec=MatVecMethod.DIRECT,
        preconditioner=PrecondMethod.NONE,
        solver=SolverMethod.DIRECT,
    ),
    "operator-toeplitz-matmul": Case(
        "operators",
        matvec=MatVecMethod.TOEPLITZ_MATMUL,
        preconditioner=PrecondMethod.CIRCULANT_DENSE,
        solver=SolverMethod.CG,
    ),
    "operator-toeplitz-einsum": Case(
        "operators",
        matvec=MatVecMethod.TOEPLITZ_EINSUM,
        preconditioner=PrecondMethod.CIRCULANT_DENSE,
        solver=SolverMethod.CG,
    ),
    "operator-toeplitz-banded": Case(
        "operators",
        matvec=MatVecMethod.TOEPLITZ_BANDED,
        preconditioner=PrecondMethod.CIRCULANT_BANDED,
        solver=SolverMethod.CG,
    ),
    "operator-gaussian-stencil": Case(
        "operators",
        matvec=MatVecMethod.GAUSSIAN_STENCIL,
        preconditioner=PrecondMethod.NONE,
        solver=SolverMethod.CG,
    ),
    "preconditioner-none": Case(
        "preconditioners",
        matvec=MatVecMethod.TOEPLITZ_MATMUL,
        preconditioner=PrecondMethod.NONE,
        solver=SolverMethod.CG,
    ),
    "preconditioner-banded": Case(
        "preconditioners",
        matvec=MatVecMethod.TOEPLITZ_MATMUL,
        preconditioner=PrecondMethod.CIRCULANT_BANDED,
        solver=SolverMethod.CG,
    ),
    "preconditioner-dense": Case(
        "preconditioners",
        matvec=MatVecMethod.TOEPLITZ_MATMUL,
        preconditioner=PrecondMethod.CIRCULANT_DENSE,
        solver=SolverMethod.CG,
    ),
    "projection-direct": Case("projection", basis=BasisMethod.DIRECT),
    "projection-factorise": Case("projection", basis=BasisMethod.FACTORISE),
    "projection-fft": Case("projection", basis=BasisMethod.FFT),
    "reconstruction-direct": Case("reconstruction", basis=BasisMethod.DIRECT),
    "reconstruction-factorise": Case(
        "reconstruction", basis=BasisMethod.FACTORISE
    ),
    "reconstruction-fft": Case("reconstruction", basis=BasisMethod.FFT),
}

STUDIES = ("operators", "preconditioners", "projection", "reconstruction")
DEFAULT_CONFIG_PATH = Path(__file__).with_name("configs") / "default.toml"
RESULT_PREFIX = "VNT_BENCHMARK_RESULT="
MIB = 1024**2
CSV_FIELDS = (
    "repeat",
    "study",
    "case",
    "k",
    "npoints",
    "setup_seconds",
    "solve_seconds",
    "operation_seconds",
    "total_seconds",
    "baseline_rss_mib",
    "peak_rss_mib",
    "peak_increment_mib",
    "operator_calls",
    "preconditioner_calls",
    "relative_residual",
    "converged",
)


def _reject_unknown(values: dict, allowed: set[str], location: str) -> None:
    unknown = set(values) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown keys in {location}: {names}")


def _require_keys(values: dict, required: set[str], location: str) -> None:
    missing = required - set(values)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Missing required keys in {location}: {names}")


def _section(config: dict, name: str) -> dict:
    value = config.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"Configuration section [{name}] must be a table.")
    return value


def _load_config(path: Path) -> dict:
    with path.open("rb") as handle:
        config = tomllib.load(handle)

    _reject_unknown(
        config,
        {"run", "transform", "solver", "memory", "studies"},
        "configuration root",
    )
    _require_keys(
        config,
        {"run", "transform", "solver", "memory", "studies"},
        "configuration root",
    )

    run_keys = {"ks", "repeats", "warmups", "seed", "output", "direct_max_k"}
    run = _section(config, "run")
    _reject_unknown(run, run_keys, "[run]")
    _require_keys(run, run_keys, "[run]")

    transform_keys = {"omega_min", "omega_max"}
    transform = _section(config, "transform")
    _reject_unknown(transform, transform_keys, "[transform]")
    _require_keys(transform, transform_keys, "[transform]")

    solver_keys = {"rtol", "atol", "maxiter"}
    solver = _section(config, "solver")
    _reject_unknown(solver, solver_keys, "[solver]")
    _require_keys(solver, solver_keys, "[solver]")

    memory_keys = {"sample_interval"}
    memory = _section(config, "memory")
    _reject_unknown(memory, memory_keys, "[memory]")
    _require_keys(memory, memory_keys, "[memory]")

    studies = _section(config, "studies")
    _reject_unknown(studies, {"enabled", *STUDIES}, "[studies]")
    _require_keys(studies, {"enabled"}, "[studies]")
    enabled = studies["enabled"]
    if not isinstance(enabled, list) or not all(
        isinstance(study, str) for study in enabled
    ):
        raise ValueError("studies.enabled must be a list of study names.")
    unknown_studies = set(enabled) - set(STUDIES)
    if unknown_studies:
        names = ", ".join(sorted(unknown_studies))
        raise ValueError(f"Unknown benchmark studies: {names}")
    if len(set(enabled)) != len(enabled):
        raise ValueError("A benchmark study was enabled more than once.")

    for study in enabled:
        _require_keys(studies, {study}, "[studies]")
        study_config = studies[study]
        if not isinstance(study_config, dict):
            raise ValueError(f"[studies.{study}] must be a table.")
        _reject_unknown(study_config, {"cases"}, f"[studies.{study}]")
        _require_keys(study_config, {"cases"}, f"[studies.{study}]")
    _configured_cases(studies, enabled)
    return config


def _configured_cases(studies_config: dict, studies: list[str]) -> list[str]:
    selected: list[str] = []
    for study in studies:
        study_config = studies_config.get(study)
        if not isinstance(study_config, dict) or "cases" not in study_config:
            raise ValueError(
                f"Selected study {study} requires [studies.{study}].cases."
            )
        names = study_config["cases"]
        if not isinstance(names, list) or not all(
            isinstance(name, str) for name in names
        ):
            raise ValueError(f"studies.{study}.cases must be a list of names.")
        for name in names:
            if name not in CASES:
                raise ValueError(f"Unknown benchmark case: {name}")
            if CASES[name].study != study:
                raise ValueError(
                    f"Case {name} does not belong to study {study}."
                )
            if name in selected:
                raise ValueError(
                    f"Benchmark case selected more than once: {name}"
                )
            selected.append(name)
    return selected


def _resolve_args(cli: argparse.Namespace) -> argparse.Namespace:
    config = _load_config(cli.config)
    run = _section(config, "run")
    transform = _section(config, "transform")
    solver = _section(config, "solver")
    memory = _section(config, "memory")
    studies_config = _section(config, "studies")

    args = argparse.Namespace(**vars(cli))
    args.ks = cli.ks if cli.ks is not None else run["ks"]
    args.repeats = cli.repeats if cli.repeats is not None else run["repeats"]
    args.warmups = cli.warmups if cli.warmups is not None else run["warmups"]
    args.seed = cli.seed if cli.seed is not None else run["seed"]
    output = cli.output if cli.output is not None else run["output"]
    args.output = Path(output)
    args.direct_max_k = (
        cli.direct_max_k
        if cli.direct_max_k is not None
        else run["direct_max_k"]
    )
    args.omega_min = (
        cli.omega_min if cli.omega_min is not None else transform["omega_min"]
    )
    args.omega_max = (
        cli.omega_max if cli.omega_max is not None else transform["omega_max"]
    )
    args.rtol = cli.rtol if cli.rtol is not None else solver["rtol"]
    args.atol = cli.atol if cli.atol is not None else solver["atol"]
    args.maxiter = (
        cli.maxiter if cli.maxiter is not None else solver["maxiter"]
    )
    args.memory_interval = (
        cli.memory_interval
        if cli.memory_interval is not None
        else memory["sample_interval"]
    )

    configured_studies = studies_config["enabled"]
    args.studies = cli.studies or configured_studies
    if not isinstance(args.studies, list) or not all(
        isinstance(study, str) for study in args.studies
    ):
        raise ValueError("studies.enabled must be a list of study names.")
    unknown_studies = set(args.studies) - set(STUDIES)
    if unknown_studies:
        names = ", ".join(sorted(unknown_studies))
        raise ValueError(f"Unknown benchmark studies: {names}")
    if len(set(args.studies)) != len(args.studies):
        raise ValueError("A benchmark study was selected more than once.")

    if cli.cases is not None:
        args.cases = cli.cases
        if cli.studies is None:
            args.studies = list(
                dict.fromkeys(CASES[name].study for name in args.cases)
            )
        for name in args.cases:
            if CASES[name].study not in args.studies:
                raise ValueError(
                    f"Case {name} is outside the selected studies."
                )
    else:
        args.cases = _configured_cases(studies_config, args.studies)

    if not isinstance(args.ks, list) or not args.ks:
        raise ValueError("run.ks must be a non-empty list.")
    if not all(
        isinstance(k, int) and not isinstance(k, bool) and k >= 2
        for k in args.ks
    ):
        raise ValueError("Every k value must be an integer of at least 2.")
    integer_fields = {
        "repeats": args.repeats,
        "warmups": args.warmups,
        "seed": args.seed,
        "direct_max_k": args.direct_max_k,
        "maxiter": args.maxiter,
    }
    for name, value in integer_fields.items():
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{name} must be an integer.")
    if args.repeats < 1:
        raise ValueError("repeats must be at least 1.")
    if args.warmups < 0:
        raise ValueError("warmups cannot be negative.")
    if args.direct_max_k < 0:
        raise ValueError("direct_max_k cannot be negative.")
    if args.maxiter < 1:
        raise ValueError("maxiter must be at least 1.")
    numeric_fields = {
        "omega_min": args.omega_min,
        "omega_max": args.omega_max,
        "rtol": args.rtol,
        "atol": args.atol,
        "memory.sample_interval": args.memory_interval,
    }
    for name, value in numeric_fields.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{name} must be numeric.")
    if args.omega_min < 0.0 or args.omega_max <= args.omega_min:
        raise ValueError("omega values require 0 <= omega_min < omega_max.")
    if args.rtol < 0.0 or args.atol < 0.0:
        raise ValueError("Solver tolerances cannot be negative.")
    if args.memory_interval <= 0.0:
        raise ValueError("memory.sample_interval must be positive.")
    return args


def _enum_name(value) -> str | None:
    return value.name if value is not None else None


def _resolved_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        "source_config": str(args.config.resolve()),
        "run": {
            "ks": args.ks,
            "repeats": args.repeats,
            "warmups": args.warmups,
            "seed": args.seed,
            "output": str(args.output),
            "direct_max_k": args.direct_max_k,
        },
        "transform": {
            "omega_min": args.omega_min,
            "omega_max": args.omega_max,
        },
        "solver": {
            "rtol": args.rtol,
            "atol": args.atol,
            "maxiter": args.maxiter,
        },
        "memory": {"sample_interval": args.memory_interval},
        "studies": {"enabled": args.studies, "cases": args.cases},
        "case_definitions": {
            name: {
                "study": CASES[name].study,
                "basis": _enum_name(CASES[name].basis),
                "matvec": _enum_name(CASES[name].matvec),
                "preconditioner": _enum_name(CASES[name].preconditioner),
                "solver": _enum_name(CASES[name].solver),
            }
            for name in args.cases
        },
    }


def _count_calls(
    operator: LinearOperator, increment: Callable[[], None]
) -> LinearOperator:
    def matvec(vector: np.ndarray) -> np.ndarray:
        increment()
        return operator.matvec(vector)

    def rmatvec(vector: np.ndarray) -> np.ndarray:
        return operator.rmatvec(vector)

    return LinearOperator(
        operator.shape,
        dtype=operator.dtype,
        matvec=matvec,
        rmatvec=rmatvec,
    )


def _base_result(args: argparse.Namespace, case: Case) -> dict[str, object]:
    return {
        "study": case.study,
        "case": args.case,
        "k": args.k,
        "npoints": args.k**2,
        "setup_seconds": None,
        "solve_seconds": None,
        "operation_seconds": None,
        "total_seconds": None,
        "operator_calls": None,
        "preconditioner_calls": None,
        "relative_residual": None,
        "converged": None,
    }


def _measure_operation(
    args: argparse.Namespace,
    operation: Callable[[], np.ndarray],
) -> tuple[np.ndarray, dict[str, object]]:
    if args.measurement == "time":
        timing = measure_time(operation)
        return timing.value, {
            "operation_seconds": timing.seconds,
            "total_seconds": timing.seconds,
        }

    memory = measure_peak_rss(
        operation,
        sample_interval=args.memory_interval,
    )
    return memory.value, {
        "baseline_rss_mib": memory.baseline_rss_bytes / MIB,
        "peak_rss_mib": memory.peak_rss_bytes / MIB,
        "peak_increment_mib": memory.peak_increment_bytes / MIB,
    }


def _basis_worker(
    args: argparse.Namespace,
    case: Case,
    vnt: VonNeumannTransform,
    rng: np.random.Generator,
) -> dict[str, object]:
    basis = case.basis
    assert basis is not None
    if case.study == "projection":
        signal = rng.standard_normal(vnt.npoints) + 1j * rng.standard_normal(
            vnt.npoints
        )

        def operation() -> np.ndarray:
            return vnt.get_signal_projection(
                vnt.w_grid,
                vnt.w_n_arr,
                vnt.t_n_arr,
                vnt.alpha,
                signal,
                basis,
            )

    else:
        coefficients = rng.standard_normal((vnt.k, vnt.k)) + 1j * (
            rng.standard_normal((vnt.k, vnt.k))
        )

        def operation() -> np.ndarray:
            return vnt.inverse_transform(coefficients, method=basis)

    gc.collect()
    value, measurements = _measure_operation(args, operation)
    if not np.all(np.isfinite(value)):
        raise RuntimeError("Measured operation returned non-finite values.")
    return {**_base_result(args, case), **measurements}


def _solver_worker(
    args: argparse.Namespace,
    case: Case,
    vnt: VonNeumannTransform,
    rng: np.random.Generator,
) -> dict[str, object]:
    assert case.matvec is not None
    assert case.preconditioner is not None
    assert case.solver is not None

    signal = rng.standard_normal(vnt.npoints) + 1j * rng.standard_normal(
        vnt.npoints
    )
    alpha_nm = vnt.get_signal_projection(
        vnt.w_grid,
        vnt.w_n_arr,
        vnt.t_n_arr,
        vnt.alpha,
        signal,
        BasisMethod.FFT,
    )
    del signal
    gc.collect()

    operator_calls = 0
    preconditioner_calls = 0

    def increment_operator() -> None:
        nonlocal operator_calls
        operator_calls += 1

    def increment_preconditioner() -> None:
        nonlocal preconditioner_calls
        preconditioner_calls += 1

    def setup():
        return vnt.get_ovlp(
            vnt.alpha,
            vnt.w_n_arr,
            vnt.t_n_arr,
            case.matvec,
            case.preconditioner,
        )

    def solve(overlap):
        measured_overlap = overlap
        if isinstance(overlap, tuple):
            measured_overlap = (
                _count_calls(overlap[0], increment_operator),
                _count_calls(overlap[1], increment_preconditioner),
            )
        return vnt.solve_ovlp(
            measured_overlap,
            alpha_nm,
            case.solver,
            rtol=args.rtol,
            atol=args.atol,
            maxiter=args.maxiter,
        )

    if args.measurement == "time":
        setup_measurement = measure_time(setup)
        overlap = setup_measurement.value
        solve_measurement = measure_time(lambda: solve(overlap))
        coefficients = solve_measurement.value
        measurements = {
            "setup_seconds": setup_measurement.seconds,
            "solve_seconds": solve_measurement.seconds,
            "total_seconds": (
                setup_measurement.seconds + solve_measurement.seconds
            ),
        }
    else:
        overlap = None

        def setup_and_solve() -> np.ndarray:
            nonlocal overlap
            overlap = setup()
            return solve(overlap)

        memory = measure_peak_rss(
            setup_and_solve,
            sample_interval=args.memory_interval,
        )
        coefficients = memory.value
        measurements = {
            "baseline_rss_mib": memory.baseline_rss_bytes / MIB,
            "peak_rss_mib": memory.peak_rss_bytes / MIB,
            "peak_increment_mib": memory.peak_increment_bytes / MIB,
        }

    assert overlap is not None
    flat_coefficients = coefficients.ravel()
    if isinstance(overlap, tuple):
        residual = overlap[0] @ flat_coefficients - alpha_nm.ravel()
    else:
        residual = overlap @ flat_coefficients - alpha_nm.ravel()
    rhs_norm = np.linalg.norm(alpha_nm)
    residual_norm = np.linalg.norm(residual)
    relative_residual = float(residual_norm / rhs_norm)
    threshold = max(args.rtol, args.atol / rhs_norm)

    return {
        **_base_result(args, case),
        **measurements,
        "operator_calls": operator_calls,
        "preconditioner_calls": preconditioner_calls,
        "relative_residual": relative_residual,
        "converged": bool(relative_residual <= threshold),
    }


def _worker(args: argparse.Namespace) -> dict[str, object]:
    case = CASES[args.case]
    vnt = VonNeumannTransform(
        args.k**2,
        args.omega_min,
        args.omega_max,
    )
    rng = np.random.default_rng(args.seed)
    if case.study in ("projection", "reconstruction"):
        return _basis_worker(args, case, vnt, rng)
    return _solver_worker(args, case, vnt, rng)


def _run_subprocess(
    args: argparse.Namespace, case: str, k: int, measurement: str
) -> dict[str, object]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--case",
        case,
        "--k",
        str(k),
        "--omega-min",
        str(args.omega_min),
        "--omega-max",
        str(args.omega_max),
        "--rtol",
        str(args.rtol),
        "--atol",
        str(args.atol),
        "--maxiter",
        str(args.maxiter),
        "--seed",
        str(args.seed),
        "--memory-interval",
        str(args.memory_interval),
        "--measurement",
        measurement,
    ]
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=True,
    )
    lines = completed.stdout.splitlines()
    result_lines = [line for line in lines if line.startswith(RESULT_PREFIX)]
    for line in lines:
        if not line.startswith(RESULT_PREFIX):
            print(line, file=sys.stderr)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    if len(result_lines) != 1:
        raise RuntimeError("Worker did not emit exactly one benchmark result.")
    return json.loads(result_lines[0][len(RESULT_PREFIX) :])


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_resolved_config(path: Path, args: argparse.Namespace) -> Path:
    metadata_path = path.with_name(f"{path.stem}.resolved.json")
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(_resolved_config(args), handle, indent=2)
        handle.write("\n")
    return metadata_path


def _selected_cases(args: argparse.Namespace) -> list[str]:
    selected = args.cases or [
        name for name, case in CASES.items() if case.study in args.studies
    ]
    return [name for name in selected if CASES[name].study in args.studies]


def _main(args: argparse.Namespace) -> None:
    rows: list[dict[str, object]] = []
    selected = _selected_cases(args)
    if not selected:
        raise RuntimeError("No cases match the requested studies.")

    for case_name in selected:
        for k in args.ks:
            if (
                "direct" in case_name
                and args.direct_max_k
                and k > args.direct_max_k
            ):
                print(
                    f"Skipping {case_name} at k={k} "
                    f"(direct limit: {args.direct_max_k})."
                )
                continue
            for _ in range(args.warmups):
                _run_subprocess(args, case_name, k, "time")
            for repeat in range(1, args.repeats + 1):
                result = _run_subprocess(args, case_name, k, "time")
                memory = _run_subprocess(args, case_name, k, "memory")
                for key in (
                    "baseline_rss_mib",
                    "peak_rss_mib",
                    "peak_increment_mib",
                ):
                    result[key] = memory[key]
                row: dict[str, object] = {field: None for field in CSV_FIELDS}
                row.update(result)
                row["repeat"] = repeat
                rows.append(row)

                seconds = result.get("operation_seconds")
                if seconds is None:
                    seconds = result["total_seconds"]
                print(
                    f"{case_name:31s} k={k:4d} repeat={repeat:2d} "
                    f"time={seconds:.6f}s "
                    f"peak_increment={result['peak_increment_mib']:.2f}MiB"
                )

    if not rows:
        raise RuntimeError("All requested cases were skipped.")
    _write_csv(args.output, rows)
    metadata_path = _write_resolved_config(args.output, args)
    print(f"Wrote {len(rows)} measurements to {args.output}")
    print(f"Wrote resolved configuration to {metadata_path}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="TOML experiment configuration",
    )
    parser.add_argument(
        "--worker", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument("--case", choices=CASES, help=argparse.SUPPRESS)
    parser.add_argument("--k", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--studies",
        nargs="+",
        choices=STUDIES,
        default=None,
        help="override the studies enabled by the configuration",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=CASES,
        help="optional cases selected from the requested studies",
    )
    parser.add_argument("--ks", nargs="+", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--warmups", type=int, default=None)
    parser.add_argument(
        "--direct-max-k",
        type=int,
        default=None,
        help="override the direct-case size guard. 0 disables it",
    )
    parser.add_argument("--omega-min", type=float, default=None)
    parser.add_argument("--omega-max", type=float, default=None)
    parser.add_argument("--rtol", type=float, default=None)
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument("--maxiter", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--memory-interval", type=float, default=None)
    parser.add_argument(
        "--measurement",
        choices=("time", "memory"),
        default="time",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="override the configured CSV output path",
    )
    return parser


if __name__ == "__main__":
    parsed = _parser().parse_args()
    if parsed.worker:
        if parsed.case is None or parsed.k is None:
            raise SystemExit("Workers require --case and --k.")
        print(RESULT_PREFIX + json.dumps(_worker(parsed), sort_keys=True))
    else:
        try:
            resolved = _resolve_args(parsed)
        except (OSError, tomllib.TOMLDecodeError, ValueError) as error:
            _parser().error(str(error))
        _main(resolved)
