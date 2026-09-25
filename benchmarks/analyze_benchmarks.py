"""Summarize and plot one benchmark CSV and its resolved configuration.

uv run --group benchmark python benchmarks/analyze_benchmarks.py
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, NullLocator

STUDIES = ("operators", "preconditioners", "projection", "reconstruction")
METRICS = (
    "total_seconds",
    "setup_seconds",
    "solve_seconds",
    "peak_increment_mib",
    "cg_iterations",
    "relative_residual",
)
REQUIRED_COLUMNS = {
    "repeat",
    "study",
    "case",
    "k",
    "npoints",
    "total_seconds",
    "setup_seconds",
    "solve_seconds",
    "peak_increment_mib",
    "preconditioner_calls",
    "relative_residual",
    "converged",
}
PALETTE = plt.get_cmap("tab10")


def _read_data(path: Path) -> tuple[list[dict], dict]:
    metadata_path = path.with_name(f"{path.stem}.resolved.json")
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"Missing CSV columns: {', '.join(sorted(missing))}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError("Benchmark CSV has no rows.")
    return rows, metadata


def _number(row: dict, name: str, *, required: bool = True) -> float | None:
    value = row.get(name, "")
    if value in ("", None):
        if required:
            raise ValueError(f"Missing {name} in {row['case']} k={row['k']}.")
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid {name}: {value!r}") from error
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and non-negative: {value!r}")
    return number


def _validate(rows: list[dict], metadata: dict) -> list[str]:
    run = metadata["run"]
    definitions = metadata["case_definitions"]
    selected = metadata["studies"]["cases"]
    ks = run["ks"]
    repeats = run["repeats"]
    direct_limit = run["direct_max_k"]
    expected = {
        (case, k, repeat)
        for case in selected
        for k in ks
        if not ("direct" in case and direct_limit and k > direct_limit)
        for repeat in range(1, repeats + 1)
    }
    seen = set()
    failed = []
    tolerance = metadata["solver"]["rtol"]
    check_relative_tolerance = metadata["solver"]["atol"] == 0
    for row in rows:
        case = row["case"]
        if case not in definitions:
            raise ValueError(f"CSV contains unconfigured case {case!r}.")
        if row["study"] != definitions[case]["study"]:
            raise ValueError(f"Study mismatch for {case}.")
        try:
            k = int(row["k"])
            npoints = int(row["npoints"])
            repeat = int(row["repeat"])
        except ValueError as error:
            raise ValueError(
                f"Invalid grid size or repeat in {case}."
            ) from error
        if npoints != k * k:
            raise ValueError(f"npoints != k² in {case} k={k}.")
        key = (case, k, repeat)
        if key in seen:
            raise ValueError(f"Duplicate measurement: {key}.")
        seen.add(key)
        if key not in expected:
            raise ValueError(f"Unexpected measurement: {key}.")

        for name in ("total_seconds", "peak_increment_mib"):
            _number(row, name)
        if row["study"] in ("operators", "preconditioners"):
            for name in (
                "setup_seconds",
                "solve_seconds",
                "relative_residual",
            ):
                _number(row, name)
            if definitions[case]["solver"] == "CG":
                iterations = _number(row, "preconditioner_calls")
                assert iterations is not None
                if not iterations.is_integer():
                    raise ValueError(
                        f"Non-integer CG iteration count in {key}."
                    )
                row["cg_iterations"] = str(int(iterations))
            if row["converged"] not in ("True", "False"):
                raise ValueError(f"Invalid converged value in {key}.")
            if check_relative_tolerance:
                meets_tolerance = float(row["relative_residual"]) <= tolerance
                if (row["converged"] == "True") != meets_tolerance:
                    raise ValueError(
                        f"Convergence flag disagrees with residual in {key}."
                    )
            if row["converged"] == "False":
                failed.append(key)
        else:
            row["cg_iterations"] = ""

    missing = expected - seen
    if missing:
        raise ValueError(
            f"Missing {len(missing)} configured measurements. "
            f"First: {min(missing)}."
        )
    return [
        f"Validated {len(rows)} measurements across {len(selected)} cases.",
        f"Each configured case and k has {repeats} repetitions.",
        f"Non-converged solver measurements: {len(failed)}.",
        (
            "CG iterations are the recorded preconditioner applications. "
            "SciPy CG applies M once per iteration in these solves."
        ),
        "Non-converged measurements are excluded from summaries and plots.",
    ]


def _summary(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["study"], row["case"], int(row["k"]))].append(row)
    summary = []
    for (study, case, k), group in sorted(groups.items()):
        valid = [row for row in group if row["converged"] != "False"]
        for metric in METRICS:
            values = [
                float(row[metric])
                for row in valid
                if row.get(metric) not in ("", None)
            ]
            if not values:
                continue
            data = np.asarray(values)
            summary.append(
                {
                    "study": study,
                    "case": case,
                    "k": k,
                    "npoints": k * k,
                    "metric": metric,
                    "n": len(data),
                    "n_excluded": len(group) - len(valid),
                    "median": float(np.median(data)),
                    "q25": float(np.quantile(data, 0.25)),
                    "q75": float(np.quantile(data, 0.75)),
                    "minimum": float(np.min(data)),
                    "maximum": float(np.max(data)),
                    "mean": float(np.mean(data)),
                    "std": (
                        float(np.std(data, ddof=1)) if len(data) > 1 else 0.0
                    ),
                }
            )
    return summary


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _scaling(summary: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in summary:
        if (
            row["metric"] in ("total_seconds", "peak_increment_mib")
            and row["median"] > 0
        ):
            groups[(row["study"], row["case"], row["metric"])].append(row)
    results = []
    for (study, case, metric), group in sorted(groups.items()):
        group.sort(key=lambda row: row["npoints"])
        if len(group) < 3:
            continue
        x = np.log([row["npoints"] for row in group])
        y = np.log([row["median"] for row in group])
        slope, intercept = np.polyfit(x, y, 1)
        predicted = slope * x + intercept
        total = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - np.sum((y - predicted) ** 2) / total if total else 1.0
        results.append(
            {
                "study": study,
                "case": case,
                "metric": metric,
                "exponent": float(slope),
                "r_squared": float(r_squared),
                "n_sizes": len(group),
                "n_min": group[0]["npoints"],
                "n_max": group[-1]["npoints"],
            }
        )
    return results


def _label(case: str) -> str:
    preconditioners = {
        "preconditioner-none": "identity",
        "preconditioner-banded": "circulant banded",
        "preconditioner-dense": "circulant dense",
    }
    if case in preconditioners:
        return preconditioners[case]
    label = case.split("-", 1)[1].replace("-", " ")
    return (
        label.replace("toeplitz", "Toeplitz")
        .replace("gaussian", "Gaussian")
        .replace("fft", "FFT")
    )


def _panel(
    ax,
    rows: list[dict],
    study: str,
    metric: str,
    title: str,
    ylabel: str,
    cases: list[str],
) -> None:
    k_values: set[int] = set()
    for index, case in enumerate(cases):
        points = sorted(
            [
                row
                for row in rows
                if row["case"] == case and row["metric"] == metric
            ],
            key=lambda row: row["npoints"],
        )
        if not points:
            continue
        k_values.update(row["k"] for row in points)
        x = np.array([row["npoints"] for row in points])
        median = np.array([row["median"] for row in points])
        q25 = np.array([row["q25"] for row in points])
        q75 = np.array([row["q75"] for row in points])
        color = PALETTE(index % 10)
        ax.plot(
            x, median, "o-", label=_label(case), color=color, linewidth=1.8
        )
        ax.fill_between(x, q25, q75, color=color, alpha=0.16)
    ax.set_xscale("log", base=10)
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel(r"signal length $N = k^2$")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(True, which="minor", alpha=0.10)

    ks = sorted(k_values)
    top_axis = ax.secondary_xaxis("top")
    top_axis.set_xscale("log", base=2)
    top_axis.set_xticks(
        [k * k for k in ks],
        labels=[str(k) for k in ks],
    )
    top_axis.xaxis.set_minor_locator(NullLocator())
    top_axis.grid(False)
    ax.text(
        -0.03,
        1.02,
        r"$k =$",
        transform=top_axis.transAxes,
        ha="right",
        va="bottom",
    )


def _figure() -> tuple:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.subplots_adjust(
        left=0.11,
        right=0.96,
        bottom=0.09,
        top=0.80,
        wspace=0.24,
        hspace=0.45,
    )
    return fig, axes


def _figure_heading(fig, axes, title: str, legend_columns: int = 3) -> None:
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle(title, y=0.98, fontsize=15)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=legend_columns,
        frameon=False,
    )


def _save_figures(
    summary: list[dict], output_dir: Path, selected_cases: list[str]
) -> list[str]:
    figures = []
    layouts = {
        "operators": (
            ("total_seconds", "Total Time", "time / s"),
            ("setup_seconds", "Setup Time", "time / s"),
            ("solve_seconds", "Solve Time", "time / s"),
            ("peak_increment_mib", "Peak Memory", "peak RSS increase / MiB"),
        ),
        "preconditioners": (
            ("total_seconds", "Total Time", "time / s"),
            ("solve_seconds", "Solve Time", "time / s"),
            ("cg_iterations", "CG Iterations", "iterations"),
            ("peak_increment_mib", "Peak Memory", "peak RSS increase / MiB"),
        ),
    }
    for study, panels in layouts.items():
        if not any(row["study"] == study for row in summary):
            continue
        fig, axes = _figure()
        prefix = {
            "operators": "operator-",
            "preconditioners": "preconditioner-",
        }[study]
        cases = [case for case in selected_cases if case.startswith(prefix)]
        for ax, (metric, title, ylabel) in zip(axes.flat, panels):
            _panel(ax, summary, study, metric, title, ylabel, cases)
        heading = {
            "operators": "Operator Methods",
            "preconditioners": "Preconditioners",
        }[study]
        _figure_heading(
            fig, axes, heading, legend_columns=5 if study == "operators" else 3
        )
        path = output_dir / f"{study}.svg"
        fig.savefig(path)
        plt.close(fig)
        figures.append(path.name)

    if any(
        row["study"] in ("projection", "reconstruction") for row in summary
    ):
        fig, axes = _figure()
        for row_index, study in enumerate(("projection", "reconstruction")):
            for column, (metric, title, ylabel) in enumerate(
                (
                    ("total_seconds", "Time", "time / s"),
                    (
                        "peak_increment_mib",
                        "Peak Memory",
                        "peak RSS increase / MiB",
                    ),
                )
            ):
                cases = [
                    case
                    for case in selected_cases
                    if case.startswith(study + "-")
                ]
                _panel(
                    axes[row_index, column],
                    summary,
                    study,
                    metric,
                    f"{study.title()} {title}",
                    ylabel,
                    cases,
                )
        _figure_heading(fig, axes, "Signal Projection and Reconstruction")
        path = output_dir / "basis_methods.svg"
        fig.savefig(path)
        plt.close(fig)
        figures.append(path.name)
    return figures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("benchmarks/results/benchmarks.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/reports"),
    )
    args = parser.parse_args()
    rows, metadata = _read_data(args.input)
    messages = _validate(rows, metadata)
    summary = _summary(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "summary.csv", summary)
    _write_csv(args.output_dir / "scaling.csv", _scaling(summary))
    figures = _save_figures(
        summary, args.output_dir, metadata["studies"]["cases"]
    )
    (args.output_dir / "validation.txt").write_text(
        "\n".join(messages) + "\n", encoding="utf-8"
    )
    for message in messages:
        print(message)
    print(
        f"Wrote summary, scaling estimates, and {len(figures)} figures to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
