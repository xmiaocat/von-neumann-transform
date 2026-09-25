"""Reusable timing and resident-memory measurements for benchmarks."""

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, Self, TypeVar

import psutil

T = TypeVar("T")


@dataclass(frozen=True)
class TimingMeasurement(Generic[T]):
    """The return value and elapsed wall-clock time of one operation."""

    value: T
    seconds: float


@dataclass(frozen=True)
class MemoryMeasurement(Generic[T]):
    """The return value and sampled RSS of one operation."""

    value: T
    baseline_rss_bytes: int
    peak_rss_bytes: int

    @property
    def peak_increment_bytes(self) -> int:
        return self.peak_rss_bytes - self.baseline_rss_bytes


class _PeakRSS:
    def __init__(self, interval: float) -> None:
        if interval <= 0.0:
            raise ValueError("RSS sampling interval must be positive.")
        self.interval = interval
        self.process = psutil.Process()
        self.baseline = 0
        self.peak = 0
        self._stop = threading.Event()

    def _sample(self) -> None:
        while not self._stop.wait(self.interval):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __enter__(self) -> Self:
        self.baseline = self.process.memory_info().rss
        self.peak = self.baseline
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.peak = max(self.peak, self.process.memory_info().rss)
        self._stop.set()
        self._thread.join()


def measure_time(operation: Callable[[], T]) -> TimingMeasurement[T]:
    """Run any zero-argument callable once and measure wall-clock time."""

    start = time.perf_counter()
    value = operation()
    return TimingMeasurement(value=value, seconds=time.perf_counter() - start)


def measure_peak_rss(
    operation: Callable[[], T], *, sample_interval: float = 0.001
) -> MemoryMeasurement[T]:
    """Run any zero-argument callable once and sample its process RSS.

    Keep timing and memory runs separate: the sampling thread intentionally
    adds a small amount of work while the operation is running.
    """

    with _PeakRSS(sample_interval) as monitor:
        value = operation()
    return MemoryMeasurement(
        value=value,
        baseline_rss_bytes=monitor.baseline,
        peak_rss_bytes=monitor.peak,
    )
