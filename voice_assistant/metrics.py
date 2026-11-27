
"""Performance monitoring for the offline voice assistant."""

import logging
from contextlib import contextmanager
from time import perf_counter
from collections.abc import Iterator

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """Collect and report execution times in seconds."""

    def __init__(self):
        self.timings: dict[str, float] = {}

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        start = perf_counter()

        try:
            yield
        finally:
            self.timings[stage] = perf_counter() - start

    @property
    def response_latency(self) -> float:
        stages = ("stt", "llm", "tts_synthesis")

        return sum(
            self.timings.get(stage, 0.0)
            for stage in stages
        )

    @property
    def total_time(self) -> float:
        return sum(self.timings.values())

    def report(self, language: str) -> None:
        logger.info("=" * 45)
        logger.info("PERFORMANCE REPORT [%s]", language)
        logger.info("=" * 45)

        for stage, duration in self.timings.items():
            logger.info(
                "%-20s %.3f s",
                stage,
                duration,
            )

        logger.info(
            "Response latency:    %.3f s",
            self.response_latency,
        )

        logger.info(
            "Total measured time: %.3f s",
            self.total_time,
        )

        logger.info("=" * 45)
