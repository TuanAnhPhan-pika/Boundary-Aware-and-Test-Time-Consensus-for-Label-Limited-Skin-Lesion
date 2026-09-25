"""Minimal time/metric harness required by the generated experiment."""

import json
import math
import time
from pathlib import Path


class ExperimentHarness:
    def __init__(self, time_budget):
        self.time_budget = float(time_budget)
        self.started = time.monotonic()
        self.metrics = {}

    def should_stop(self):
        return (time.monotonic() - self.started) >= 0.8 * self.time_budget

    def check_value(self, value, name):
        valid = math.isfinite(float(value))
        if not valid:
            self.metrics[name] = None
        return valid

    def report_metric(self, name, value):
        self.metrics[name] = float(value)

    def finalize(self):
        payload = {
            "elapsed_seconds": time.monotonic() - self.started,
            "time_budget_seconds": self.time_budget,
            "time_guard_triggered": self.should_stop(),
            "metrics": self.metrics,
        }
        Path("results.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
