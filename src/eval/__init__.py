"""The shared evaluation harness.

PROJECT_PLAN.md §8: "every method is evaluated identically. The benchmark table itself is a
contribution." Every model -- Heston now, the flow-matched neural SDE in Phase 2 -- is scored
through this package and lands as a row in reports/benchmark.md.

Hard rule: `src/eval` must NEVER import `src/algorithms`. The harness cannot be allowed to
learn which model produced the numbers it is scoring, and keeping the dependency one-way is
what enforces that. Models depend on the harness's contracts (see paths.Paths), not the
reverse.

`plots` is deliberately NOT re-exported here: importing any submodule executes this file, and
heston_mc imports paths -- so re-exporting plots would drag matplotlib onto the library's
import path for every calibration run.
"""
from .eval_config import BenchmarkConfig, SplitConfig
from .paths import Paths

__all__ = ["BenchmarkConfig", "Paths", "SplitConfig"]
