"""Run all benchmarks and dump results to results/*.json."""
from __future__ import annotations

import json
import os
import time

from benchmarks import (
    b1_accuracy,
    b2_efficiency,
    b3_robustness,
    b4_interpretability,
    b5_hdcnn,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

BENCHMARKS = {
    "b1_accuracy": b1_accuracy.run,
    "b2_efficiency": b2_efficiency.run,
    "b3_robustness": b3_robustness.run,
    "b4_interpretability": b4_interpretability.run,
    "b5_hdcnn": b5_hdcnn.run,
}


def main() -> None:
    env = {"host": os.uname().nodename,
           "python": __import__("sys").version,
           "numpy": __import__("numpy").__version__,
           "sklearn": __import__("sklearn").__version__,
           "torch": __import__("torch").__version__}
    for name, fn in BENCHMARKS.items():
        t0 = time.perf_counter()
        result = fn()
        result["_env"] = env
        path = os.path.join(RESULTS_DIR, f"{name}.json")
        with open(path, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"[{name}] done in {time.perf_counter() - t0:.1f}s -> {path}")


if __name__ == "__main__":
    main()
