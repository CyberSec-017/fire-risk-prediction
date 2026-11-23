"""Regenerate the statistical-validation and model-benchmark artifacts.

This script only reads observational datasets and archived experiment CSVs.
It never loads, refits, overwrites or saves models/fire_model.joblib or the
model card. The deployed 20-feature contract is used as a read-only assertion.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
for candidate in (str(ROOT / "src"), str(ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import config  # noqa: E402
import model_benchmark as mb  # noqa: E402
import statistical_validation as sv  # noqa: E402
from train_baseline import load_data, time_split  # noqa: E402


def _write_json(data: dict, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(f"{target.suffix}.tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        import json
        json.dump(data, handle, indent=2, ensure_ascii=False)
    tmp.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=sv.DEFAULT_ALPHA)
    parser.add_argument("--seed", type=int, default=sv.DEFAULT_SEED)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    print("Building statistical report from observed fire panel…")
    panel = pd.read_parquet(config.FIRE_PANEL)
    report = sv.build_statistical_report(panel, alpha=args.alpha, seed=args.seed)
    _write_json(report, output_dir / sv.STATISTICAL_REPORT_PATH.name)
    print(f"  {len(report['tests'])} tests -> {output_dir / sv.STATISTICAL_REPORT_PATH.name}")

    print("Building benchmark from archived experiments + identical-split prior baseline…")
    df, features = load_data()
    train, val, _ = time_split(df)
    benchmark, _ = mb.build_benchmark_artifacts(
        output_dir=output_dir,
        train_df=train,
        val_df=val,
        features=features,
        seed=args.seed,
    )
    print(f"  {len(benchmark)} rows -> {output_dir / mb.BENCHMARK_CSV.name}")
    print(f"  protocol -> {output_dir / mb.BENCHMARK_PROTOCOL_JSON.name}")
    print("Production model artifacts were neither loaded for inference nor modified.")


if __name__ == "__main__":
    main()
