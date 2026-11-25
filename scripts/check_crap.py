"""CHANGE RISK (CRAP) gate for CI — fails when a measured function regresses.

Runs after ``pytest --cov=src`` so the saved ``.coverage`` file carries the
data. Reuses ``pytest_crap.calculator.calculate_crap`` (the same AST mapping
and formula as the pytest-crap report) so the gate can never drift from the
report.

Ratchet semantics (pre-existing offenders): a BASELINE file lists functions
that already exceeded the threshold BEFORE a change (file:line label + the
CRAP score at the time). The gate

* FAILS on any offender NOT listed in the baseline (a new offender), and
* FAILS when a baseline-listed function's CRAP score rises above its recorded
  value (a regression — the fix direction is only ever down).

Usage:
    uv run python scripts/check_crap.py --source src --threshold 30 \
        --baseline scripts/crap-baseline.txt

Exits non-zero listing every violation.
"""

import argparse
import sys
from pathlib import Path

import coverage
from pytest_crap.calculator import calculate_crap

_BASELINE_ENTRY = "{path}:{start_line}:{name}:{crap:.2f}"


def _covered_lines_per_file(coverage_file: str) -> dict[str, set[int]]:
    """Load executed lines per measured file from the pytest-cov data file."""
    cov_data = coverage.CoverageData(basename=coverage_file)
    cov_data.read()
    lines_by_file: dict[str, set[int]] = {}
    for measured in cov_data.measured_files():
        lines_by_file[measured] = set(cov_data.lines(measured) or [])
    return lines_by_file


def _python_files(source_root: Path) -> list[Path]:
    """Collect the Python files under the source root (deterministic order)."""
    return sorted(source_root.rglob("*.py"))


def _load_baseline(baseline_path: Path) -> dict[str, float]:
    """Load ``path:name:crap`` allowances; missing file ⇒ empty ratchet."""
    if not baseline_path.exists():
        return {}
    allowances: dict[str, float] = {}
    for raw in baseline_path.read_text().splitlines():
        entry = raw.split("#")[0].strip()
        if not entry:
            continue
        path, name, crap = entry.rsplit(":", 2)
        allowances[f"{path}:{name}"] = float(crap)
    return allowances


def main() -> int:
    """Run the CRAP gate over every function under ``--source``."""
    parser = argparse.ArgumentParser(description="CHANGE RISK (CRAP) gate")
    parser.add_argument("--source", default="src", help="source tree to gate (default: src)")
    parser.add_argument("--threshold", type=float, default=30.0, help="maximum CRAP score per function")
    parser.add_argument("--coverage-file", default=".coverage", help="pytest-cov data file")
    parser.add_argument(
        "--baseline",
        default="",
        help="ratchet file with pre-existing allowances 'path:line:name:crap' (one per line)",
    )
    args = parser.parse_args()

    covered_by_file = _covered_lines_per_file(args.coverage_file)
    source_root = Path(args.source).resolve()
    allowances = _load_baseline(Path(args.baseline)) if args.baseline else {}

    new_offenders: list[str] = []
    regressions: list[str] = []
    stale_allowances = set(allowances)
    checked = 0
    for path in _python_files(source_root):
        measured = covered_by_file.get(str(path))
        if measured is None:
            continue  # never executed by the suite — flagged by coverage itself
        for score in calculate_crap(str(path), measured):
            checked += 1
            if score.crap < args.threshold:
                continue
            rel = score.file_path.replace(f"{source_root}/", "src/")
            entry = f"{rel}:{score.start_line}:{score.name}:{score.crap:.2f}"
            allowance = allowances.get(f"{rel}:{score.name}")
            if allowance is None:
                new_offenders.append(f"{entry} (CC {score.cc}, coverage {score.coverage_percent:.1f}%)")
                continue
            stale_allowances.discard(f"{rel}:{score.name}")
            # Ratchet comparison at the RECORDED precision (2 decimals): the
            # raw float always carries more digits than the baseline string.
            if round(score.crap, 2) > round(allowance, 2):
                regressions.append(
                    f"{rel}:{score.name} — CRAP rose from {allowance:.2f} to {score.crap:.2f}"
                    f" (now at line {score.start_line})"
                )

    print(f"CRAP gate: {checked} functions checked (threshold {args.threshold:g}, source '{args.source}')")
    if new_offenders:
        print(f"CRAP gate FAILED — {len(new_offenders)} NEW function(s) above the threshold:")
        for offender in new_offenders:
            print(f"  - {offender}")
    if regressions:
        print(f"CRAP gate FAILED — {len(regressions)} baseline functions REGRESSED:")
        for offender in regressions:
            print(f"  - {offender}")
    for stale in sorted(stale_allowances):
        print(f"CRAP gate note: baseline entry no longer offends (prune it): {stale}")
    if not new_offenders and not regressions:
        print("CRAP gate PASSED — no new offender and no baseline regression")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
