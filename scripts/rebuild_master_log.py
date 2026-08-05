"""Rebuild ``results/MASTER_LOG.csv`` from the artefacts on disk.

The log is the register of record: a run that is not in it does not exist.
The previous log was missing one completed run and seven of the required
columns (AUDIT.md D-19).

Fields that cannot be recovered for historical runs -- peak VRAM was never
sampled -- are left **empty**. They are not back-filled with a guess.

Usage::

    python scripts/rebuild_master_log.py            # rewrite the log
    python scripts/rebuild_master_log.py --check    # verify, exit 1 on drift
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
MASTER_LOG = RESULTS / "MASTER_LOG.csv"

FIELDS: tuple[str, ...] = (
    "run_id", "git_sha", "method", "env", "n_agents", "seed",
    "total_steps", "wall_time_s", "peak_vram_mib", "hardware",
    "config", "config_sha256", "output_path",
    "final_eval_mean", "final_eval_std", "timestamp",
)

#: Historical runs predate per-run hardware capture; they were all produced on
#: the 4x RTX 2080 Ti host (see HARDWARE.md "Provenance of the existing runs").
LEGACY_HARDWARE = "4x NVIDIA GeForce RTX 2080 Ti (11264 MiB, driver 575.57.08)"


def method_of(model_cfg: dict[str, Any]) -> str:
    """Canonical arm name from a run's model config."""
    if not model_cfg.get("use_comm"):
        return "mappo"
    mode = model_cfg.get("attn_mode", "dense")
    if mode == "dense":
        return "dense_attn"
    if mode == "topk":
        return f"topk_fixed_k{model_cfg.get('topk', 0)}"
    if mode == "adaptive_topk":
        return "topk_adaptive"
    return mode


def last_wall_time(run_dir: Path) -> float | None:
    """Final ``wall_time_s`` recorded in ``metrics.csv``, if present."""
    m = run_dir / "metrics.csv"
    if not m.exists():
        return None
    try:
        with m.open() as f:
            rows = list(csv.DictReader(f))
        for row in reversed(rows):
            v = row.get("wall_time_s")
            if v:
                return float(v)
    except (OSError, ValueError):
        return None
    return None


def sha256_of(path: Path) -> str | None:
    p = path if path.is_absolute() else ROOT / path
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else None


def prior_rows() -> dict[str, dict[str, str]]:
    """Existing log rows keyed by ``run_name/seed_N``, for field carry-over."""
    if not MASTER_LOG.exists():
        return {}
    with MASTER_LOG.open() as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        rid = r.get("run_id") or f"{r.get('run_name')}/seed_{r.get('seed')}"
        out[rid] = r
    return out


def scan() -> list[dict[str, Any]]:
    """One row per completed run directory found under ``results/``."""
    prior = prior_rows()
    rows: list[dict[str, Any]] = []
    for cfg_path in sorted(RESULTS.glob("*/seed_*/config.json")):
        run_dir = cfg_path.parent
        if not (run_dir / "final_eval.json").exists():
            continue
        cfg = json.loads(cfg_path.read_text())
        model, trainer = cfg.get("model", {}), cfg.get("trainer", {})
        run_name, seed = run_dir.parent.name, cfg.get("seed")
        rid = f"{run_name}/seed_{seed}"
        ev = json.loads((run_dir / "final_eval.json").read_text())
        old = prior.get(rid, {})
        cfg_rel = f"configs/{run_name}.yaml"

        rows.append({
            "run_id": rid,
            "git_sha": old.get("git_sha", ""),
            "method": method_of(model),
            "env": "mpe_simple_spread_v3",
            "n_agents": model.get("n_agents"),
            "seed": seed,
            "total_steps": trainer.get("total_steps"),
            "wall_time_s": (old.get("wall_time_s")
                            or (f"{last_wall_time(run_dir):.6f}"
                                if last_wall_time(run_dir) is not None else "")),
            # Never sampled for these runs; deliberately blank, not guessed.
            "peak_vram_mib": old.get("peak_vram_mib", ""),
            "hardware": old.get("hardware") or LEGACY_HARDWARE,
            "config": cfg_rel,
            "config_sha256": sha256_of(Path(cfg_rel)) or "",
            "output_path": str(run_dir.relative_to(ROOT)),
            "final_eval_mean": ev.get("eval_return_mean"),
            "final_eval_std": ev.get("eval_return_std"),
            "timestamp": old.get("timestamp", ""),
        })
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true")
    args = p.parse_args()

    rows = scan()
    if args.check:
        existing = prior_rows()
        missing = [r["run_id"] for r in rows if r["run_id"] not in existing]
        if missing:
            print(f"FAIL: {len(missing)} completed run(s) absent from the log:",
                  file=sys.stderr)
            for m in missing:
                print(f"  {m}", file=sys.stderr)
            return 1
        print(f"MASTER_LOG.csv OK ({len(rows)} runs)")
        return 0

    with MASTER_LOG.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    n_vram = sum(1 for r in rows if r["peak_vram_mib"])
    print(f"wrote {MASTER_LOG.relative_to(ROOT)}: {len(rows)} runs, "
          f"{len(FIELDS)} columns")
    print(f"  peak_vram_mib populated for {n_vram}/{len(rows)} runs "
          f"(historical runs never sampled it; left blank rather than guessed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
