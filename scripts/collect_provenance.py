"""Collect run provenance into ``results/provenance.json``.

The manuscript pulls its compute figure from this file rather than from a
hand-typed constant, so a stale or placeholder GPU-hours number is
structurally impossible: if the runs are not on disk, the number is not
emitted.

Usage::

    python scripts/collect_provenance.py
    python scripts/collect_provenance.py --check   # non-zero exit if stale
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
MASTER_LOG = RESULTS / "MASTER_LOG.csv"
OUT = RESULTS / "provenance.json"


def _run(cmd: list[str]) -> str:
    """Run a command and return stripped stdout, or ``""`` on failure."""
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=30, cwd=ROOT
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def git_info() -> dict[str, Any]:
    """Current commit, branch, and whether the working tree is dirty."""
    sha = _run(["git", "rev-parse", "HEAD"])
    dirty = bool(_run(["git", "status", "--porcelain", "--untracked-files=no"]))
    return {
        "sha": sha,
        "short_sha": sha[:7],
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": dirty,
    }


def hardware_info() -> dict[str, Any]:
    """GPU model/memory/driver plus host CPU and platform strings."""
    gpus: list[dict[str, Any]] = []
    raw = _run([
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,driver_version,compute_cap",
        "--format=csv,noheader",
    ])
    for line in (ln for ln in raw.splitlines() if ln.strip()):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5:
            gpus.append({
                "index": int(parts[0]),
                "name": parts[1],
                "memory_total": parts[2],
                "driver_version": parts[3],
                "compute_capability": parts[4],
            })
    if gpus:
        names = sorted({g["name"] for g in gpus})
        hw_string = f"{len(gpus)}x {', '.join(names)} (driver {gpus[0]['driver_version']})"
    else:
        hw_string = "no NVIDIA GPU detected"
    return {
        "gpus": gpus,
        "hardware_string": hw_string,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
    }


def env_lock_hash() -> dict[str, Any]:
    """sha256 of ``configs/environment.lock``, plus key package versions."""
    lock = ROOT / "configs" / "environment.lock"
    digest = None
    if lock.exists():
        digest = hashlib.sha256(lock.read_bytes()).hexdigest()
    versions: dict[str, str] = {}
    for pkg in ("torch", "numpy", "scipy", "pettingzoo", "mpe2", "gymnasium"):
        try:
            from importlib.metadata import version

            versions[pkg] = version(pkg)
        except Exception:  # noqa: BLE001 - package simply absent
            versions[pkg] = "absent"
    try:
        import torch

        versions["torch_cuda"] = torch.version.cuda or "cpu"
        versions["torch_arch_list"] = ",".join(torch.cuda.get_arch_list())
    except Exception:  # noqa: BLE001
        pass
    return {"environment_lock_sha256": digest, "versions": versions}


def config_hash(path: Path) -> str | None:
    """sha256 of a config file, or ``None`` if it is missing."""
    p = path if path.is_absolute() else ROOT / path
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def load_runs() -> list[dict[str, Any]]:
    """Per-run provenance rows read from ``MASTER_LOG.csv``."""
    if not MASTER_LOG.exists():
        return []
    with MASTER_LOG.open() as f:
        rows = list(csv.DictReader(f))
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append({
            "run_id": r.get("run_id") or f"{r.get('run_name')}/seed_{r.get('seed')}",
            "run_name": r.get("run_name"),
            "seed": _as_int(r.get("seed")),
            "method": r.get("method"),
            "env": r.get("env"),
            "n_agents": _as_int(r.get("n_agents")),
            "git_sha": r.get("git_sha"),
            "total_steps": _as_int(r.get("total_steps")),
            "wall_time_s": _as_float(r.get("wall_time_s")),
            "peak_vram_mib": _as_float(r.get("peak_vram_mib")),
            "hardware": r.get("hardware"),
            "config": r.get("config"),
            "config_sha256": r.get("config_sha256"),
            "output_path": r.get("output_path"),
        })
    return out


def _as_int(v: str | None) -> int | None:
    try:
        return int(float(v))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _as_float(v: str | None) -> float | None:
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def orphan_runs() -> list[str]:
    """Run directories present on disk but absent from ``MASTER_LOG.csv``.

    A run that is not in the log does not exist, so these are defects.
    """
    logged = {r["run_id"] for r in load_runs()}
    found: list[str] = []
    for cfg in RESULTS.glob("*/seed_*/config.json"):
        rid = f"{cfg.parent.parent.name}/{cfg.parent.name}"
        if rid not in logged:
            found.append(rid)
    return sorted(found)


def build() -> dict[str, Any]:
    """Assemble the full provenance record."""
    runs = load_runs()
    wall = [r["wall_time_s"] for r in runs if r["wall_time_s"] is not None]
    vram = [r["peak_vram_mib"] for r in runs if r["peak_vram_mib"] is not None]
    total_wall_s = float(sum(wall))
    return {
        "git": git_info(),
        "hardware": hardware_info(),
        "environment": env_lock_hash(),
        "runs": {
            "n_runs": len(runs),
            "n_orphans": len(orphan_runs()),
            "orphans": orphan_runs(),
            "total_wall_clock_s": total_wall_s,
            "total_wall_clock_hours": round(total_wall_s / 3600.0, 2),
            # Only emitted when peak VRAM was actually sampled; absent
            # otherwise rather than silently defaulting.
            "peak_vram_mib_max": max(vram) if vram else None,
            "total_env_steps": sum(
                r["total_steps"] for r in runs if r["total_steps"] is not None
            ),
        },
        "per_run": runs,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--check", action="store_true",
                   help="Exit non-zero if the tree is dirty or runs are unlogged.")
    args = p.parse_args()

    rec = build()
    Path(args.out).write_text(json.dumps(rec, indent=2) + "\n")
    hrs = rec["runs"]["total_wall_clock_hours"]
    print(f"wrote {args.out}")
    print(f"  git      {rec['git']['short_sha']} "
          f"({'dirty' if rec['git']['dirty'] else 'clean'}) on {rec['git']['branch']}")
    print(f"  hardware {rec['hardware']['hardware_string']}")
    print(f"  runs     {rec['runs']['n_runs']} logged, "
          f"{rec['runs']['n_orphans']} unlogged on disk")
    print(f"  compute  {hrs} wall-clock hours, "
          f"{rec['runs']['total_env_steps']:,} env steps")

    if args.check:
        problems = []
        if rec["git"]["dirty"]:
            problems.append("working tree is dirty")
        if rec["runs"]["n_orphans"]:
            problems.append(
                f"{rec['runs']['n_orphans']} run(s) on disk are absent from "
                f"MASTER_LOG.csv: {rec['runs']['orphans']}"
            )
        if problems:
            for msg in problems:
                print(f"FAIL: {msg}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
