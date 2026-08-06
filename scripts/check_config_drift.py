"""Fail if a YAML config disagrees with the runs that were produced from it.

`reproduce.sh` reads `total_steps` from the YAML. If a completed run under
`results/` was trained with a different value, re-running the script produces
different numbers than the ones in the manuscript while appearing to succeed.
That is exactly what had happened (AUDIT.md D-15): the N=3 and N=6 YAMLs
claimed 1.5M and 1.0M steps while every run on disk recorded 800k.

Exit code 0 if consistent, 1 otherwise.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CONFIGS = ROOT / "configs"

#: Trainer keys whose drift would change the reported numbers.
CRITICAL_KEYS: tuple[str, ...] = (
    "total_steps", "n_envs", "rollout_len", "ppo_epochs",
    "n_minibatches", "lr", "gamma", "gae_lambda", "clip_ratio",
    "entropy_coef", "value_coef", "per_agent_reward",
)


def load_yaml(path: Path) -> dict[str, Any]:
    """Parse a run config, tolerating YAML's `1_000_000` underscore form."""
    text = path.read_text()
    cfg = yaml.safe_load(text)
    trainer = cfg.get("trainer", {})
    for k, v in list(trainer.items()):
        if isinstance(v, str) and v.replace("_", "").isdigit():
            trainer[k] = int(v.replace("_", ""))
    return cfg


def completed_runs(run_name: str) -> list[Path]:
    """config.json paths for every finished seed of a run."""
    return sorted(
        p for p in RESULTS.glob(f"{run_name}/seed_*/config.json")
        if (p.parent / "final_eval.json").exists()
    )


def main() -> int:
    problems: list[str] = []
    checked = 0

    for cfg_path in sorted(CONFIGS.glob("*.yaml")):
        cfg = load_yaml(cfg_path)
        run_name = cfg.get("run_name")
        if not run_name:
            continue
        runs = completed_runs(run_name)
        if not runs:
            continue  # nothing produced yet; nothing to contradict
        want = cfg.get("trainer", {})
        for rp in runs:
            got = json.loads(rp.read_text()).get("trainer", {})
            checked += 1
            for key in CRITICAL_KEYS:
                if key not in want:
                    continue
                if key in got and got[key] != want[key]:
                    problems.append(
                        f"{cfg_path.name}: {key}={want[key]!r} but "
                        f"{rp.parent.relative_to(ROOT)} was run with {got[key]!r}"
                    )

    if problems:
        print(f"config drift detected in {len(problems)} field(s):",
              file=sys.stderr)
        for p in sorted(set(problems)):
            print(f"  {p}", file=sys.stderr)
        return 1

    print(f"config/run consistency OK ({checked} completed runs checked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
