"""Write corrected partial accounting without modifying benchmark artifacts."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.analysis.selector_costs import MODEL_VERSION, partial_counts


def derive() -> dict:
    rows = []
    for filename in ("prescore.json", "prescore_repeat.json", "prescore_selectors.json"):
        for r in json.loads((ROOT / "results" / filename).read_text())["rows"]:
            current = partial_counts(r["selector"], r["batch"], r["n_agents"],
                                     r["head_dim"], r["k"])
            rows.append({"source": filename, "selector_name": r["selector"],
                         "historical_ratio": r["flop_ratio_vs_dense"], **current})
    return {"model": MODEL_VERSION,
            "interpretation": "upper bound on the declared partial scalar count ratio; not latency or a complete selector count",
            "rows": rows}


if __name__ == "__main__":
    out = ROOT / "results" / "analysis" / "selector_costs_b1.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(derive(), indent=2) + "\n")
    print(f"wrote {out.relative_to(ROOT)}")
