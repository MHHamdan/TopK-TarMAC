"""Fail the build if the bibliography contains uncited entries.

A bibliography padded with work the text never cites reads as citation
padding. This gate keeps the invariant "every entry is cited at least once"
mechanically true rather than aspirational.

Exit code 0 if clean, 1 if any entry is uncited or any citation is missing
from the .bib.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIB = ROOT / "literature" / "references.bib"
PAPER = ROOT / "paper"

#: Matches \cite, \citep, \citet, \citealp, \cite*, and optional-arg forms
#: such as \citep[see][p.~3]{key1,key2}.
CITE_RE = re.compile(r"\\cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}")
ENTRY_RE = re.compile(r"^@\w+\s*\{\s*([^,\s]+)\s*,", re.MULTILINE)


def bib_keys(bib_path: Path) -> list[str]:
    """Every entry key defined in the .bib file, in file order."""
    return [k.strip() for k in ENTRY_RE.findall(bib_path.read_text())]


def cited_keys(tex_root: Path) -> set[str]:
    """Every key referenced by any .tex file under ``tex_root``."""
    out: set[str] = set()
    for tex in sorted(tex_root.rglob("*.tex")):
        for group in CITE_RE.findall(tex.read_text()):
            for key in group.split(","):
                key = key.strip()
                if key:
                    out.add(key)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bib", default=str(BIB))
    p.add_argument("--tex-root", default=str(PAPER))
    p.add_argument("--list-uncited", action="store_true",
                   help="Print uncited keys and exit 0 (for pruning workflows).")
    args = p.parse_args()

    bib_path, tex_root = Path(args.bib), Path(args.tex_root)
    if not bib_path.exists():
        print(f"no bibliography at {bib_path}", file=sys.stderr)
        return 1

    keys = bib_keys(bib_path)
    cited = cited_keys(tex_root)

    dupes = sorted({k for k in keys if keys.count(k) > 1})
    uncited = [k for k in keys if k not in cited]
    dangling = sorted(c for c in cited if c not in set(keys))

    if args.list_uncited:
        for k in uncited:
            print(k)
        return 0

    print(f"bib entries: {len(keys)}   cited keys: {len(cited)}   "
          f"uncited: {len(uncited)}   dangling: {len(dangling)}")

    ok = True
    if dupes:
        ok = False
        print(f"FAIL: {len(dupes)} duplicate key(s): {dupes}", file=sys.stderr)
    if dangling:
        ok = False
        print(f"FAIL: {len(dangling)} citation(s) with no .bib entry:",
              file=sys.stderr)
        for k in dangling:
            print(f"  {k}", file=sys.stderr)
    if uncited:
        ok = False
        print(f"FAIL: {len(uncited)} uncited .bib entr(ies) — cite them in a "
              f"scoped sentence or delete them:", file=sys.stderr)
        for k in uncited:
            print(f"  {k}", file=sys.stderr)

    if ok:
        print("bibliography OK — every entry is cited, every citation resolves")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
