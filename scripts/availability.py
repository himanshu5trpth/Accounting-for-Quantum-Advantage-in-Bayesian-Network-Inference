"""Which candidate items are recorded for most admissions? (training split only; reads the discover caches)

Availability = share of training admissions where the item's source was recorded at admission:
  lab          the test was measured in the first 24 h (normal or abnormal)
  medication   the admission has any first-24 h medication record
  history      the patient has an earlier coded admission
  demographic  always
Run after `qddx discover`:  python scripts/availability.py [--min 0.5]
Writes reports/availability.md and artifacts/vocab/availability.csv.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

V = Path("artifacts/vocab")


def main(min_avail: float, top: int):
    parts = {s: pd.read_parquet(V / f"part_{s}_v2.parquet") for s in ("lab", "rx", "hx", "demo")}
    n = parts["demo"].hadm_id.nunique()  # every training admission has demographics
    src_avail = {"medication": parts["rx"].hadm_id.nunique() / n, "history": parts["hx"].hadm_id.nunique() / n, "demographic": 1.0}
    lab_avail = (parts["lab"].groupby("item").size() / n).to_dict()  # lab rows exist only when measured (0 normal, 1 abnormal)

    cands = json.loads((V / "candidates.json").read_text())
    rows = []
    for c in cands:
        a = lab_avail.get(c["item"], 0.0) if c["source"] == "lab" else src_avail[c["source"]]
        cov = (c.get("note") or {}).get("coverage")
        rows.append({"rank": c["rank"], "item": c["item"], "title": c["title"], "source": c["source"], "availability": a,
                     "prevalence": c["prevalence_inner_train"], "note_coverage": cov, "excluded": c["excluded_by_default"],
                     "flags": ";".join(f.split(":")[0] for f in c["flags"])})
    df = pd.DataFrame(rows).sort_values("rank")
    df.to_csv(V / "availability.csv", index=False)

    ok = df[(df.availability >= min_avail) & ~df.excluded]
    L = ["# Item availability (training admissions)", "",
         f"{n:,} training admissions. Recorded at admission: any medication record {src_avail['medication']:.0%}, "
         f"an earlier coded admission (history) {src_avail['history']:.0%}, demographics 100%; labs per test below.", "",
         f"## Clean candidates recorded for >= {min_avail:.0%} of admissions ({len(ok)} of {(~df.excluded).sum()})", "",
         "| rank | item | source | recorded | positive | note coverage |", "|---|---|---|---|---|---|"]
    for r in ok.head(top).itertuples():
        cov = "structured" if r.source == "demographic" else (f"{r.note_coverage:.0%}" if pd.notna(r.note_coverage) else "n/a")
        L.append(f"| {r.rank} | {r.title[:70]} | {r.source} | {r.availability:.0%} | {r.prevalence:.1%} | {cov} |")
    labs = df[df.source == "lab"].drop_duplicates("item").sort_values("availability", ascending=False)
    L += ["", "## Lab tests by how often they are measured in the first 24 h", "",
          "| test (direction) | measured | excluded |", "|---|---|---|"]
    L += [f"| {r.title} | {r.availability:.0%} | {r.flags if r.excluded else ''} |" for r in labs.head(60).itertuples()]
    out = Path("reports/availability.md")
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L[:8 + min(top, len(ok))]))
    print(f"\nwrote {out} and {V / 'availability.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--min", type=float, default=0.5, help="minimum share of admissions with the item recorded")
    ap.add_argument("--top", type=int, default=40)
    a = ap.parse_args()
    main(a.min, a.top)
