"""Round 3 (tasks/plan_round3.md) on the clinical W6 pruned instances, local VM, aggregates only.
  python scripts/round3_w6.py kappa   # Q2: W6 primary with kappa_MLAE (clinical) and instance-specific kappa (Q3 map)
  python scripts/round3_w6.py pk      # Q4: Pauker-Kassirer thresholds, two decision metrics, all test instances
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
import bands as B  # noqa: E402

QP = Path("artifacts/quantum_paper")


def frame() -> pd.DataFrame:
    """W6 instances on the pruned network (as bands.w6_summary): p, LW samples, a_E, V1 ratio, all test instances."""
    v1 = json.loads((B.ART / "v1_counts.json").read_text())
    parts = []
    for b in B.PRIMARY:
        X = B._instances(b, v1)
        w = pd.concat([pd.read_parquet(q) for q in sorted((B.ART / b).glob("w6_*.parquet"))]).drop_duplicates(["pattern", "target"])
        X = X.merge(w[["pattern", "target", "lw_samples", "p_evidence", "logW", "ratio_v1", "evidence_given", "evidence_used", "nodes_kept"]].rename(
            columns=lambda c: c if c in ("pattern", "target") else f"w_{c}"), on=["pattern", "target"], how="left")
        assert X.w_lw_samples.notna().all(), b
        parts.append(pd.DataFrame({"band": b, "subject_id": X.subject_id, "target": X.target, "p": X.p,
                                   "lw_samples": X.w_lw_samples, "aE": X.w_p_evidence / np.exp(X.w_logW), "ratio_v1": X.w_ratio_v1,
                                   "evidence_given": X.w_evidence_given, "evidence_used": X.w_evidence_used, "nodes_kept": X.w_nodes_kept}))
    return pd.concat(parts, ignore_index=True)


def kappa():
    from w8 import kappa_map_at
    from scipy.interpolate import RegularGridInterpolator  # noqa: F401  (kappa_map_at needs scipy)
    X = frame()
    W = X[(X.p >= B.WIN[0]) & (X.p <= B.WIN[1])].reset_index(drop=True)
    k_iae = json.loads((B.RES / "frontier.json").read_text())["kappa_tuned_iae"]
    kc = json.loads((QP / "kappa_clinical.json").read_text())
    m = json.loads((QP / "kappa_map.json").read_text())

    def mapped(est):
        from scipy.ndimage import distance_transform_edt
        M = np.array(m["estimators"][est]["log10_kappa"], float)
        _, (ii, jj) = distance_transform_edt(~np.isfinite(M), return_indices=True)
        f = RegularGridInterpolator((np.log10(m["a_E"]), np.log10(m["p"])), M[ii, jj])
        la, lp = np.log10(m["a_E"]), np.log10(m["p"])
        return 10 ** f(np.column_stack([np.clip(np.log10(W.aE), la[0], la[-1]), np.clip(np.log10(W.p), lp[0], lp[-1])]))
    from scipy.interpolate import RegularGridInterpolator
    ki, km = mapped("kappa_iae"), mapped("kappa_mlae16")
    variants = {"published: IAE, constant 11.69": k_iae,
                "MLAE16, constant (clinical calibration)": kc["mlae16"]["measured_median"],
                "IAE, per instance (Q3 map)": ki,
                "MLAE16, per instance (Q3 map; map fails its test)": km,
                "best of IAE/MLAE16 per instance (Q3 map)": np.minimum(ki, km)}
    out = {"population": "W6 primary: common + uncommon, connected targets, window p in [0.001, 0.04], decision t = 1%, V1 gates, pruned",
           "clinical_kappa": kc, "variants": {}}
    for name, k in variants.items():
        W["lr"] = B._log_ratio(W, B.T_DEC, 1.0, "v1") + np.log10(k)  # R is linear in kappa
        out["variants"][name] = {"median_kappa": float(np.median(k)), **B._endpoint(W, "lr")}
        print(name, json.dumps(out["variants"][name]), flush=True)
    (QP / "w6_kappa.json").write_text(json.dumps(out, indent=1))


def pk():
    """Q4. t = H/(H+B). Metric A: absolute error 0.1 max(p, t). Metric B (decision margin): max(0.1 max(p, t), |p - t|/2).
    LW cost N_lw(10%) (0.1 p / err)^2, floored at one sample; qLW kappa pi p / err (1/sqrt(a_E) + 1/sqrt(a_T)) model
    evaluations, floored at one per amplitude (2), times the V1 gate ratio. Descriptive, no pass rule."""
    X = frame()
    k = json.loads((B.RES / "frontier.json").read_text())["kappa_tuned_iae"]
    out = {"population": "W6 pruned, common + uncommon, connected targets, all test instances", "instances": int(len(X)),
           "patients": int(X.subject_id.nunique()), "kappa": k, "thresholds": {}}
    for hb in (1 / 100, 1 / 50, 1 / 20, 1 / 10, 1 / 5, 1 / 2, 1):
        t = hb / (1 + hb)
        rec = {"t": t}
        for name, err in (("error_0.1max(p,t)", 0.1 * np.maximum(X.p, t)),
                          ("decision_margin", np.maximum(0.1 * np.maximum(X.p, t), np.abs(X.p - t) / 2))):
            eps = err / X.p
            lw = np.maximum(X.lw_samples * (B.REL / eps) ** 2, 1.0)
            q = np.maximum(k * math.pi / eps * (1 / np.sqrt(X.aE) + 1 / np.sqrt(X.aE * X.p)), 2.0) * X.ratio_v1
            lr = np.log10(q / lw)
            rec[name] = {"median_ratio": float(10 ** np.median(lr)), "share_qlw_cheaper": float((lr < 0).mean()),
                         "by_band": {b: {"median_ratio": float(10 ** np.median(lr[X.band == b])),
                                         "share_qlw_cheaper": float((lr[X.band == b] < 0).mean())} for b in B.PRIMARY},
                         "share_lw_at_one_sample_floor": float((X.lw_samples * (B.REL / eps) ** 2 < 1).mean())}
        out["thresholds"][f"H/B={hb:g}"] = rec
        print(f"H/B={hb:g} t={t:.4f}", {n: (round(v["median_ratio"], 3), round(v["share_qlw_cheaper"], 3)) for n, v in rec.items() if n != "t"}, flush=True)
    (QP / "w6_pauker_kassirer.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"kappa": kappa, "pk": pk}[sys.argv[1]]()
