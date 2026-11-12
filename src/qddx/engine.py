"""Quantum-symbolic engine front end: posteriors over diagnoses with interchangeable backends, drivers,
and the deterministic f-string report (no language model in the output path)."""
from __future__ import annotations

from .bn import BN, Evidence, drivers, posterior_exact, posterior_lw, posterior_rejection
from .quantum import posterior_quantum

BACKENDS = ("exact", "rejection", "lw", "quantum", "quantum-emulated", "quantum-circuit")


def platt(p, a: float, b: float):
    """Monotone per-diagnosis recalibration fitted on validation patients (vocabulary option engine.calibration)."""
    import numpy as np
    q = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-(a * np.log(q / (1 - q)) + b)))


def calibrate(post: dict, cfg: dict) -> dict:
    cal = (cfg.get("engine") or {}).get("calibration")
    if not cal:
        return post
    return {d: float(platt(v, *cal[d])) if d in cal else v for d, v in post.items()}


def dx_names(cfg: dict) -> list[str]:
    return [k for k, p in cfg["propositions"].items() if p["kind"] == "diagnosis"]


def posterior(bn: BN, ev: Evidence, cfg: dict, backend: str = "exact", seed: int = 0):
    """-> ({diagnosis: P(d | E)}, P(E), cost)."""
    dx, e = dx_names(cfg), cfg["engine"]
    if backend == "exact":
        return posterior_exact(bn, ev, dx)
    if backend == "rejection":
        return posterior_rejection(bn, ev, dx, e["budget_samples"], seed)
    if backend == "lw":
        return posterior_lw(bn, ev, dx, e["budget_samples"], seed)
    if backend.startswith("quantum"):
        mode = {"quantum": "analytic", "quantum-emulated": "emulated", "quantum-circuit": "circuit"}[backend]
        return posterior_quantum(bn, ev, dx, e["eps"], e["alpha"], e["shots"], seed, mode=mode, rel=e.get("rel_eps"))
    raise ValueError(f"unknown backend {backend!r}; choose from {BACKENDS}")


def finding_text(k: str, m: float, cfg: dict) -> str:
    label = cfg["propositions"][k]["label"]
    if m >= 1.0:
        return label
    if m <= 0.0:
        return f"no {label}"
    return f"{label} (graded {m:.2f})"


def explain(bn: BN, m: dict, cfg: dict, backend: str = "exact", seed: int = 0) -> dict:
    """Posterior, ranking and leave-one-out drivers for one patient's memberships."""
    ev = Evidence.from_memberships(m)
    post, pe, cost = posterior(bn, ev, cfg, backend, seed)
    post = calibrate(post, cfg)
    ranked = sorted(post, key=lambda d: -post[d])
    drv = {d: drivers(bn, ev, d, k=3) for d in ranked[: cfg["engine"]["top_k"]]}
    return {"posterior": post, "p_evidence": pe, "cost": cost, "ranked": ranked, "drivers": drv}


def report(note_id: str, m: dict, res: dict, cfg: dict, truth: dict | None = None) -> str:
    """Every word below is a template; every number comes from `res`. Nothing is generated."""
    P = cfg["propositions"]
    lines = [f"Q-DDx differential for note {note_id} (admission information only)", ""]
    for i, d in enumerate(res["ranked"], 1):
        p = res["posterior"][d]
        why = res["drivers"].get(d)
        reason = "; ".join(f"{finding_text(f, m[f], cfg)} (+{eff:.2f})" for f, eff in why) if why else \
            ("no finding raises it above its baseline" if d in res["drivers"] else "")
        lines.append(f"{i}. {P[d]['label']}: {p:.2f}" + (f" - because: {reason}" if reason else ""))
    used = [finding_text(k, v, cfg) for k, v in m.items() if v is not None]
    unknown = [P[k]["label"] for k, v in m.items() if v is None]
    lines += ["", f"Findings used ({len(used)}): " + (", ".join(used) if used else "none")]
    if unknown:
        lines.append(f"Not documented: {', '.join(unknown)}")
    c = res["cost"]
    lines.append(f"Engine: {c['method']}; P(findings) = {res['p_evidence']:.2e}"
                 + (f"; model evaluations = {c['oracle_calls']:,}" if "oracle_calls" in c else ""))
    if truth is not None:
        top = res["ranked"][: cfg["engine"]["top_k"]]
        got = [P[d]["label"] for d in truth if truth[d]]
        hit = [P[d]["label"] for d in top if truth.get(d)]
        lines.append(f"Discharge diagnoses in scope: {', '.join(got) if got else 'none'}; "
                     f"in top {len(top)}: {', '.join(hit) if hit else 'none'}")
    lines.append("Research output from a model learned on MIMIC-IV; not for clinical use.")
    return "\n".join(lines)
