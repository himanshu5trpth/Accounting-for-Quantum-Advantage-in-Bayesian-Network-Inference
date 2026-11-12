"""W3-W5 and W7 of the quantum paper plan (tasks/plan_quantum_paper.md, amendment 1). Public and synthetic networks
only; runs in the cloud workspace.

  python scripts/benchmarks.py instances           # W3.2: 1,000 instances per network -> artifacts/quantum_paper/benchmark_instances.json (+ sha256)
  python scripts/benchmarks.py costs NETWORK       # W3.3: per-instance quantities (pruned and unpruned)
  python scripts/benchmarks.py w4                  # W4: the anti-correlation -> benchmark_costs.json, w4.png
  python scripts/benchmarks.py w5-pass1            # W5 networks + instances + structural predictors only (no probabilities)
  python scripts/benchmarks.py w5-pass2 PART PARTS # W5 costs (after the W7 predictions are committed)
"""
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from msbn import MSBN, from_pgmpy, instance, lw_quantities, nq, prune, qlw_cost, ve  # noqa: E402

OUT, SEED, REL, T_DEC = Path("artifacts/quantum_paper"), 2026, 0.1, 0.01
NETWORKS = ["asia", "cancer", "earthquake", "win95pts", "andes", "survey", "sachs", "child", "insurance", "water",
            "alarm", "hailfinder", "hepar2", "pigs"]
EXTRA = ["munin"]  # attempted last; reported only if exact inference is tractable
KAPPA = 11.685850819903646  # tuned IAE (frontier.json); kappa = 1 reported as the ideal bound


def load(name: str) -> MSBN:
    import warnings
    from pgmpy.utils import get_example_model
    warnings.filterwarnings("ignore")
    return from_pgmpy(get_example_model(name))


def instances():
    OUT.mkdir(parents=True, exist_ok=True)
    out = {}
    for k, name in enumerate(NETWORKS + EXTRA):
        bn = load(name)
        rng = np.random.default_rng(SEED + 1000 * k)
        rows = []
        for i in range(1000):
            ev, t, s = instance(bn, rng)
            rows.append({"i": i, "evidence": {str(a): int(b) for a, b in ev.items()}, "target": t, "state": s})
        out[name] = {"nodes": bn.n, "names_hash": hashlib.sha256(json.dumps(bn.names).encode()).hexdigest()[:16], "instances": rows}
        print(name, bn.n, flush=True)
    p = OUT / "benchmark_instances.json"
    p.write_text(json.dumps(out))
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    (OUT / "benchmark_instances.sha256").write_text(h + "  benchmark_instances.json\n")
    print(h)


def _prep_cx(bn: MSBN, nodes) -> int:
    return int(sum(2 ** (sum(nq(bn.card[p]) for p in bn.parents[j]) + b)
                   for j in nodes for b in range(nq(bn.card[j])) if sum(nq(bn.card[p]) for p in bn.parents[j]) + b >= 1))


def costs(name: str):
    from cost_model import mcx_noancilla_table
    from cost_model_ms import law
    table = mcx_noancilla_table()
    bn = load(name)
    inst = json.loads((OUT / "benchmark_instances.json").read_text())[name]["instances"]
    rows, t0 = [], time.perf_counter()
    for r in inst:
        ev = {int(a): b for a, b in r["evidence"].items()}
        t, s = r["target"], r["state"]
        rec = {"network": name, "i": r["i"], "target": t, "state": s, "network_nodes": bn.n}
        try:
            q = lw_quantities(bn, t, s, ev, pruned=True)
        except MemoryError as e:
            rec["status"] = f"intractable: {e}"
            rows.append(rec)
            continue
        rec.update(q)
        try:
            _, big_full, w_full = ve(bn, set(range(bn.n)), ev, [t], max_log10=8.0)
            rec["log10_max_factor_unpruned"], rec["width_unpruned"] = big_full, w_full
        except MemoryError:
            rec["log10_max_factor_unpruned"], rec["width_unpruned"] = math.nan, -1
        if not (0 < q["p"] < 1):
            rec["status"] = "p is 0 or 1"
            rows.append(rec)
            continue
        rel, keep = prune(bn, t, ev)
        n_q = sum(nq(bn.card[j]) for j in keep)
        L = sum(nq(bn.card[j]) for j in rel) + nq(bn.card[t])
        lw_ = law(_prep_cx(bn, keep), n_q, L, table)
        rec.update(qubits=n_q, literals=L, ratio_v1=lw_["ratio_v1"], ratio_v0=lw_["ratio_v0"])
        for kname, k in (("tuned", KAPPA), ("ideal", 1.0)):
            for mname, tt in (("rel", None), ("dec", T_DEC)):
                eps = REL if tt is None else REL * max(q["p"], tt) / q["p"]
                nlw = q["lw_samples"] * (REL / eps) ** 2
                nq_me = qlw_cost(q["a_E"], q["a_T"], k, eps)
                rec[f"R_{kname}_{mname}_v1"] = nq_me * lw_["ratio_v1"] / nlw
                rec[f"R_{kname}_{mname}_evals"] = nq_me / nlw
        rec["status"] = "ok"
        rows.append(rec)
    d = pd.DataFrame(rows)
    d.to_parquet(OUT / f"bench_{name}.parquet")
    print(f"{name}: {len(d)} instances, {int((d.status == 'ok').sum())} ok, {time.perf_counter() - t0:.0f}s", flush=True)




# ------------------------------------------------------------------------------------- W4
R_COL = "R_tuned_rel_v1"


def _frame(names, prefix="bench_") -> pd.DataFrame:
    d = pd.concat([pd.read_parquet(OUT / f"{prefix}{n}.parquet") for n in names if (OUT / f"{prefix}{n}.parquet").exists()], ignore_index=True)
    ok = d[d.status == "ok"].copy()
    floor = ok.lw_samples / np.maximum(ok.lw_samples, 1.0)  # LW cannot use fewer than one sample
    for c in [c for c in ok.columns if c.startswith("R_") and "_rel_" in c]:
        ok[c] = ok[c] * floor
    dec_lw = ok.lw_samples * (ok.p / np.maximum(ok.p, T_DEC)) ** 2
    dfl = dec_lw / np.maximum(dec_lw, 1.0)
    for c in [c for c in ok.columns if c.startswith("R_") and "_dec_" in c]:
        ok[c] = ok[c] * dfl
    ok["logR"] = np.log10(ok[R_COL])
    ok["cost"] = ok.log10_max_factor
    return d, ok


def _spearman(x, y) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(x, y).statistic)


def _net_boot(d: pd.DataFrame, stat, n: int = 1000, seed: int = SEED):
    nets = d.network.unique()
    groups = {k: g for k, g in d.groupby("network")}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        pick = rng.choice(nets, len(nets), replace=True)
        out.append(stat(pd.concat([groups[k] for k in pick], ignore_index=True)))
    out = np.array(out)
    out = out[np.isfinite(out)]
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


def _within_ranks(d: pd.DataFrame, cols) -> pd.DataFrame:
    return d.groupby("network")[cols].rank(pct=True)


def w4_stats(ok: pd.DataFrame) -> dict:
    res = {"instances": int(len(ok)), "networks": int(ok.network.nunique())}
    res["primary_pooled_spearman"] = {"rho": _spearman(ok.cost, ok.logR), "ci95_network_bootstrap": _net_boot(ok, lambda g: _spearman(g.cost, g.logR))}
    res["primary_pooled_spearman"]["pass"] = res["primary_pooled_spearman"]["rho"] < 0 and res["primary_pooled_spearman"]["ci95_network_bootstrap"][1] < 0

    def within(g):
        r = _within_ranks(g, ["cost", "logR"])
        return float(np.corrcoef(r.cost, r.logR)[0, 1]) if r.cost.std() > 0 else math.nan
    res["within_network_pooled_rank_correlation"] = {"r": within(ok), "ci95_network_bootstrap": _net_boot(ok, within)}

    def partial(g):
        from scipy.stats import rankdata
        x, y, z = rankdata(g.cost), rankdata(g.logR), rankdata(np.log10(g.network_nodes))
        Z = np.column_stack([np.ones(len(z)), z])
        rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
        ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
        return float(np.corrcoef(rx, ry)[0, 1])
    res["partial_spearman_given_network_size"] = {"r": partial(ok), "ci95_network_bootstrap": _net_boot(ok, partial)}
    med = ok.groupby("network").cost.transform("median")
    win = ok[ok[R_COL] < 1]
    below = (win.cost < med[win.index]).mean() if len(win) else math.nan
    at_or_below = (win.cost <= med[win.index]).mean() if len(win) else math.nan
    exc = win[win.cost >= med[win.index]]
    res["secondary_R_below_1"] = {"instances": int(len(win)), "share_cost_below_network_median": float(below),
                                  "share_cost_at_or_below_network_median": float(at_or_below), "pass": bool(below >= 0.9),
                                  "exceptions_by_network": exc.network.value_counts().to_dict(),
                                  "exceptions_examples": exc.sort_values("cost", ascending=False).head(10)[
                                      ["network", "i", "target", "p", "cost", "a_E", "D_T", R_COL]].to_dict("records")}
    per = {}
    for k, g in ok.groupby("network"):
        m = g.cost.median()
        w = g[g[R_COL] < 1]
        per[k] = {"n": int(len(g)), "nodes": int(g.network_nodes.iloc[0]),
                  "spearman": _spearman(g.cost, g.logR) if g.cost.nunique() > 1 else None,
                  "distinct_cost_values": int(g.cost.nunique()), "share_R_below_1": float((g[R_COL] < 1).mean()),
                  "R_below_1_share_cost_below_median": float((w.cost < m).mean()) if len(w) else None,
                  "median_log10_R": float(g.logR.median()), "median_cost_pruned": float(m),
                  "median_cost_unpruned": float(g.log10_max_factor_unpruned.median())}
    res["per_network"] = per
    gap = ok.log10_max_factor_unpruned - ok.cost
    res["mechanism_gap_unpruned_minus_pruned"] = {"spearman_with_logR": _spearman(gap, ok.logR),
                                                  "ci95_network_bootstrap": _net_boot(ok.assign(gap=gap), lambda g: _spearman(g.gap, g.logR))}
    res["secondary_metrics"] = {c: {"pooled_spearman": _spearman(ok.cost, np.log10(ok[c])), "share_below_1": float((ok[c] < 1).mean())}
                                for c in ok.columns if c.startswith("R_")}
    return res


def w4():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d, ok = _frame(NETWORKS)
    out = {"dropped": d[d.status != "ok"].groupby(["network", "status"]).size().reset_index().values.tolist(),
           "R": R_COL + " (tuned IAE, 10% relative, V1 gates, pruned network; LW floored at one sample)",
           "cost": "log10 largest intermediate factor, min-fill, pruned network"}
    out.update(w4_stats(ok))
    if (OUT / "bench_munin.parquet").exists():
        dm, okm = _frame(["munin"])
        out["munin"] = {"status": dm.status.value_counts().to_dict(), "ok": int(len(okm)),
                        "spearman": _spearman(okm.cost, okm.logR) if len(okm) > 10 else None}
    (OUT / "benchmark_costs.json").write_text(json.dumps(out, indent=1, default=float))
    nets = sorted(ok.network.unique(), key=lambda k: ok[ok.network == k].network_nodes.iloc[0])
    fig, axs = plt.subplots(4, 4, figsize=(12, 10.5), sharey=True)
    for ax, k in zip(axs.flat, nets):
        g = ok[ok.network == k]
        ax.scatter(g.cost, g.logR, s=5, alpha=0.35, color="#2a78d6", lw=0)
        ax.axhline(0, color="#1f1f1e", lw=0.8, ls=":")
        rho = out["per_network"][k]["spearman"]
        ax.set_title(f"{k} ({g.network_nodes.iloc[0]} nodes)  ρ = {rho:.2f}" if rho is not None else f"{k} ({g.network_nodes.iloc[0]} nodes)", fontsize=8.5, loc="left")
        ax.tick_params(labelsize=7)
    ax = axs.flat[len(nets)]
    ax.scatter(ok.cost, ok.logR, s=3, alpha=0.2, color="#1f1f1e", lw=0)
    ax.axhline(0, color="#1f1f1e", lw=0.8, ls=":")
    p = out["primary_pooled_spearman"]
    ax.set_title(f"all 14 pooled  ρ = {p['rho']:.2f} [{p['ci95_network_bootstrap'][0]:.2f}, {p['ci95_network_bootstrap'][1]:.2f}]", fontsize=8.5, loc="left")
    for a in axs.flat[len(nets) + 1:]:
        a.axis("off")
    fig.supxlabel("Exact-inference cost: log10 largest intermediate table (min-fill, pruned network)", fontsize=9)
    fig.supylabel("log10 R = quantum ÷ classical LW cost (tuned IAE, V1 gates; below 0 = quantum cheaper)", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "w4.png", dpi=150)
    print(json.dumps({k: v for k, v in out.items() if k not in ("per_network", "secondary_metrics")}, indent=1, default=float))
    for k, v in out["per_network"].items():
        print(k, v)


# ------------------------------------------------------------------------------------- W5 (+ W7 held-out set)
W5_WIDTHS, W5_C, W5_DEPTH, W5_NETS, W5_INST = (2, 4, 6, 8, 10, 12, 14, 16), (0.01, 0.1, 1.0), 6, 10, 100
W5_CAP = 7.0  # largest intermediate table, log10 entries: the operational limit of tractable exact inference here


def layered(w: int, c: float, rng) -> tuple[MSBN, int]:
    """Binary layered network: depth 6, width w, each node in layer l > 0 has min(4, w) parents in layer l - 1.
    The target (first node of layer 3) has its CPT shrunk toward its prior: row <- (1 - c) * mean row + c * row."""
    names, parents, cpts = [], [], []
    for l in range(W5_DEPTH):
        for i in range(w):
            j = l * w + i
            pa = () if l == 0 else tuple(sorted(rng.choice(range((l - 1) * w, l * w), size=min(4, w), replace=False).tolist()))
            p1 = rng.uniform(0.05, 0.95, 2 ** len(pa))
            names.append(f"L{l}_{i}")
            parents.append(pa)
            cpts.append(np.stack([1 - p1, p1], 1).reshape((2,) * len(pa) + (2,)))
    t = 3 * w
    m = cpts[t].reshape(-1, 2).mean(0)
    cpts[t] = (1 - c) * m + c * cpts[t]
    cpts[t] = cpts[t] / cpts[t].sum(-1, keepdims=True)
    return MSBN(names, [2] * len(names), parents, cpts), t


def _w5_networks():
    for wi, w in enumerate(W5_WIDTHS):
        for ci, c in enumerate(W5_C):
            for r in range(W5_NETS):
                rng = np.random.default_rng(SEED + 100000 + 1000 * wi + 100 * ci + r)
                bn, t = layered(w, c, rng)
                yield f"w{w}_c{c}_r{r}", w, c, bn, t, rng


def w5_pass1():
    """Instances (target fixed at the designed node; evidence by the W3.2 protocol among the other nodes) and the
    structural predictors only (no probability computed): pruned and unpruned min-fill cost, evidence share removed,
    full-network treewidth bound."""
    from msbn import elim_cost
    rows = []
    t0 = time.perf_counter()
    for name, w, c, bn, t, rng in _w5_networks():
        full_w = elim_cost(bn, set(range(bn.n)), {}, [], cap=12)
        for i in range(W5_INST):
            x = __import__("msbn").forward_sample(bn, rng)
            L = int(rng.integers(1, max(1, min(10, bn.n // 4)) + 1))
            others = rng.permutation([j for j in range(bn.n) if j != t])
            ev = {int(j): x[j] for j in others[:L]}
            s = int(rng.integers(2))
            rel, keep = prune(bn, t, ev)
            pc = elim_cost(bn, keep, rel, [t], cap=9)
            uc = elim_cost(bn, set(range(bn.n)), ev, [t], cap=9)
            rows.append({"network": name, "width_param": w, "coupling": c, "i": i, "target": t, "state": s,
                         "evidence": json.dumps({str(a): b for a, b in ev.items()}), "network_nodes": bn.n,
                         "full_width": full_w[1], "full_log10_max_factor": full_w[0], "cost": pc[0], "pruned_width": pc[1],
                         "log10_max_factor_unpruned": uc[0], "share_evidence_removed": 1 - len(rel) / len(ev)})
        print(name, f"{time.perf_counter() - t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_parquet(OUT / "w5_pass1.parquet")


def w5_pass2(part: str = "0", parts: str = "1"):
    from cost_model import mcx_noancilla_table
    from cost_model_ms import law
    table = mcx_noancilla_table()
    p1 = pd.read_parquet(OUT / "w5_pass1.parquet")
    nets = {name: (bn, t) for name, w, c, bn, t, rng in _w5_networks()}
    names = sorted(nets)[int(part)::int(parts)]
    rows, t0 = [], time.perf_counter()
    for name in names:
        bn, _ = nets[name]
        for r in p1[p1.network == name].itertuples():
            ev = {int(a): b for a, b in json.loads(r.evidence).items()}
            rec = {"network": name, "i": r.i}
            if not np.isfinite(r.cost) or r.cost > W5_CAP:
                rec["status"] = f"beyond this study's exact-inference limit (largest factor > 10^{W5_CAP})"
                rows.append(rec)
                continue
            q = lw_quantities(bn, r.target, r.state, ev, pruned=True, max_log10=W5_CAP + 1e-9)
            rec.update(q)
            if not (0 < q["p"] < 1):
                rec["status"] = "p is 0 or 1"
                rows.append(rec)
                continue
            rel, keep = prune(bn, r.target, ev)
            lw_ = law(_prep_cx(bn, keep), len(keep), len(rel) + 1, table)
            rec["ratio_v1"] = lw_["ratio_v1"]
            for kname, k in (("tuned", KAPPA), ("ideal", 1.0)):
                for mname, tt in (("rel", None), ("dec", T_DEC)):
                    eps = REL if tt is None else REL * max(q["p"], tt) / q["p"]
                    nlw = q["lw_samples"] * (REL / eps) ** 2
                    nqe = qlw_cost(q["a_E"], q["a_T"], k, eps)
                    rec[f"R_{kname}_{mname}_v1"] = nqe * lw_["ratio_v1"] / nlw
                    rec[f"R_{kname}_{mname}_evals"] = nqe / nlw
            rec["status"] = "ok"
            rows.append(rec)
        print(name, f"{time.perf_counter() - t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_parquet(OUT / f"w5_pass2_{part}.parquet")  # after every network: survives a restart


def w5():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    p1 = pd.read_parquet(OUT / "w5_pass1.parquet")
    r = pd.concat([pd.read_parquet(q) for q in sorted(OUT.glob("w5_pass2_*.parquet"))], ignore_index=True)
    status = r.status.value_counts().to_dict()
    r = r[r.status == "ok"].copy()
    r[R_COL] = r[R_COL] * (r.lw_samples / np.maximum(r.lw_samples, 1.0))
    d = p1.drop(columns=["target", "state", "evidence"]).merge(r[["network", "i", R_COL, "p", "a_E", "D_T", "log10_max_factor"]], on=["network", "i"])
    assert np.allclose(d.cost, d.log10_max_factor)  # symbolic and numeric elimination agree
    d["logR"] = np.log10(d[R_COL])
    out = {"status": status, "instances": int(len(d)), "networks": int(d.network.nunique()),
           "pooled_spearman": {"rho": _spearman(d.cost, d.logR), "ci95_network_bootstrap": _net_boot(d, lambda g: _spearman(g.cost, g.logR))}}
    rk = _within_ranks(d, ["cost", "logR"])
    out["within_network_rank_correlation"] = float(np.corrcoef(rk.cost, rk.logR)[0, 1])
    out["logR_vs_log_p_spearman"] = _spearman(np.log10(d.p), d.logR)
    by = {}
    for (w, c), g in d.groupby(["width_param", "coupling"]):
        by[f"w{w}|c{c}"] = {"n": int(len(g)), "median_full_width": float(g.full_width.median()), "median_cost": float(g.cost.median()),
                            "spearman_cost_logR": _spearman(g.cost, g.logR) if g.cost.nunique() > 1 else None,
                            "share_R_below_1": float((g[R_COL] < 1).mean()), "median_log10_R": float(g.logR.median()),
                            "median_p": float(g.p.median())}
    out["by_width_and_coupling"] = by
    out["by_width"] = {int(w): {"spearman_cost_logR": _spearman(g.cost, g.logR), "share_R_below_1": float((g[R_COL] < 1).mean()),
                                "median_full_width": float(g.full_width.median())} for w, g in d.groupby("width_param")}
    (OUT / "treewidth_sweep.json").write_text(json.dumps(out, indent=1, default=float))
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, c in zip(axs, W5_C):
        g = d[d.coupling == c]
        sc = ax.scatter(g.cost, g.logR, c=g.full_width, cmap="Blues", vmin=0, vmax=d.full_width.max(), s=5, alpha=0.6, lw=0)
        ax.axhline(0, color="#1f1f1e", lw=0.8, ls=":")
        ax.set_title(f"coupling c = {c}: ρ = {_spearman(g.cost, g.logR):.2f}", loc="left", fontsize=9.5)
    fig.supxlabel("Exact-inference cost: log10 largest intermediate table (min-fill, pruned network)", fontsize=9)
    axs[0].set_ylabel("log10 R (tuned IAE, V1 gates; below 0 = quantum cheaper)")
    fig.colorbar(sc, ax=axs, label="treewidth bound (min-fill width, full network)")
    fig.savefig(OUT / "treewidth_sweep.png", dpi=150, bbox_inches="tight")
    print(json.dumps({k: v for k, v in out.items() if k != "by_width_and_coupling"}, indent=1, default=float))
    for k, v in by.items():
        print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()})


# ------------------------------------------------------------------------------------- Q1: factorial W5 (round 3)
W5F_B = (0.3, 0.03, 0.003, 0.0003)


def layered_mag(w: int, b: float, rng) -> tuple[MSBN, int]:
    """W5 network with the target's state-1 probability per parent row = clip(b exp(N(0, 0.5^2)), 1e-6, 0.95)."""
    bn, t = layered(w, 1.0, rng)
    rows = int(np.prod([2] * len(bn.parents[t])))
    p1 = np.clip(b * np.exp(rng.normal(0, 0.5, rows)), 1e-6, 0.95)
    bn.cpts[t] = np.stack([1 - p1, p1], 1).reshape((2,) * len(bn.parents[t]) + (2,))
    return bn, t


def w5f(part: str = "0", parts: str = "1"):
    from cost_model import mcx_noancilla_table
    from cost_model_ms import law
    from msbn import elim_cost, forward_sample
    table = mcx_noancilla_table()
    cells = [(w, b, r) for w in W5_WIDTHS for b in W5F_B for r in range(W5_NETS)]
    prev = OUT / f"w5f_{part}.parquet"
    rows = pd.read_parquet(prev).to_dict("records") if prev.exists() else []  # resume after a restart
    done = {x["network"] for x in rows}
    t0 = time.perf_counter()
    for idx, (w, b, r) in enumerate(cells):
        if idx % int(parts) != int(part) or f"w{w}_b{b}_r{r}" in done:
            continue
        rng = np.random.default_rng(SEED + 200000 + 1000 * W5_WIDTHS.index(w) + 100 * W5F_B.index(b) + r)
        bn, t = layered_mag(w, b, rng)
        name = f"w{w}_b{b}_r{r}"
        full_w = elim_cost(bn, set(range(bn.n)), {}, [], cap=12)[1]
        for i in range(W5_INST):
            x = forward_sample(bn, rng)
            L = int(rng.integers(1, max(1, min(10, bn.n // 4)) + 1))
            others = rng.permutation([j for j in range(bn.n) if j != t])
            ev = {int(j): x[j] for j in others[:L]}
            rel, keep = prune(bn, t, ev)
            cost, pw = elim_cost(bn, keep, rel, [t], cap=9)
            rec = {"network": name, "width_param": w, "base_rate": b, "i": i, "full_width": full_w, "cost": cost,
                   "network_nodes": bn.n, "share_evidence_removed": 1 - len(rel) / len(ev)}
            if not np.isfinite(cost) or cost > W5_CAP:
                rec["status"] = "beyond exact-inference limit"
                rows.append(rec)
                continue
            q = lw_quantities(bn, t, 1, ev, pruned=True, max_log10=W5_CAP + 1e-9)
            rec.update(q)
            if not (0 < q["p"] < 1):
                rec["status"] = "p is 0 or 1"
                rows.append(rec)
                continue
            lw_ = law(_prep_cx(bn, keep), len(keep), len(rel) + 1, table)
            rec["ratio_v1"] = lw_["ratio_v1"]
            nqe = qlw_cost(q["a_E"], q["a_T"], 1.0)
            rec["qlw_unit"] = nqe  # kappa = 1; any kappa multiplies it
            rec[R_COL] = KAPPA * nqe * lw_["ratio_v1"] / max(q["lw_samples"], 1.0)
            rec["status"] = "ok"
            rows.append(rec)
        print(name, f"{time.perf_counter() - t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_parquet(OUT / f"w5f_{part}.parquet")


def w5f_analyse():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import rankdata
    d = pd.concat([pd.read_parquet(q) for q in sorted(OUT.glob("w5f_*.parquet"))], ignore_index=True)
    status = d.status.value_counts().to_dict()
    d = d[d.status == "ok"].copy()
    d["logR"] = np.log10(d[R_COL])
    d["logp"] = np.log10(d.p)
    out = {"status": status, "instances": int(len(d)), "networks": int(d.network.nunique()), "cells": {}}
    for (w, b), g in d.groupby(["width_param", "base_rate"]):
        out["cells"][f"w{w}|b{b}"] = {"n": int(len(g)), "median_full_width": float(g.full_width.median()), "median_p": float(g.p.median()),
                                      "share_R_below_1": float((g[R_COL] < 1).mean()), "median_log10_R": float(g.logR.median())}
    # (i) share R < 1 rises as b falls, at every width
    mono = {}
    for w in W5_WIDTHS:
        sh = [out["cells"].get(f"w{w}|b{b}", {}).get("share_R_below_1", np.nan) for b in W5F_B]
        mono[w] = {"shares_b_high_to_low": sh, "non_decreasing": bool(all(sh[k + 1] >= sh[k] for k in range(len(sh) - 1)))}
    # (ii) within each magnitude level, |Spearman(log R, cost)| < 0.2
    within = {str(b): _spearman(g.cost, g.logR) for b, g in d.groupby("base_rate")}
    # (iii) partial Spearman of log R with log p given cost
    x, y, z = rankdata(d.logp), rankdata(d.logR), rankdata(d.cost)
    Z = np.column_stack([np.ones(len(z)), z])
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    partial = float(np.corrcoef(rx, ry)[0, 1])
    out["tests"] = {"i_share_rises_as_b_falls_every_width": {"by_width": mono, "pass": all(v["non_decreasing"] for v in mono.values())},
                    "ii_within_magnitude_spearman_cost": {"by_b": within, "pass": all(abs(v) < 0.2 for v in within.values())},
                    "iii_partial_spearman_logR_logp_given_cost": {"value": partial,
                                                                  "ci95_network_bootstrap": _net_boot(d, lambda g: float(np.corrcoef(
                                                                      *(lambda X, Y, Zc: (X - Zc @ np.linalg.lstsq(Zc, X, rcond=None)[0], Y - Zc @ np.linalg.lstsq(Zc, Y, rcond=None)[0]))(
                                                                          rankdata(g.logp), rankdata(g.logR), np.column_stack([np.ones(len(g)), rankdata(g.cost)])))[0, 1]), n=300),
                                                                  "pass": partial >= 0.7}}
    out["thesis_supported"] = all(v["pass"] for v in out["tests"].values())
    (OUT / "w5_factorial.json").write_text(json.dumps(out, indent=1, default=float))
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.3))
    cols = {0.3: "#c9d7f0", 0.03: "#86a8e0", 0.003: "#3f74c9", 0.0003: "#173f8a"}
    for b in W5F_B:
        sh = [out["cells"].get(f"w{w}|b{b}", {}).get("share_R_below_1", np.nan) for w in W5_WIDTHS]
        tw = [out["cells"].get(f"w{w}|b{b}", {}).get("median_full_width", np.nan) for w in W5_WIDTHS]
        axs[0].plot(tw, sh, marker="o", ms=4, lw=2, color=cols[b], label=f"base rate {b}")
    axs[0].set_xlabel("treewidth bound (min-fill width, median per width setting)")
    axs[0].set_ylabel("share of instances where quantum is cheaper (R < 1)")
    axs[0].set_ylim(-0.02, 1.02)
    axs[0].legend(frameon=False, fontsize=8)
    axs[0].set_title("Magnitude sets the outcome; treewidth does not", loc="left", fontsize=10)
    for b in W5F_B:
        g = d[d.base_rate == b]
        axs[1].scatter(g.cost, g.logR, s=3, alpha=0.25, color=cols[b], lw=0)
    axs[1].axhline(0, color="#1f1f1e", lw=0.8, ls=":")
    axs[1].set_xlabel("log10 largest table, pruned network (min-fill)")
    axs[1].set_ylabel("log10 R (below 0 = quantum cheaper)")
    axs[1].set_title("R against exact-inference cost, coloured by base rate", loc="left", fontsize=10)
    for a_ in axs:
        a_.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "w5_factorial.png", dpi=150)
    print(json.dumps({k: v for k, v in out.items() if k != "cells"}, indent=1, default=float))
    for k, v in out["cells"].items():
        print(k, {a: round(x, 4) if isinstance(x, float) else x for a, x in v.items()})


# ------------------------------------------------------------------------------------- W7
W7_X = ["cost", "log10_max_factor_unpruned", "share_evidence_removed", "log10_nodes"]


def _w7_features(d: pd.DataFrame) -> np.ndarray:
    X = d[W7_X[:3]].to_numpy(float)
    return np.column_stack([X, np.log10(d.network_nodes.to_numpy(float))])


def w7_predict():
    """Fit sign(R - 1) on the 14 benchmark networks from structural predictors only (amendment 1, item 5), predict
    every W5 instance from its pass-1 predictors, and write the predictions (and their sha256) before pass 2."""
    import hashlib
    from sklearn.linear_model import LogisticRegression
    _, ok = _frame(NETWORKS)
    ok["share_evidence_removed"] = 1 - ok.evidence_used / ok.evidence_given
    ok = ok[np.isfinite(ok.log10_max_factor_unpruned)]
    y = (ok[R_COL] < 1).astype(int).to_numpy()
    m = LogisticRegression(max_iter=1000, class_weight=None).fit(_w7_features(ok), y)
    p1 = pd.read_parquet(OUT / "w5_pass1.parquet")
    fin = np.isfinite(p1.cost) & np.isfinite(p1.log10_max_factor_unpruned)
    pred = np.full(len(p1), -1)
    pred[fin.to_numpy()] = m.predict(_w7_features(p1[fin]))
    out = p1[["network", "i"]].copy()
    out["predicted_R_below_1"] = pred  # -1: predictor not computable (intractable)
    path = OUT / "w7_predictions.parquet"
    out.to_parquet(path)
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT / "w7_predictions.sha256").write_text(h + "  w7_predictions.parquet\n")
    model = {"features": W7_X, "coef": m.coef_[0].tolist(), "intercept": float(m.intercept_[0]), "train_instances": int(len(ok)),
             "train_networks": int(ok.network.nunique()), "train_accuracy": float((m.predict(_w7_features(ok)) == y).mean()),
             "train_share_R_below_1": float(y.mean()), "heldout_predicted_share_R_below_1": float((pred[pred >= 0]).mean()),
             "heldout_not_predictable": int((pred < 0).sum()), "sha256": h}
    (OUT / "w7_model.json").write_text(json.dumps(model, indent=1))
    print(json.dumps(model, indent=1))


def w7_score():
    p = pd.read_parquet(OUT / "w7_predictions.parquet")
    r = pd.concat([pd.read_parquet(q) for q in sorted(OUT.glob("w5_pass2_*.parquet"))], ignore_index=True)
    r = r[r.status == "ok"].copy()
    floor = r.lw_samples / np.maximum(r.lw_samples, 1.0)
    r[R_COL] = r[R_COL] * floor
    d = p.merge(r[["network", "i", R_COL]], on=["network", "i"])
    d = d[d.predicted_R_below_1 >= 0]
    d["correct"] = (d.predicted_R_below_1 == (d[R_COL] < 1).astype(int)).astype(float)
    acc = float(d.correct.mean())
    ci = _net_boot(d, lambda g: g.correct.mean())
    base = float(max((d[R_COL] < 1).mean(), 1 - (d[R_COL] < 1).mean()))
    out = {"heldout_instances": int(len(d)), "heldout_networks": int(d.network.nunique()), "accuracy": acc, "ci95_network_bootstrap": ci,
           "majority_class_accuracy": base, "pass": ci[0] > 0.8,
           "by_width": {int(w): float(g.correct.mean()) for w, g in d.assign(w=d.network.str.extract(r"w(\d+)_")[0].astype(int)).groupby("w")},
           "by_coupling": {c: float(g.correct.mean()) for c, g in d.assign(c=d.network.str.extract(r"_c([0-9.]+)_")[0]).groupby("c")}}
    (OUT / "w7_score.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    {"instances": instances, "costs": costs, "w4": w4, "w5-pass1": w5_pass1, "w5-pass2": w5_pass2,
     "w5": w5, "w5f": w5f, "w5f-analyse": w5f_analyse, "w7-predict": w7_predict, "w7-score": w7_score}[cmd](*args)
