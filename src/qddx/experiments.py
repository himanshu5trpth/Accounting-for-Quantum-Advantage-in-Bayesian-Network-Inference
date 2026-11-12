"""Post-freeze analyses (git tag heldout-v1). None of them changes a default; all are reported as robustness or
secondary analyses.

  sensitivity     one-at-a-time sweeps of hedge membership, graded-lab width and max parents (held-out notes)
  followup-curve  agent performance after 0..3 engine-chosen follow-ups (cached answers, no LLM), and whether the
                  engine's question is more informative than a random eligible one (expected and realised entropy)
  temporal        knowledge base trained on early-era patients, tested on later eras (structured data and notes)
  resources       logical circuit resources vs P(findings) and vs number of propositions
  agent-models    the held-out agent sample read by other local LLMs (general and medical), paired comparisons
"""
from __future__ import annotations

import copy
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import kb
from . import notes as N
from .agent import Agent, FakeClient, LLMError, OllamaClient, assert_local, ollama_info
from .bn import BN, Struct
from .engine import dx_names
from .evaluate import bootstrap, with_ci
from .mimicnotes import Posteriors, _crisp, _evaluate, _stream, structured_of
from .quantum import conj_oracle, iae_emulated, iae_relative, problem_circuit, resources, state_prep

log = logging.getLogger("qddx")


def _slug(model: str) -> str:
    return model.replace(":", "_").replace("/", "_")


def _with(cfg: dict, **changes) -> dict:
    """Copy of cfg with dotted-path overrides, e.g. _with(cfg, **{"notes.hedge_membership": 0.8})."""
    c = copy.deepcopy(cfg)
    for path, val in changes.items():
        d = c
        *parts, last = path.split(".")
        for p in parts:
            d = d[p]
        d[last] = val
    return c


def _eligible(art: Path) -> pd.DataFrame:
    p = art / "mimic_notes" / "eligible.parquet"
    if not p.exists():
        raise SystemExit("run `qddx mimic-notes` first (eligible held-out notes are its output)")
    return pd.read_parquet(p)


def _truths(cfg: dict, df: pd.DataFrame) -> dict[str, np.ndarray]:
    dx = dx_names(cfg)
    return {"note": df[[f"note_{d}" for d in dx]].to_numpy(int), "icd": df[[f"kb_{d}" for d in dx]].to_numpy(int)}


def _stored_memberships(cfg: dict, df: pd.DataFrame) -> list[dict]:
    ev = kb.evidence(cfg)
    cols = df[[f"m_{k}" for k in ev]].to_numpy(float)
    return [{k: (None if np.isnan(v) else float(v)) for k, v in zip(ev, row)} for row in cols]


# ------------------------------------------------------------------------------------- agent replay
class Offline:
    """Client for replaying cached agent answers: any uncached call fails like a server error and is counted."""

    def __init__(self, model: str):
        self.model, self.calls = model, 0

    def chat(self, *a, **k):
        self.calls += 1
        raise LLMError("offline replay: answer not in cache")


def sample_texts(cfg: dict, art: Path) -> dict[str, str]:
    """Evidence text of the frozen agent sample (artifacts/mimic_notes/agent_sample.txt), in sample order."""
    p = art / "mimic_notes" / "agent_sample.txt"
    if not p.exists():
        raise SystemExit("run `qddx mimic-notes` first (it freezes the agent sample)")
    ids = p.read_text().split()
    want, out = set(ids), {}
    for r in _stream(Path(cfg["paths"]["mimic_notes"])):
        if r["note_id"] in want:
            out[r["note_id"]] = N.evidence_text(N.deidentify(r["text"])[0], cfg)
            if len(out) == len(want):
                break
    missing = want - set(out)
    if missing:
        raise SystemExit(f"{len(missing)} sampled notes not found in {cfg['paths']['mimic_notes']}")
    return {i: out[i] for i in ids}


def run_sample(cfg: dict, bn: BN, limits: dict, texts: dict[str, str], client, cache: Path, max_steps: int | None = None,
               protocol: str = "tools") -> dict:
    c = cfg if max_steps is None else _with(cfg, **{"agent.max_steps": max_steps})
    ag = Agent(client, c, cache, protocol=protocol)
    out, t0 = {}, time.perf_counter()
    for i, (n, t) in enumerate(texts.items(), 1):
        out[n] = ag.run(t, bn, limits)
        if not isinstance(client, Offline) and (i % 25 == 0 or i == len(texts)):
            el = time.perf_counter() - t0
            log.info("agent %s: %d/%d notes, %.1fs/note, ~%.0f min left", getattr(client, "model", "?"), i, len(texts),
                     el / i, el / i * (len(texts) - i) / 60)
    return out


def replay_reference(cfg: dict, kbdir: Path, art: Path, texts: dict, max_steps: int | None = None) -> tuple[dict, int]:
    ref = cfg["experiments"]["reference_model"]
    bn, limits = kb.load(cfg, kbdir)
    client = Offline(ref)
    res = run_sample(cfg, bn, limits, texts, client, art / "agent_cache" / _slug(ref), max_steps)
    return res, client.calls


def loop_stats(res: dict) -> dict:
    tr = [t for r in res.values() for t in r["trace"]]
    return {"notes": len(res), "follow_ups": len(tr), "answered": sum(t["status"] != "unknown" for t in tr),
            "changed_membership": sum(bool(t.get("changed")) for t in tr), "searches": sum(len(t["queries"]) for t in tr),
            "grounding_rejected": sum(len(r["rejected"]) for r in res.values()),
            "followup_rejected": sum(bool(t["rejected"]) for t in tr),
            "llm_errors": sum(len(r["errors"]) for r in res.values()),
            "grounding_fallbacks": sum(any(e.startswith("ground") for e in r["errors"]) for r in res.values())}


# ------------------------------------------------------------------------------------- sensitivity
def raw_findings(cfg: dict, art: Path, df: pd.DataFrame) -> dict[str, dict]:
    """Keyword-extractor findings (status, lab value) for every eligible note; cached in findings.parquet."""
    p = art / "mimic_notes" / "findings.parquet"
    ev = [k for k in kb.evidence(cfg) if cfg["propositions"][k]["kind"] != "demographic"]  # demographics: admission record
    if not p.exists():
        want, rows = set(df.note_id), []
        t0 = time.perf_counter()
        for r in _stream(Path(cfg["paths"]["mimic_notes"])):
            if r["note_id"] not in want:
                continue
            f = N.baseline_findings(N.evidence_text(N.deidentify(r["text"])[0], cfg), cfg)
            rows.append({"note_id": r["note_id"], **{f"s_{k}": f[k]["status"] for k in ev},
                         **{f"v_{k}": f[k]["value"] for k in ev}})
            if len(rows) % 10000 == 0:
                log.info("findings: %d/%d notes (%.0fs)", len(rows), len(want), time.perf_counter() - t0)
        pd.DataFrame(rows).to_parquet(p)
    t = pd.read_parquet(p).set_index("note_id").loc[df.note_id]
    out = {}
    for n, row in t.iterrows():
        out[n] = {k: {"status": row[f"s_{k}"], "value": None if pd.isna(row[f"v_{k}"]) else float(row[f"v_{k}"])} for k in ev}
    return out


def _learn(cfg: dict, kbdir: Path, max_parents: int) -> BN:
    if max_parents == cfg["knowledge"]["max_parents"]:
        return kb.load(cfg, kbdir)[0]
    df = pd.read_parquet(kbdir / "cohort.parquet")
    tr = kb.split_mask(df.subject_id.to_numpy(), cfg["knowledge"]["split"][0], cfg["seed"])
    return kb.learn_bn(cfg, df[tr], max_parents=max_parents)


def sensitivity(cfg: dict, kbdir: Path, art: Path) -> dict:
    x, dx = cfg["experiments"], dx_names(cfg)
    df = _eligible(art).reset_index(drop=True)
    _, limits = kb.load(cfg, kbdir)
    F = raw_findings(cfg, art, df)
    SD = {r["note_id"]: structured_of(cfg, r) for _, r in df.iterrows()}
    M0 = [N.memberships(F[n], cfg, limits, structured=SD[n]) for n in df.note_id]
    stored = _stored_memberships(cfg, df)
    same = all(a == b or (a is not None and b is not None and abs(a - b) < 1e-9) for m, s in zip(M0, stored) for a, b in
               ((m[k], s[k]) for k in m))
    if not same:
        raise SystemExit("re-extracted default memberships differ from the frozen eligible.parquet")
    texts = sample_texts(cfg, art)
    A, misses = replay_reference(cfg, kbdir, art, texts)
    sub = df.set_index("note_id").loc[list(texts)].reset_index()
    h0, w0, p0 = cfg["notes"]["hedge_membership"], cfg["knowledge"]["fuzzy_width"], cfg["knowledge"]["max_parents"]
    settings = [("default", {}, p0)]
    settings += [(f"hedge={h:g}", {"notes.hedge_membership": h}, p0) for h in x["hedge_grid"] if h != h0]
    settings += [(f"width={w:g}", {"knowledge.fuzzy_width": w}, p0) for w in x["width_grid"] if w != w0]
    settings += [(f"max_parents={p}", {}, p) for p in x["parents_grid"] if p != p0]
    bns = {p: _learn(cfg, kbdir, p) for p in {s[2] for s in settings}}
    P_all, P_agent, edges = {}, {}, {}
    for name, ch, p in settings:
        c = _with(cfg, **ch)
        post = Posteriors(bns[p], dx)
        P_all[name] = np.array([post(N.memberships(F[n], c, limits, structured=SD[n])) for n in df.note_id])
        P_agent[name] = np.array([post(N.memberships(A[n]["findings"], c, limits, structured=SD[n])) for n in sub.note_id])
        edges[name] = sum(len(pa) for pa in bns[p].struct.parents)
        log.info("sensitivity: %s done", name)
    post0 = Posteriors(bns[p0], dx)
    P_all["crisp"] = np.array([post0(_crisp(m)) for m in M0])
    P_agent["crisp"] = np.array([post0(_crisp(N.memberships(A[n]["findings"], cfg, limits, structured=SD[n]))) for n in sub.note_id])
    pairs = [(v, "default") for v in P_all if v != "default"]
    B = int(x["bootstrap"])
    Y, Ys = _truths(cfg, df), _truths(cfg, sub)
    return {"settings": [s[0] for s in settings], "edges": edges, "replay_cache_misses": misses,
            "extractor": {t: _evaluate(cfg, P_all, Y[t], df.subject_id.to_numpy(), pairs, B) for t in Y},
            "agent": {t: _evaluate(cfg, P_agent, Ys[t], sub.subject_id.to_numpy(), pairs, B) for t in Ys}}


# ------------------------------------------------------------------------------------- follow-up curve
def followup_curve(cfg: dict, kbdir: Path, art: Path) -> dict:
    x, dx = cfg["experiments"], dx_names(cfg)
    bn, limits = kb.load(cfg, kbdir)
    texts = sample_texts(cfg, art)
    sub = _eligible(art).set_index("note_id").loc[list(texts)].reset_index()
    post = Posteriors(bn, dx, cfg["engine"].get("calibration"))
    SD = {r["note_id"]: structured_of(cfg, r) for _, r in sub.iterrows()}
    P, per_step, misses = {}, {}, {}
    runs = {}
    for s in x["followup_steps"]:
        runs[s], misses[s] = replay_reference(cfg, kbdir, art, texts, max_steps=s)
        P[f"step {s}"] = np.array([post(N.memberships(runs[s][n]["findings"], cfg, limits, structured=SD[n])) for n in sub.note_id])
        st = loop_stats(runs[s])
        per_step[s] = {"follow_ups": st["follow_ups"], "changed_membership": st["changed_membership"]}
    Y = _truths(cfg, sub)
    g = sub.subject_id.to_numpy()
    first = f"step {x['followup_steps'][0]}"
    res = {"steps": x["followup_steps"], "per_step": per_step, "replay_calls": misses,
           "metrics": {t: _evaluate(cfg, P, Y[t], g, [(v, first) for v in P if v != first], int(x["bootstrap"])) for t in Y}}
    # reproduction check against the frozen mimic-notes run (grounding only = step 0, full loop = configured max)
    mn = art / "results" / "mimic-notes.json"
    if mn.exists():
        ref = json.loads(mn.read_text()).get("agent_sample", {}).get("note", {}).get("variants", {})
        last = f"step {cfg['agent']['max_steps']}"
        got = {v: res["metrics"]["note"]["variants"][v]["macro_auroc"]["value"] for v in (first, last) if v in res["metrics"]["note"]["variants"]}
        want = {first: ref.get("agent_ground", {}).get("macro_auroc", {}).get("value"),
                last: ref.get("agent", {}).get("macro_auroc", {}).get("value")}
        res["reproduces_mimic_notes"] = {v: {"replay": got.get(v), "frozen": want[v],
                                             "match": got.get(v) is not None and want[v] is not None and abs(got[v] - want[v]) < 1e-9}
                                         for v in want}
    # was the engine's question more informative than a random eligible one? (exact; no LLM)
    full = runs[max(x["followup_steps"])]
    q = [(sub.subject_id[sub.note_id == n].iloc[0], t) for n, r in full.items() for t in r["trace"] if t.get("candidate_gains")]
    if q:
        groups = np.array([s for s, _ in q])
        eng = np.array([t["info_gain"] for _, t in q])
        rnd = np.array([np.mean(list(t["candidate_gains"].values())) for _, t in q])
        real = np.array([t["entropy_before"] - t["entropy_after"] for _, t in q])
        changed = np.array([bool(t["changed"]) for _, t in q])

        def stats(w):
            return {"engine_expected_gain": float(np.average(eng, weights=w)), "random_expected_gain": float(np.average(rnd, weights=w)),
                    "engine_minus_random": float(np.average(eng - rnd, weights=w)), "realised_gain": float(np.average(real, weights=w))}
        res["questions"] = {"n": int(len(q)), "engine_better_than_random_mean": float((eng > rnd + 1e-12).mean()),
                            "realised_gain_when_changed": float(real[changed].mean()) if changed.any() else None,
                            "realised_gain_when_unchanged": float(real[~changed].mean()) if (~changed).any() else None,
                            "changed": int(changed.sum()),
                            **with_ci(stats(np.ones(len(q))), bootstrap(groups, int(x["bootstrap"]), cfg["seed"], stats))}
    return res


# ------------------------------------------------------------------------------------- temporal shift
def temporal(cfg: dict, kbdir: Path, art: Path) -> dict:
    x, dx = cfg["experiments"], dx_names(cfg)
    pat = pd.read_csv(kb.find_table(Path(cfg["paths"]["mimic_root"]), "patients"), usecols=["subject_id", "anchor_year_group"])
    era = dict(zip(pat.subject_id.astype(int), pat.anchor_year_group))
    df = pd.read_parquet(kbdir / "cohort.parquet")
    df["era"] = df.subject_id.map(era)
    tr = kb.split_mask(df.subject_id.to_numpy(), cfg["knowledge"]["split"][0], cfg["seed"])
    test_subjects = set(df.subject_id[~tr])
    early = df[tr & df.era.isin(x["train_eras"])]
    assert not (set(early.subject_id) & test_subjects)
    models = {"all eras (main KB)": kb.load(cfg, kbdir)[0], f"early ({', '.join(x['train_eras'])})": kb.learn_bn(cfg, early)}
    notes = _eligible(art)
    notes["era"] = notes.subject_id.map(era)
    out = {"train_rows": {"early": int(len(early))}, "eras": {}}
    for te in x["test_eras"]:
        matched = df[tr & (df.era == te)]
        assert not (set(matched.subject_id) & test_subjects)
        ms = dict(models)
        ms[f"same era ({te})"] = kb.learn_bn(cfg, matched)
        rows = df[~tr & (df.era == te)].reset_index(drop=True)
        nt = notes[notes.era == te].reset_index(drop=True)
        r = {"train_rows_same_era": int(len(matched)), "test_admissions": int(len(rows)), "test_notes": int(len(nt))}
        early_name = list(models)[1]
        pairs = [(early_name, list(ms)[0]), (early_name, list(ms)[2])]
        B = int(x["bootstrap"])
        if len(rows):
            P = {m: kb.posteriors_by_pattern(cfg, b, rows)[0] for m, b in ms.items()}
            r["structured"] = _evaluate(cfg, P, rows[dx].to_numpy(int), rows.subject_id.to_numpy(), pairs, B)
        if len(nt):
            M = _stored_memberships(cfg, nt)
            P = {m: np.array([pp(mm) for mm in M]) for m, b in ms.items() for pp in [Posteriors(b, dx)]}
            Y = _truths(cfg, nt)
            r["notes"] = {t: _evaluate(cfg, P, Y[t], nt.subject_id.to_numpy(), pairs, B) for t in Y}
        out["eras"][te] = r
        log.info("temporal: %s done", te)
    return out


# ------------------------------------------------------------------------------------- quantum resources
def _grover(circ):
    from qiskit_algorithms import EstimationProblem
    return EstimationProblem(circ, objective_qubits=[circ.num_qubits - 1]).grover_operator


def quantum_resources(cfg: dict, kbdir: Path, art: Path) -> dict:
    """Logical resources (all-to-all, rz/sx/x/cx, no error correction) for the IAE circuits."""
    x, dx = cfg["experiments"], dx_names(cfg)
    bn, _ = kb.load(cfg, kbdir)
    n = bn.struct.n
    test = kb.test_split(cfg, kbdir)
    post, pe = kb.posteriors_by_pattern(cfg, bn, test)
    ev = kb.evidence(cfg)
    A = state_prep(bn)
    rA = resources(A)
    profiles = []
    for qn in x["resource_quantiles"]:
        target_pe = float(np.quantile(pe, qn))
        i = int(np.argmin(np.abs(np.log(pe) - np.log(target_pe))))
        row = test.iloc[i]
        obs = {bn.struct.index(k): int(row[k]) for k in ev if row[k] >= 0}
        t = dx[int(np.argmax(post[i]))]
        circ = problem_circuit(A, conj_oracle(n, {**obs, bn.struct.index(t): 1}))
        rC, rQ = resources(circ), resources(_grover(circ))
        a_e, a_t = float(pe[i]), float(pe[i] * post[i].max())
        k = max(iae_relative(lambda e: iae_emulated(a, e, 0.05, cfg["engine"]["shots"], cfg["seed"]), cfg["engine"]["eps"],
                             x["resource_rel_error"])["max_grover_power"] for a in (a_e, a_t))
        profiles.append({"quantile": qn, "p_evidence": a_e, "findings": len(obs), "target": t, "qubits": rC["qubits"],
                         "k_max": int(k), "problem_depth": rC["depth"], "problem_cx": rC["two_qubit_gates"],
                         "grover_depth": rQ["depth"], "grover_cx": rQ["two_qubit_gates"],
                         "deepest_depth": rC["depth"] + k * rQ["depth"], "deepest_cx": rC["two_qubit_gates"] + k * rQ["two_qubit_gates"]})
        log.info("resources: P(E) quantile %g done", qn)
    rng = np.random.default_rng(cfg["seed"])
    pa_counts = [len(p) for p in bn.struct.parents]
    frac_ev = len(ev) / n
    scaling = []
    for size in x["resource_sizes"]:
        parents = [tuple(sorted(rng.choice(j, size=min(j, int(rng.choice(pa_counts))), replace=False).tolist())) for j in range(size)]
        cpts = [np.stack([1 - p1, p1], 1) for p1 in (rng.uniform(0.05, 0.95, 2 ** len(pa)) for pa in parents)]
        b = BN(Struct([f"x{j}" for j in range(size)], parents), cpts)
        lits = {int(j): int(rng.integers(0, 2)) for j in rng.choice(size, size=max(1, round(frac_ev * size)), replace=False)}
        circ = problem_circuit(state_prep(b), conj_oracle(size, lits))
        rP, rQ = resources(state_prep(b)), resources(_grover(circ))
        scaling.append({"propositions": size, "qubits": size + 1, "evidence_literals": len(lits), "prep_depth": rP["depth"],
                        "prep_cx": rP["two_qubit_gates"], "grover_depth": rQ["depth"], "grover_cx": rQ["two_qubit_gates"],
                        "classical_joint_states": 2 ** size})
        log.info("resources: %d propositions done", size)
    return {"assumptions": "logical qubits, all-to-all connectivity, basis rz/sx/x/cx (optimization level 1), no error "
                           "correction; deepest circuit = problem circuit + k_max Grover operators",
            "kb_state_prep": rA, "profiles": profiles, "scaling": scaling}


# ------------------------------------------------------------------------------------- other LLMs
def preflight(cfg: dict, model: str) -> dict:
    """Probe an Ollama model once: reachable/pulled, whether it accepts `think`, whether it supports tool calling.
    -> {"ok", "think", "tools", "reason"}; models without tools run grounding only (no follow-ups)."""
    a = cfg["agent"]
    msg = [{"role": "user", "content": 'Reply with the JSON object {"ok": true}.'}]
    think = False
    for attempt in (False, None):
        try:
            OllamaClient(a["url"], model, a["timeout_s"], attempt, cfg["seed"], retries=1, num_ctx=a.get("num_ctx")).chat(msg, fmt="json")
            think = attempt
            break
        except LLMError as e:
            low = str(e).lower()
            if attempt is False and "think" in low:
                continue
            return {"ok": False, "think": None, "tools": False, "reason": str(e)[:200]}
        except RuntimeError as e:  # server not reachable
            raise SystemExit(str(e)) from e
    from .agent import SEARCH_TOOL
    try:
        OllamaClient(a["url"], model, a["timeout_s"], think, cfg["seed"], retries=1, num_ctx=a.get("num_ctx")).chat(
            [{"role": "user", "content": "Search the note for 'ckd'."}], tools=[SEARCH_TOOL])
        tools = True
    except LLMError as e:
        if "tool" not in str(e).lower():
            return {"ok": False, "think": think, "tools": False, "reason": str(e)[:200]}
        tools = False
    return {"ok": True, "think": think, "tools": tools, "reason": None}


def agent_models(cfg: dict, kbdir: Path, art: Path) -> dict:
    """The frozen 1,500-note sample read by every model in experiments.agent_models (each cached separately).
    Primary comparison: grounding only (every model can do it); follow-ups for models with tool calling."""
    x, dx = cfg["experiments"], dx_names(cfg)
    ref, a = x["reference_model"], cfg["agent"]
    bn, limits = kb.load(cfg, kbdir)
    texts = sample_texts(cfg, art)
    sub = _eligible(art).set_index("note_id").loc[list(texts)].reset_index()
    post = Posteriors(bn, dx, cfg["engine"].get("calibration"))
    SD = {r["note_id"]: structured_of(cfg, r) for _, r in sub.iterrows()}
    old, misses = replay_reference(cfg, kbdir, art, texts)
    P = {"keyword extractor": np.array([post(m) for m in _stored_memberships(cfg, sub)]),
         f"{ref}: grounding": np.array([post(N.memberships(old[n]["ground_findings"], cfg, limits, structured=SD[n])) for n in sub.note_id]),
         f"{ref}: + follow-ups": np.array([post(N.memberships(old[n]["findings"], cfg, limits, structured=SD[n])) for n in sub.note_id])}
    models = {ref: {"ok": True, "think": a.get("think"), "tools": True, "reason": "reference (cached replay)",
                    "loop": loop_stats(old)}}
    panel = list(x["agent_models"]) + (list(x.get("agent_models_extended", [])) if x.get("include_extended") else [])
    for m in panel:
        if m == ref:
            continue
        try:
            assert_local(m)
        except ValueError as e:
            models[m] = {"ok": False, "think": None, "tools": False, "reason": str(e)}
            continue
        if a["backend"] == "fake":
            info, client = {"ok": True, "think": None, "tools": True, "reason": None}, FakeClient(cfg)
        else:
            info = preflight(cfg, m)
            if not info["ok"]:
                log.warning("agent-models: skipping %s (%s)", m, info["reason"])
                models[m] = info
                continue
            client = OllamaClient(a["url"], m, a["timeout_s"], info["think"], cfg["seed"], num_ctx=a.get("num_ctx"))
        info["protocol"] = "tools" if info["tools"] else "text"  # no tool calling: searches requested as JSON text
        log.info("agent-models: %s (protocol=%s, think=%s)", m, info["protocol"], info["think"])
        c = _with(cfg, **{"agent.model": m})
        res = run_sample(c, bn, limits, texts, client, art / "agent_cache" / _slug(m), protocol=info["protocol"])
        info["loop"] = loop_stats(res)
        models[m] = info
        P[f"{m}: grounding"] = np.array([post(N.memberships(res[n]["ground_findings"], cfg, limits, structured=SD[n])) for n in sub.note_id])
        P[f"{m}: + follow-ups"] = np.array([post(N.memberships(res[n]["findings"], cfg, limits, structured=SD[n])) for n in sub.note_id])
    ran = [m for m in models if models[m].get("ok") and m != ref]
    pairs = [(f"{m}: grounding", f"{ref}: grounding") for m in ran] + [(f"{m}: grounding", "keyword extractor") for m in ran]
    pairs += [(f"{med}: grounding", f"{base}: grounding") for med, base in x.get("model_pairs", []) if med in ran and base in ran]
    Y = _truths(cfg, sub)
    server = ollama_info(a["url"], [m for m in models if models[m].get("ok")]) if a["backend"] == "llm" else {"version": None}
    return {"reference": ref, "replay_cache_misses": misses, "models": models, "ollama": server,
            "metrics": {t: _evaluate(cfg, P, Y[t], sub.subject_id.to_numpy(), pairs, int(x["bootstrap"])) for t in Y}}


# ------------------------------------------------------------------------------------- report
def write_md(R: dict, cfg: dict, path: Path) -> Path:
    from .report import _ci, _f
    k = cfg["engine"]["top_k"]
    L = ["# Post-freeze analyses (tag heldout-v1)", "",
         "Robustness and secondary analyses on the held-out MIMIC-IV-Note set and the MIMIC test split. Defaults were "
         "fixed before these ran and are not changed by them. Aggregates only.", ""]

    def table(res: dict, title: str):
        out = []
        for t, lab in (("note", "Discharge Diagnosis section"), ("icd", "ICD codes")):
            if t not in res:
                continue
            r = res[t]
            out += ["", f"**{title}, ground truth = {lab}** ({r['n']:,} notes/admissions, {r['patients']:,} patients)", "",
                    f"| variant | macro AUROC | macro Brier | top-{k} hit |", "|---|---|---|---|"]
            out += [f"| {v} | {_ci(m['macro_auroc'])} | {_ci(m['macro_brier'], '.4f')} | {_ci(m[f'top{k}_hit'])} |" for v, m in r["variants"].items()]
            if r["paired"]:
                out += ["", f"| paired difference | macro AUROC | macro Brier | top-{k} hit |", "|---|---|---|---|"]
                out += [f"| {p} | {_ci(m['macro_auroc'], '+.4f')} | {_ci(m['macro_brier'], '+.4f')} | {_ci(m[f'top{k}_hit'], '+.3f')} |"
                        for p, m in r["paired"].items()]
        return out

    s = R.get("sensitivity")
    if s:
        L += ["## 1. Sensitivity of design constants", "",
              "One constant at a time; every other setting at its default (hedge 0.7, graded-lab width 0.05, max parents 3). "
              f"Agent findings are the frozen run's cached answers (uncached calls during replay: {s['replay_cache_misses']}, "
              "which fail as in the original run). Network edges per setting: " + ", ".join(f"{k} {v}" for k, v in s["edges"].items()) + "."]
        L += table(s["extractor"], "Keyword extractor, all eligible notes") + table(s["agent"], "Agent, frozen 1,500-note sample")
    f = R.get("followup-curve")
    if f:
        L += ["", "## 2. Follow-up budget curve", "",
              "The agent replayed from its cached answers with 0..3 engine-chosen follow-ups (uncached calls per step: "
              + ", ".join(f"{s} -> {c}" for s, c in f["replay_calls"].items()) + "). Follow-ups / changed findings per step: "
              + ", ".join(f"{s}: {v['follow_ups']}/{v['changed_membership']}" for s, v in f["per_step"].items()) + "."]
        if "reproduces_mimic_notes" in f:
            L.append("Reproduction of the frozen run (macro AUROC, note labels): " + "; ".join(
                f"{v} replay {_f(c['replay'], '.4f')} vs frozen {_f(c['frozen'], '.4f')} ({'match' if c['match'] else 'MISMATCH'})"
                for v, c in f["reproduces_mimic_notes"].items()) + ".")
        L += table(f["metrics"], "Agent sample")
        q = f.get("questions")
        if q:
            L += ["", f"**Was the engine's question better than a random eligible one?** {q['n']:,} engine-chosen questions; "
                  "expected information gain is exact (summed diagnosis entropy, nats); the random value is the mean over all eligible "
                  "questions at that step. Realised gain = entropy before minus after the agent's answer.", "",
                  "| quantity | mean [95% CI] |", "|---|---|",
                  f"| expected gain, engine's question | {_ci(q['engine_expected_gain'], '.4f')} |",
                  f"| expected gain, random question | {_ci(q['random_expected_gain'], '.4f')} |",
                  f"| engine minus random | {_ci(q['engine_minus_random'], '+.4f')} |",
                  f"| realised gain after the answer | {_ci(q['realised_gain'], '.4f')} |", "",
                  f"The engine's question beat the random mean in {q['engine_better_than_random_mean']:.1%} of cases. Realised gain "
                  f"when the answer changed a finding the engine uses ({q['changed']:,}): {_f(q['realised_gain_when_changed'], '.4f')}; "
                  f"otherwise {_f(q['realised_gain_when_unchanged'], '.4f')}. Entropies use the open-world view the engine selects "
                  "with, so confirming an undocumented item as absent counts as information here although the closed-world default "
                  "already treats it as absent."]
    t = R.get("temporal")
    if t:
        L += ["", "## 3. Temporal shift", "",
              f"Early-era knowledge base trained on {t['train_rows']['early']:,} training-split admissions; all test rows come "
              "from the held-out test split. Eras are MIMIC's patient-level anchor year groups. Lab limits are unchanged."]
        for era, r in t["eras"].items():
            L += ["", f"### Test era {era}: {r['test_admissions']:,} admissions, {r['test_notes']:,} notes "
                  f"(same-era knowledge base: {r['train_rows_same_era']:,} training admissions)"]
            if "structured" in r:
                L += table({"icd": r["structured"]}, "Structured MIMIC data")
            if "notes" in r:
                L += table(r["notes"], "Keyword extractor on notes")
    q = R.get("resources")
    if q:
        L += ["", "## 4. Quantum resource estimates", "", f"Assumptions: {q['assumptions']}.", "",
              "| P(findings) quantile | P(findings) | qubits | k_max | problem depth / CX | Grover depth / CX | deepest circuit depth / CX |",
              "|---|---|---|---|---|---|---|"]
        L += [f"| {p['quantile']:g} | {p['p_evidence']:.2e} | {p['qubits']} | {p['k_max']} | {p['problem_depth']:,} / {p['problem_cx']:,} | "
              f"{p['grover_depth']:,} / {p['grover_cx']:,} | {p['deepest_depth']:,} / {p['deepest_cx']:,} |" for p in q["profiles"]]
        L += ["", "| propositions | qubits | state prep depth / CX | Grover depth / CX | classical joint states |", "|---|---|---|---|---|"]
        L += [f"| {s['propositions']} | {s['qubits']} | {s['prep_depth']:,} / {s['prep_cx']:,} | {s['grover_depth']:,} / {s['grover_cx']:,} | "
              f"{s['classical_joint_states']:,} |" for s in q["scaling"]]
    am = R.get("agent-models")
    if am:
        L += ["", f"## 5. Other LLMs on the same frozen 1,500 notes (reference: {am['reference']})", "",
              "Grounding = the model's single reading of the note; + follow-ups = engine-chosen searches (native tool calling, or "
              "the JSON text protocol for models without it). Medical-vs-base pairs isolate the effect of medical fine-tuning.", "",
              "| model | ran | search protocol | think | follow-ups (answered / changed) | discarded (grounding / follow-up) | LLM errors | note |",
              "|---|---|---|---|---|---|---|---|"]
        for m, i in am["models"].items():
            st = i.get("loop", {})
            L.append(f"| {m} | {'yes' if i.get('ok') else 'no'} | {i.get('protocol', 'tools' if i.get('tools') else '-')} | {i.get('think')} | "
                     f"{st.get('follow_ups', 0):,} ({st.get('answered', 0):,} / {st.get('changed_membership', 0):,}) | "
                     f"{st.get('grounding_rejected', 0):,} / {st.get('followup_rejected', 0):,} | {st.get('llm_errors', 0):,} | {i.get('reason') or ''} |")
        L += table(am["metrics"], "Agent sample")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L))
    return path
