"""Markdown report: every number is read from the saved results; all prose is templated."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _ci(x: dict, fmt: str = ".3f") -> str:
    v, (lo, hi) = x["value"], x["ci"]
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v:{fmt}}" + (f" [{lo:{fmt}}, {hi:{fmt}}]" if lo is not None else "")


def _f(v, fmt=".3f"):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:{fmt}}"


def figure(adv: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for m, lab, mk in (("rejection", "rejection sampling", "o"), ("lw", "likelihood weighting", "s"), ("iae", "amplitude estimation (IAE)", "^")):
        pts = [(1 / r["p_evidence"], r[m]) for r in adv["profiles"] if r[m] is not None]
        if pts:
            x, y = np.array(pts).T
            sl = adv["slopes"][m]["slope"]
            ax.scatter(x, y, marker=mk, s=18, alpha=0.7, label=f"{lab}" + (f" (slope {sl:.2f})" if sl is not None else ""))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("1 / P(findings)")
    ax.set_ylabel(f"model evaluations for {adv['rel_error']:.0%} relative error")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write(R: dict, cfg: dict, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    P = cfg["propositions"]
    dx = [k for k, p in P.items() if p["kind"] == "diagnosis"]
    k = cfg["engine"]["top_k"]
    L = ["# Q-DDx: early differential diagnosis from the admission note", "",
         "An agentic language model reads the de-identified admission note into graded findings; a quantum-symbolic "
         "engine (one qubit per proposition, dependencies learned from MIMIC-IV) returns P(diagnosis | findings) with "
         "reasons; the result is checked against the diagnoses the patient was discharged with. "
         "All figures below are read from saved results. Research only; not for clinical use.", ""]
    s = R.get("kb_summary")
    if s:
        L += ["## 1. Knowledge base (MIMIC-IV v3.1, hosp)", "",
              f"{s['admissions']:,} adult admissions of {s['subjects']:,} patients; subject-level split "
              f"{s['train_admissions']:,} train / {s['test_admissions']:,} test. Propositions: {len(P)} "
              f"({len(dx)} diagnoses). Learned edges ({len(s['edges'])}, max {cfg['knowledge']['max_parents']} parents): "
              + ", ".join(f"{a} -> {b}" for a, b in s["edges"]) + ".", "",
              "| proposition | kind | train prevalence | missing |", "|---|---|---|---|"]
        L += [f"| {n} | {P[n]['kind']} | {s['prevalence_train'][n]:.3f} | {s['missing_train'][n]:.1%} |" for n in P]
        L += ["", "Data-derived lab limits (median MIMIC reference range): " + "; ".join(
            f"{n} {_f(v['lower'], 'g')}-{_f(v['upper'], 'g')}" for n, v in s["limits"].items()) + ".", ""]
    e = R.get("eval_kb")
    if e:
        L += ["## 2. MIMIC-scale validation (held-out admissions)", "",
              f"{e['n_test']:,} test admissions ({e['n_test_subjects']:,} patients). Evidence: coded history, first-24 h "
              f"medications, first admission labs. 95% CIs: patient bootstrap ({cfg['evaluation']['bootstrap_kb']} resamples).", "",
              f"| model | macro AUROC | macro Brier | top-{k} hit |", "|---|---|---|---|"]
        for name, m in e["models"].items():
            L.append(f"| {name} | {_ci(m['macro_auroc'])} | {_ci(m['macro_brier'], '.4f')} | {_ci(m[f'top{k}_hit'])} |")
        L += ["", "| diagnosis | test prevalence | AUROC learned network | AUROC logistic | ECE learned network |", "|---|---|---|---|---|"]
        mn, lr = e["models"]["learned network"], e["models"]["logistic regression"]
        L += [f"| {P[d]['label']} | {e['prevalence_test'][d]:.3f} | {_ci(mn[f'auroc_{d}'])} | {_ci(lr[f'auroc_{d}'])} | {_f(mn[f'ece_{d}']['value'])} |" for d in dx]
        q = e["p_evidence_quantiles"]
        L += ["", f"Rarity of full admission profiles: median P(findings) = {q['50%']:.2e} (1st percentile {q['1%']:.2e}); "
              f"{e['frac_p_evidence_below']['0.001']:.1%} of test admissions have P(findings) < 1e-3 and "
              f"{e['frac_p_evidence_below']['0.0001']:.1%} < 1e-4; {e['frac_profiles_unseen_in_train']:.1%} of test admissions have "
              f"a finding profile never seen in training ({e['unique_profiles_test']:,} distinct profiles). "
              f"Exact inference over all test admissions took {e['exact_seconds']:.1f} s.", ""]
    n = R.get("eval_notes")
    if n:
        L += ["## 3. Admission-note study", "",
              f"{n['n_notes']} notes with a discharge-diagnosis section ({n['n_subjects']} patients). Ground truth: that "
              f"section, matched to the diagnosis vocabulary. Positives: " + ", ".join(f"{P[d]['label']} {c}" for d, c in n["label_counts"].items())
              + f". 95% CIs: patient bootstrap ({cfg['evaluation']['bootstrap']} resamples); with this many positives they are wide.", "",
              f"| variant | macro AUROC | macro Brier | top-{k} hit |", "|---|---|---|---|"]
        names = {"baseline": "keyword/negation extractor (no LLM)", "agent_ground": "agent, grounding only (no follow-ups)",
                 "agent": "agent with engine-driven follow-ups (main)", "agent_crisp": "agent, graded findings made crisp",
                 "agent_naive": "agent, naive network (1 parent)", "agent_logistic": "agent findings, logistic regression",
                 "baseline_naive": "extractor, naive network", "baseline_logistic": "extractor findings, logistic regression"}
        for v, m in n["variants"].items():
            L.append(f"| {names.get(v, v)} | {_ci(m['macro_auroc'])} | {_ci(m['macro_brier'], '.4f')} | {_ci(m[f'top{k}_hit'])} |")
        if n["paired"]:
            L += ["", "Paired differences (reference minus variant; positive AUROC/top-k and negative Brier favour the reference):", "",
                  f"| comparison | macro AUROC | macro Brier | top-{k} hit |", "|---|---|---|---|"]
            for c, m in n["paired"].items():
                L.append(f"| {c} | {_ci(m['macro_auroc'], '+.3f')} | {_ci(m['macro_brier'], '+.4f')} | {_ci(m[f'top{k}_hit'], '+.3f')} |")
        a = n["agent_loop"]
        L += ["", f"Agent: {a['follow_ups']} engine-requested follow-ups in {a['notes_with_follow_up']} notes "
              f"({a['resolved']} answered, {a.get('changed_membership', 0)} changed a finding the engine uses, "
              f"{a['searches']} note searches); {a['grounding_rejected']} grounding findings and "
              f"{a['followup_rejected']} follow-up answers were discarded by quote/value validation. "
              f"LLM server errors after retries: {a.get('llm_errors', 0)} (grounding fell back to the extractor in "
              f"{a.get('grounding_fallbacks', 0)} notes). "
              f"Graded findings used: {n['graded_findings']['count']} in {n['graded_findings']['notes']} notes. "
              f"Mean time per note {n['seconds_per_note']:.1f} s.", ""]
    c = R.get("engines")
    if c:
        L += ["## 4. Engines on the notes' graded evidence", "",
              f"Error = largest absolute difference from exact enumeration over the {len(dx)} diagnoses. Matched cost: IAE "
              f"(P(findings) refined to {cfg['engine']['rel_eps']:.0%} relative precision, each joint amplitude to {cfg['engine']['rel_eps']:.0%} of P(findings)) runs first; rejection sampling and "
              f"likelihood weighting then get the same number of model evaluations (capped at {cfg['engine']['budget_samples']:,}). "
              f"Median P(findings) = {c['p_evidence_median']:.2e}.", "",
              "| engine | failed | median error | 90th pct error | median model evaluations |", "|---|---|---|---|---|"]
        for b in ("rejection", "lw", "quantum"):
            m = c["summary"].get(b)
            if m:
                L.append(f"| {b} | {m['failed']}/{m['notes']} | {_f(m['median_max_abs_error'])} | {_f(m['p90_max_abs_error'])} | {m['median_cost']:,.0f} |")
        rq = c["summary"].get("rarest_quartile", {})
        cc = c["circuit_checks"]
        L += ["", "Rarest quartile of notes, median error: " + ", ".join(f"{b} {_f(v)}" for b, v in rq.items()) + ".",
              f"Gate-level check on {cc['notes']} notes: largest |circuit amplitude - exact| = {_f(cc['max_abs_error'], '.1e')}; "
              f"{cc['qubits_range'][0]}-{cc['qubits_range'][1]} qubits; state preparation depth up to {cc['prep_depth_max']} "
              f"with {cc['prep_two_qubit_gates_max']} CX (rz/sx/x/cx basis)."
              + (f" {cc['skipped_too_large']} notes skipped: circuit above {cfg['evaluation']['circuit_max_qubits']} qubits." if cc.get("skipped_too_large") else ""), ""]
    ad = R.get("advantage")
    if ad:
        sl = ad["slopes"]
        L += ["## 5. Quantum advantage under rare evidence (MIMIC test profiles)", "",
              f"{len(ad['profiles'])} test admissions with {', '.join(map(str, cfg['evaluation']['advantage']['evidence_sizes']))} findings revealed. "
              f"Cost = model evaluations (one network sample = one application of the state preparation A) for 90% of {cfg['evaluation']['advantage']['repeats']} repeats to reach "
              f"{ad['rel_error']:.0%} relative error on P(top diagnosis | findings). Theory: rejection ~ P(E)^-1, amplitude estimation ~ P(E)^-1/2.", "",
              "| estimator | slope of log cost vs log 1/P(E) | profiles reached | censored |", "|---|---|---|---|"]
        for m, lab in (("rejection", "rejection sampling"), ("lw", "likelihood weighting"), ("iae", "amplitude estimation")):
            L.append(f"| {lab} | {_f(sl[m]['slope'], '.2f')} | {sl[m]['achieved']} | {sl[m]['censored']} |")
        cr = ad.get("crossover_p_evidence")
        L += ["", (f"Fitted crossover: amplitude estimation is cheaper below P(findings) ~ {cr:.1e}. " if cr else "")
              + f"Median cost ratio rejection / amplitude estimation: {_f(ad['median_ratio_rejection_over_iae'], '.1f')}x "
              f"(rarest decile: {_f(ad['rarest_decile_ratio'], '.1f')}x). Censored = not reached within "
              f"{ad['max_samples']:,} samples (rejection) or the configured grids. Exact enumeration costs "
              f"{ad['exact_model_evaluations']:,} evaluations at this size and grows as 2^n; it is the reference, "
              f"not a scalable competitor.", "", "![advantage](advantage.png)", ""]
        cv = R.get("circuits")
        if cv:
            L += [f"Real Grover circuits (Aer) vs the analytic amplification law used above (sub-network of the findings, "
                  f"target and their ancestors; {cv.get('profiles_too_large', 0)} profiles above {cv.get('max_qubits')} qubits not simulated):", "",
                  "| P(E) | exact | circuit IAE | analytic IAE | circuit calls | analytic calls | qubits |", "|---|---|---|---|---|---|---|"]
            L += [f"| {r['p_evidence']:.3f} | {r['exact']:.3f} | {r['circuit']:.3f} | {r['analytic']:.3f} | {r['circuit_calls']:,} | {r['analytic_calls']:,} | {r['qubits']} |" for r in cv["ideal"]]
            if cv["noise"]:
                L += ["", "Depolarizing noise (simulated; not evidence about any real device):", "", "| 2-qubit error | P(E) | exact | estimate | abs error |", "|---|---|---|---|---|"]
                L += [f"| {r['p2']:g} | {r['p_evidence']:.3f} | {r['exact']:.3f} | {r['estimate']:.3f} | {r['abs_error']:.3f} |" for r in cv["noise"]]
            L.append("")
    pv = R.get("privacy")
    if pv:
        L += ["## 6. Privacy", "",
              f"{pv['phi_items']} header PHI items (values and name tokens) across {pv['notes']} notes.", "",
              "| text | notes with any PHI | PHI items found |", "|---|---|---|",
              f"| agent input without de-identification gate | {pv['agent_input_gate_off']['notes_with_leak']} | {pv['agent_input_gate_off']['items_leaked']} |",
              f"| agent input with gate (used) | {pv['agent_input_gate_on']['notes_with_leak']} | {pv['agent_input_gate_on']['items_leaked']} |",
              f"| whole de-identified note | {pv['deidentified_note']['notes_with_leak']} | {pv['deidentified_note']['items_leaked']} |",
              f"| generated reports ({pv['reports']['checked']}) | {pv['reports']['notes_with_leak']} | {pv['reports']['items_leaked']} |", "",
              "The language model runs locally (Ollama); no note leaves the machine. Reports are f-string templates over "
              "proposition labels and computed numbers, so they cannot carry note text.", ""]
    L += ["## 7. Limitations", "",
          "- Knowledge base and notes are different populations and never linked; findings in notes (home medication lists, "
          "narrative history) differ from MIMIC's coded history and first-24 h orders.",
          "- ICD codes are billing labels; note labels come from the discharge-diagnosis section and its vocabulary match.",
          "- 200 notes give wide intervals; ablations add comparisons, not statistical power.",
          "- Quantum results are simulations. The analytic amplification law is exact for a noiseless device and is "
          "checked against real circuits above; it says nothing about hardware noise at depth.",
          "- At 16 propositions exact enumeration is feasible; the sampling-cost comparison is the scalability argument.", ""]
    (out / "report.md").write_text("\n".join(L))
    if ad:
        figure(ad, out / "advantage.png")
    return out / "report.md"
