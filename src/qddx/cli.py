"""qddx command line. Stages write JSON to <artifacts>/results/ and are skipped when present (use --force)."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

from . import evaluate as E
from . import kb, report, synth

STAGES = ["build-kb", "eval-kb", "notes", "engines", "advantage", "privacy", "mimic-notes",
          "sensitivity", "followup-curve", "temporal", "resources"]
EXTRA = ["agent-models", "discover"]  # not part of `qddx run`
# Run 2 (`qddx test --vocab ...`): every stage that does not depend on the 200-note development file
TEST_STAGES = ["build-kb", "eval-kb", "mimic-notes", "followup-curve", "sensitivity", "temporal", "resources",
               "advantage", "agent-models"]


KINDS = {"history", "diagnosis", "medication", "lab", "demographic"}


def load_vocab(cfg: dict, base: Path) -> dict:
    """Fill cfg['propositions'] from the vocabulary JSON (cfg['vocab'], relative to the config file) and apply its
    optional 'options' (dotted keys, e.g. {"knowledge.max_parents": 4}). Validates kinds and definitions."""
    v = cfg.get("vocab")
    if not v:
        return cfg
    p = Path(v) if Path(v).is_absolute() else (base / v)
    voc = json.loads(p.read_text())
    props = voc["propositions"]
    errors = []
    for k, d in props.items():
        kind = d.get("kind")
        if kind not in KINDS:
            errors.append(f"{k}: unknown kind {kind!r}")
        elif kind in ("history", "diagnosis") and not (d.get("icd10") or d.get("icd9")):
            errors.append(f"{k}: needs icd10/icd9 codes")
        elif kind == "medication" and not d.get("drug_terms"):
            errors.append(f"{k}: needs drug_terms")
        elif kind == "lab" and not (d.get("itemids") and d.get("direction") in ("high", "low")):
            errors.append(f"{k}: needs itemids and direction high/low")
        elif kind == "demographic" and not ("age_ge" in d or "sex" in d):
            errors.append(f"{k}: needs age_ge or sex")
    if not any(d.get("kind") == "diagnosis" for d in props.values()):
        errors.append("no diagnosis propositions")
    if errors:
        raise SystemExit(f"invalid vocabulary {p}:\n  " + "\n  ".join(errors))
    cfg["propositions"] = props
    cfg["vocab_name"] = voc.get("name", p.stem)
    for key, val in (voc.get("options") or {}).items():
        d = cfg
        *parts, last = key.split(".")
        for q in parts:
            d = d.setdefault(q, {})
        d[last] = val
    return cfg


def load_config(path: str, sets: list[str], vocab: str | None = None) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if vocab:
        cfg["vocab"] = str(Path(vocab).resolve())
    for s in sets or []:
        key, val = s.split("=", 1)
        d = cfg
        *parts, last = key.split(".")
        for p in parts:
            d = d[p]
        d[last] = yaml.safe_load(val)
    return load_vocab(cfg, Path(path).resolve().parent)


def _json(o):
    if isinstance(o, float) and o != o:
        return None
    raise TypeError(type(o))


def stage(cfg: dict, name: str, force: bool) -> dict:
    art = Path(cfg["paths"]["artifacts"])
    kbdir, res = art / "kb", art / "results"
    res.mkdir(parents=True, exist_ok=True)
    out = res / f"{name}.json"
    if out.exists() and not force:
        return json.loads(out.read_text())
    logging.getLogger("qddx").info("stage %s", name)
    if name == "build-kb":
        r = kb.build(cfg, kbdir)
    elif name == "eval-kb":
        r = E.eval_kb(cfg, kbdir)
    elif name == "notes":
        r = E.run_notes(cfg, kbdir, art)
        r.update(E.eval_notes(cfg, art))
    elif name == "engines":
        r = E.compare_engines(cfg, kbdir, art)
    elif name == "advantage":
        r = E.advantage(cfg, kbdir)
        profs = r.pop("_profiles_obj")
        r["circuits"] = E.circuit_validation(cfg, kbdir, profs)
    elif name == "privacy":
        r = E.privacy(cfg, art)
    elif name == "mimic-notes":
        from .mimicnotes import run as mimic_run
        r = mimic_run(cfg, kbdir, art)
    elif name == "discover":
        from .vocab import discover
        r = discover(cfg, kbdir, art)
    elif name in ("sensitivity", "followup-curve", "temporal", "resources", "agent-models"):
        from . import experiments as X
        fn = {"sensitivity": X.sensitivity, "followup-curve": X.followup_curve, "temporal": X.temporal,
              "resources": X.quantum_resources, "agent-models": X.agent_models}[name]
        r = fn(cfg, kbdir, art)
    else:
        raise SystemExit(f"unknown stage {name}")
    out.write_text(json.dumps(r, indent=1, default=_json))
    return json.loads(out.read_text())


def write_report(cfg: dict) -> Path:
    res = Path(cfg["paths"]["artifacts"]) / "results"
    R = {n: json.loads((res / f"{n}.json").read_text()) for n in STAGES if (res / f"{n}.json").exists()}
    X = {n: json.loads((res / f"{n}.json").read_text()) for n in ("sensitivity", "followup-curve", "temporal", "resources",
                                                                   "agent-models") if (res / f"{n}.json").exists()}
    if X:
        from . import experiments
        experiments.write_md(X, cfg, Path(cfg["paths"]["reports"]) / "experiments.md")
    adv = R.get("advantage")
    return report.write({"kb_summary": R.get("build-kb"), "eval_kb": R.get("eval-kb"), "eval_notes": R.get("notes"),
                         "engines": R.get("engines"), "advantage": adv, "circuits": adv and adv.get("circuits"),
                         "privacy": R.get("privacy")}, cfg, Path(cfg["paths"]["reports"]))


def run_test(a, base_cfg: dict) -> None:
    """Run 2: every stage with the frozen vocabulary, in <artifacts>/<name>/ and <reports>/<name>/, on the same
    held-out notes and the same frozen agent sample as v1; then the v1-vs-v2 comparison."""
    import shutil
    from .vocab import compare_vocabs, write_compare_md
    if not a.vocab:
        raise SystemExit("qddx test needs --vocab configs/vocab_v2.1.json (make it with qddx freeze-vocab)")
    cfg = load_config(a.config, a.set, vocab=a.vocab)
    name = cfg["vocab_name"]
    art1, rep1 = Path(base_cfg["paths"]["artifacts"]), Path(base_cfg["paths"]["reports"])
    art2, rep2 = art1 / name, rep1 / name
    cfg["paths"]["artifacts"], cfg["paths"]["reports"] = str(art2), str(rep2)
    sample1, sample2 = art1 / "mimic_notes" / "agent_sample.txt", art2 / "mimic_notes" / "agent_sample.txt"
    if not sample1.exists():
        raise SystemExit(f"{sample1} missing: run `qddx mimic-notes` with v1 first (it freezes the agent sample)")
    if not sample2.exists():
        sample2.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(sample1, sample2)
    if sample1.read_text() != sample2.read_text():
        raise SystemExit(f"{sample2} differs from the frozen v1 sample")
    log = logging.getLogger("qddx")
    for s in TEST_STAGES:
        log.info("test %s: stage %s", name, s)
        stage(cfg, s, a.force)
    out = art2 / "results" / "compare.json"
    if a.force or not out.exists():
        r = compare_vocabs(base_cfg, art1, cfg, art2)
        out.write_text(json.dumps(r, indent=1, default=_json))
    write_compare_md(json.loads(out.read_text()), cfg, rep2 / "compare.md")
    print(write_report(cfg))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser("qddx", description="Q-DDx: agentic + quantum-symbolic differential diagnosis")
    ap.add_argument("command", choices=STAGES + EXTRA + ["run", "report", "explain", "synth", "freeze-vocab", "test"])
    ap.add_argument("--config", default="configs/qddx.yaml")
    ap.add_argument("--set", action="append", default=[], help="override, e.g. --set agent.backend=baseline")
    ap.add_argument("--force", action="store_true", help="recompute the stage(s)")
    ap.add_argument("--note-id", help="for explain")
    ap.add_argument("--out", default=None, help="synth: output folder; freeze-vocab: vocabulary file (configs/vocab_v2.1.json)")
    ap.add_argument("--vocab", help="test: the frozen vocabulary file (e.g. configs/vocab_v2.1.json)")
    ap.add_argument("--items", help="freeze-vocab: comma-separated pool items, top:N, recommended, or @file (one item per line)")
    ap.add_argument("--drop", default="", help="freeze-vocab: comma-separated items to leave out")
    ap.add_argument("--name", default="v2", help="freeze-vocab: vocabulary name (also the artifacts/reports subfolder)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s", stream=sys.stderr)
    logging.getLogger("qddx").setLevel(logging.INFO)
    if a.command == "synth":
        out = Path(a.out or "artifacts/synthetic")
        synth.make_mimic(out / "mimic")
        synth.make_notes(out / "notes.csv")
        print(f"wrote {out}/mimic and {out}/notes.csv\nrun: qddx run --set paths.mimic_root={out.as_posix()}/mimic "
              f"--set paths.notes_csv={out.as_posix()}/notes.csv --set paths.artifacts={out.as_posix()}/artifacts "
              f"--set paths.reports={out.as_posix()}/reports --set agent.backend=fake")
        return
    cfg = load_config(a.config, a.set)
    if a.command == "freeze-vocab":
        from .vocab import freeze, resolve_items
        if not a.items:
            raise SystemExit("give the chosen items: --items top:24  (or a comma list / @file), optionally --drop a,b")
        art = Path(cfg["paths"]["artifacts"])
        items = resolve_items(a.items, a.drop, art)
        r = freeze(cfg, art / "kb", art, items, Path(a.out or f"configs/vocab_{a.name}.json"), a.name)
        print(json.dumps(r, indent=1))
        return
    if a.command == "test":
        run_test(a, cfg)
        return
    if a.command == "explain":
        p = Path(cfg["paths"]["artifacts"]) / "note_reports" / f"{a.note_id}.txt"
        print(p.read_text() if p.exists() else f"no report for {a.note_id}; run `qddx notes` first")
        return
    if a.command in STAGES + EXTRA:
        stage(cfg, a.command, a.force)
    elif a.command == "run":
        for s in STAGES:
            stage(cfg, s, a.force)
    print(write_report(cfg))


if __name__ == "__main__":
    main()
