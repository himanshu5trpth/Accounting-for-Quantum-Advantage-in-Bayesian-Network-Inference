"""Run 1 (discover) and Run 2 (test --vocab) end to end on synthetic data, with a vocabulary that exercises every
proposition kind: previous-admission history, demographics, drugs, labs (high and low of the same item)."""
import csv
import json

import pandas as pd

from qddx.cli import load_config, main
from qddx.synth import make_mimic, make_mimic_notes, make_notes


def test_discover_then_test_vocab(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    make_mimic(tmp_path / "mimic", n_subjects=300)
    make_notes(tmp_path / "notes.csv", n=10)
    with open(tmp_path / "mimic" / "hosp" / "admissions.csv" / "admissions.csv", newline="") as f:
        adm = [(int(r["subject_id"]), int(r["hadm_id"])) for r in csv.DictReader(f)]
    make_mimic_notes(tmp_path / "discharge.csv.gz", adm)
    from conftest import ROOT
    sets = ["paths.mimic_root=mimic", "paths.notes_csv=notes.csv", "paths.mimic_notes=discharge.csv.gz", "agent.backend=fake",
            "agent.model=fake", "experiments.reference_model=fake", "evaluation.bootstrap=5", "evaluation.bootstrap_kb=5",
            "evaluation.quantum_notes=1", "evaluation.advantage.profiles=4", "evaluation.advantage.repeats=3",
            "evaluation.advantage.circuit_check_profiles=1", "evaluation.noise.depolarizing_2q=[]", "mimic_notes.agent_sample=8",
            "mimic_notes.engines_sample=3", "mimic_notes.bootstrap=5", "experiments.bootstrap=5", "experiments.parents_grid=[2]",
            "experiments.hedge_grid=[0.7]", "experiments.width_grid=[0.05]", "experiments.resource_sizes=[16]",
            "experiments.resource_quantiles=[0.5]", "experiments.agent_models=[fake, fake2]", "experiments.model_pairs=[]",
            "discover.notes_fraction=1.0", "discover.min_prevalence=0.01", "discover.k_grid=[4,6]", "discover.parents_grid=[2]",
            "discover.exclude_flags=[same_concept]", "discover.draft_path=vocab_v2.draft.json"]
    base = ["--config", str(ROOT / "configs" / "qddx.yaml")] + sum([["--set", s] for s in sets], [])
    main(["build-kb", *base])
    main(["mimic-notes", *base])
    main(["discover", *base])
    d = json.loads((tmp_path / "artifacts" / "results" / "discover.json").read_text())
    assert d["candidates"] > 0 and d["recommended"] and (tmp_path / "vocab_v2.draft.json").exists()
    assert (tmp_path / "reports" / "vocab_discovery.md").exists()
    # the discussion's choice -> frozen vocabulary (options chosen on validation patients)
    main(["freeze-vocab", *base, "--items", "recommended", "--drop", "", "--out", "vocab_frozen.json"])
    fz = json.loads((tmp_path / "vocab_frozen.json").read_text())
    assert len(fz["selection"]["items"]) == d["recommended"]["K"] and "knowledge.max_parents" in fz["options"]
    assert load_config(str(ROOT / "configs" / "qddx.yaml"), sets, vocab=str(tmp_path / "vocab_frozen.json"))["vocab_name"] == "v2"
    # no test-split admission in the pool
    pool = pd.read_parquet(tmp_path / "artifacts" / "vocab" / "pool.parquet")
    cohort = pd.read_parquet(tmp_path / "artifacts" / "kb" / "cohort.parquet")
    from qddx import kb
    cfg = load_config(str(ROOT / "configs" / "qddx.yaml"), sets)
    test_hadm = set(kb.test_split(cfg, tmp_path / "artifacts" / "kb").hadm_id)
    assert not (set(pool.hadm_id) & test_hadm) and len(cohort) > 0
    # a vocabulary with one item of every kind (labs: high and low of creatinine)
    from qddx.vocab import vocab_props
    meta = pd.read_parquet(tmp_path / "artifacts" / "vocab" / "meta.parquet")
    have = set(meta["item"])
    want = [i for i in ("demo:age_ge_65", "demo:sex_f", "rx:furosemide", "rx:metoprolol", "rx:potassium chloride", "lab:50912:high", "lab:50912:low",
                        "lab:51222:low") if i in have]
    want += [next(i for i in sorted(have) if i.startswith("hx10:N18") or i.startswith("hx10:E11"))]
    assert "rx:potassium chloride" in want
    voc = {"name": "v2", "options": {"knowledge.max_parents": 2}, "propositions": vocab_props(cfg, want, meta, {})}
    (tmp_path / "vocab_v2.json").write_text(json.dumps(voc))
    main(["test", *base, "--vocab", "vocab_v2.json"])
    v2 = tmp_path / "artifacts" / "v2"
    R = {p.stem: json.loads(p.read_text()) for p in (v2 / "results").glob("*.json")}
    assert {"build-kb", "mimic-notes", "agent-models", "sensitivity", "followup-curve", "temporal", "resources", "compare"} <= set(R)
    assert (v2 / "mimic_notes" / "agent_sample.txt").read_text() == (tmp_path / "artifacts" / "mimic_notes" / "agent_sample.txt").read_text()
    c = R["compare"]
    assert c["same_eligible_notes"] and "v2 extractor - v1 extractor" in c["metrics"]["icd"]["paired"]
    assert c["propositions"]["v2"] == len(voc["propositions"])
    assert (tmp_path / "reports" / "v2" / "compare.md").exists()
    kinds = {p["kind"] for p in voc["propositions"].values()}
    assert {"demographic", "history", "medication", "lab", "diagnosis"} <= kinds
    # Joint-diagnosis test (scripts/pairs.py) on the v2 artifacts: sanity checks, arms, verdict, outputs
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import pairs
    cfg2 = load_config(str(ROOT / "configs" / "qddx.yaml"), sets, vocab=str(tmp_path / "vocab_v2.json"))
    monkeypatch.setattr(pairs, "load_config", lambda *a, **k: cfg2)
    monkeypatch.setattr(pairs, "KBDIR", v2 / "kb")
    monkeypatch.setattr(pairs, "RES", v2 / "results")
    monkeypatch.setattr(pairs, "REP", tmp_path / "reports" / "v2")
    pairs.main(quick=True)
    pj = json.loads((v2 / "results" / "pairs.json").read_text())
    assert pj["verdict"] in {"PASS", "FAIL", "PARTIAL", "INCONCLUSIVE"} and len(pj["table"]) == 15
    assert (tmp_path / "reports" / "v2" / "pairs.md").exists() and (tmp_path / "reports" / "v2" / "pairs.png").exists()
