import csv
import json

from qddx.cli import main
from qddx.synth import make_mimic, make_mimic_notes, make_notes


def test_pipeline_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    make_mimic(tmp_path / "mimic", n_subjects=250)
    make_notes(tmp_path / "notes.csv", n=16)
    with open(tmp_path / "mimic" / "hosp" / "admissions.csv" / "admissions.csv", newline="") as f:
        adm = [(int(r["subject_id"]), int(r["hadm_id"])) for r in csv.DictReader(f)]
    make_mimic_notes(tmp_path / "discharge.csv.gz", adm)  # all admissions: train, test, dev-patient and minors
    from conftest import ROOT
    base = ["--config", str(ROOT / "configs" / "qddx.yaml"), "--set", "paths.mimic_root=mimic", "--set", "paths.notes_csv=notes.csv",
            "--set", "agent.backend=fake", "--set", "evaluation.bootstrap=30", "--set", "evaluation.bootstrap_kb=10",
            "--set", "evaluation.quantum_notes=2", "--set", "evaluation.advantage.profiles=6", "--set", "evaluation.advantage.repeats=4",
            "--set", "evaluation.advantage.circuit_check_profiles=1", "--set", "evaluation.noise.depolarizing_2q=[]",
            "--set", "paths.mimic_notes=discharge.csv.gz", "--set", "mimic_notes.agent_sample=20", "--set", "mimic_notes.engines_sample=5",
            "--set", "mimic_notes.bootstrap=20", "--set", "agent.model=fake", "--set", "experiments.reference_model=fake",
            "--set", "experiments.bootstrap=10", "--set", "experiments.parents_grid=[1,3]", "--set", "experiments.resource_sizes=[16,18]",
            "--set", "experiments.resource_quantiles=[0.5,0.1]"]
    main(["run", *base])
    R = {p.stem: json.loads(p.read_text()) for p in (tmp_path / "artifacts" / "results").glob("*.json")}
    assert set(R) == {"build-kb", "eval-kb", "notes", "engines", "advantage", "privacy", "mimic-notes",
                      "sensitivity", "followup-curve", "temporal", "resources"}
    s = R["sensitivity"]
    assert s["replay_cache_misses"] == 0 and "hedge=0.9" in s["extractor"]["note"]["variants"] and "crisp" in s["agent"]["icd"]["variants"]
    assert "max_parents=1" in s["settings"] and "default - default" not in s["extractor"]["note"]["paired"]
    f = R["followup-curve"]
    assert all(v == 0 for v in f["replay_calls"].values()) and all(c["match"] for c in f["reproduces_mimic_notes"].values())
    assert f["questions"]["engine_minus_random"]["value"] >= -1e-12
    t = R["temporal"]
    assert set(t["eras"]) == {"2017 - 2019", "2020 - 2022"} and "structured" in t["eras"]["2017 - 2019"]
    q = R["resources"]
    assert [x["propositions"] for x in q["scaling"]] == [16, 18] and all(p["k_max"] >= 0 for p in q["profiles"])
    main(["agent-models", *base, "--set", "experiments.agent_models=[fake, fake2, fake3]",
          "--set", "experiments.model_pairs=[[fake2, fake3]]"])
    am = json.loads((tmp_path / "artifacts" / "results" / "agent-models.json").read_text())
    v = am["metrics"]["note"]["variants"]
    assert am["replay_cache_misses"] == 0 and "fake2: grounding" in v and "fake3: + follow-ups" in v
    assert "fake2: grounding - fake3: grounding" in am["metrics"]["icd"]["paired"]
    main(["report", *base])
    assert "Post-freeze analyses" in (tmp_path / "reports" / "experiments.md").read_text()
    m = R["mimic-notes"]
    c = m["counts"]
    assert c["eligible"] > 0 and c["kb_train_split"] > 0 and c["dev_patient"] > 0 and c["not_in_kb"] > 0
    assert c["notes_in_file"] == sum(v for k, v in c.items() if k != "notes_in_file")
    assert m["agent_loop"]["notes"] == 20 and "agent" in m["agent_sample"]["icd"]["variants"]
    assert "structured" in m["all"]["note"]["variants"] and m["engines"]["circuit_checks"]["max_abs_error"] < 1e-9
    assert len((tmp_path / "artifacts" / "mimic_notes" / "agent_sample.txt").read_text().split()) == 20
    assert "Held-out test" in (tmp_path / "reports" / "mimic_notes.md").read_text()
    assert R["notes"]["n_notes"] == 16 and "agent" in R["notes"]["variants"]
    assert R["engines"]["circuit_checks"]["max_abs_error"] < 1e-9
    assert R["privacy"]["agent_input_gate_on"]["items_leaked"] == 0 and R["privacy"]["reports"]["items_leaked"] == 0
    assert R["privacy"]["agent_input_gate_off"]["items_leaked"] > 0
    rep = (tmp_path / "reports" / "report.md").read_text()
    assert "Quantum advantage" in rep and (tmp_path / "reports" / "advantage.png").exists()
    note_id = R["notes"] and next((tmp_path / "artifacts" / "note_reports").glob("*.txt")).stem
    main(["explain", "--note-id", note_id, *base])
