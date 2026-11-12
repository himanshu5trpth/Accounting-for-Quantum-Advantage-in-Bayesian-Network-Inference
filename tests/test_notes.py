
from qddx import notes as N
from qddx.synth import make_notes


def _note(tmp_path):
    return N.load_notes(make_notes(tmp_path / "n.csv", n=6))


def test_deidentification_removes_header_phi(tmp_path):
    for r in _note(tmp_path):
        deid, phi = N.deidentify(r["text"])
        assert phi["values"] and phi["names"]
        assert N.leaked(deid, phi) == []
        assert N.leaked(r["text"], phi)


def test_header_labels_are_not_names():
    text = "Name:  Quenby Ashgrove Unit No: 12-Q3456\nAttending: Dr. Halvard Quist\nHPI: no fever, on aspirin, Quenby ok"
    deid, phi = N.deidentify(text)
    assert set(phi["names"]) == {"Quenby", "Ashgrove", "Halvard", "Quist"} and "12-Q3456" in phi["values"]
    assert "no fever, on aspirin" in deid and "Quenby" not in deid and "12-Q3456" not in deid


def test_evidence_text_is_admission_time_only(cfg, tmp_path):
    r = _note(tmp_path)[0]
    ev = N.evidence_text(r["text"], cfg)
    assert "Treated for" not in ev                 # brief hospital course excluded
    assert "Creat-0.8 Na-140" not in ev            # discharge labs cut
    assert "Discharge Diagnosis" not in ev and "Furosemide 40 mg" not in ev
    assert "Pertinent Results" in ev and "Past Medical History" in ev


def test_labels_and_baseline_findings(cfg):
    text = ("Past Medical History:\n- chronic kidney disease\n- denies COPD\n- possible diabetes\n"
            "Pertinent Results:\nADMISSION LABS: Creat-2.4 Na-128 WBC-6.0\n"
            "Medications on Admission:\n1. Apixaban 5 mg\n"
            "Discharge Diagnosis:\nacute kidney injury\nno pneumonia\n")
    y = N.labels(text, cfg)
    assert y["aki"] == 1 and y["pneumonia"] == 0 and y["heart_failure"] == 0
    f = N.baseline_findings(N.evidence_text(text, cfg), cfg)
    assert f["ckd"]["status"] == "present" and f["copd"]["status"] == "absent" and f["diabetes"]["status"] == "possible"
    assert f["anticoagulant"]["status"] == "present" and f["creat_high"]["value"] == 2.4 and f["hgb_low"]["status"] == "unknown"
    lim = {"creat_high": {"lower": 0.5, "upper": 1.2, "direction": "high"}, "na_low": {"lower": 135, "upper": 145, "direction": "low"},
           "wbc_high": {"lower": 4, "upper": 11, "direction": "high"}, "hgb_low": {"lower": 12, "upper": 16, "direction": "low"}}
    m = N.memberships(f, cfg, lim)
    assert m["ckd"] == 1 and m["copd"] == 0 and m["diabetes"] == cfg["notes"]["hedge_membership"]
    assert m["creat_high"] == 1 and m["wbc_high"] == 0 and m["hgb_low"] is None and m["beta_blocker"] == 0
    assert 0 < m["na_low"] <= 1


def test_lab_membership_is_graded_and_monotone():
    lim = {"lower": 0.5, "upper": 1.2, "direction": "high"}
    xs = [1.0, 1.15, 1.2, 1.25, 1.4]
    ms = [N.lab_membership(x, lim, 0.05) for x in xs]
    assert ms == sorted(ms) and ms[0] == 0 and ms[-1] == 1 and abs(ms[2] - 0.5) < 1e-9


def test_lab_membership_with_one_sided_limit():
    from qddx.notes import lab_membership
    bnp = {"lower": None, "upper": 353.0, "direction": "high"}
    assert lab_membership(5000.0, bnp, 0.05) == 1.0 and lab_membership(50.0, bnp, 0.05) == 0.0
    assert lab_membership(5000.0, {"lower": None, "upper": 353.0, "direction": "low"}, 0.05) is None
