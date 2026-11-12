import numpy as np
import pandas as pd

from qddx.vocab import leak_flags, mine_lab_aliases


def test_lab_alias_mining_rejects_grades_and_single_letters():
    labs = "\n".join(["___ BLOOD Creat-1.2 UREA N-25 Anisocy-1+ INR(PT)-1.4 PT-13.1"] * 30)
    notes = [{"hadm_id": h, "labs": labs} for h in range(30)]
    vals = pd.DataFrame([(h, it, v) for h in range(30) for it, v in ((50912, 1.2), (51006, 25.0), (51237, 1.4), (51274, 13.1))],
                        columns=["hadm_id", "itemid", "v"])
    al = {k: [a for a, _, _ in v] for k, v in mine_lab_aliases(notes, vals).items()}
    assert al[50912] == ["Creat"] and al[51237] == ["INR(PT)"] and al[51274] == ["PT"]
    assert not any(a in ("N", "Anisocy") for v in al.values() for a in v)


def test_alias_precision_is_per_note_and_integers_match_exactly():
    # admission Hgb matches, two later Hgb values do not: per-occurrence precision 1/3, per-note 1.0
    labs = "___ BLOOD Hgb-10.1 ___ BLOOD Hgb-9.4 ___ BLOOD Hgb-8.8 HIV-1"
    notes = [{"hadm_id": h, "labs": labs} for h in range(30)]
    vals = pd.DataFrame([(h, it, v) for h in range(30) for it, v in ((51222, 10.1), (50912, 0.8))], columns=["hadm_id", "itemid", "v"])
    al = {k: [a for a, _, _ in v] for k, v in mine_lab_aliases(notes, vals).items()}
    assert al == {51222: ["Hgb"]}  # creatinine 0.8 is not "HIV-1"


def test_answer_leak_flags(cfg):
    rng = np.random.default_rng(0)
    Y = (rng.random((2000, 6)) < 0.1).astype(int)
    x = rng.random(2000) < 0.05
    hx = {"source": "history", "icd_version": "10", "code": "I13",
          "category_titles": "I130=Hypertensive heart and chronic kidney disease with heart failure | I1310=Hypertensive heart disease"}
    assert any(f.startswith("same_concept:heart_failure") for f in leak_flags(cfg, hx, x.astype(int), Y)[0])
    z = {"source": "history", "icd_version": "10", "code": "Z79", "category_titles": "Z7901=Long term (current) use of anticoagulants"}
    assert "administrative" in leak_flags(cfg, z, x.astype(int), Y)[0]
    assert "supply" in leak_flags(cfg, {"source": "medication", "code": "bag"}, x.astype(int), Y)[0]
    assert "routine" in leak_flags(cfg, {"source": "medication", "code": "sodium chloride"}, np.ones(2000, int), Y)[0]
    Y2 = Y.copy()
    Y2[x, 1] = 1  # drug given almost only to afib patients
    assert any(f.startswith("treatment:afib") for f in leak_flags(cfg, {"source": "medication", "code": "amiodarone"}, x.astype(int), Y2)[0])


def test_mined_history_words_never_name_a_diagnosis_or_drug(cfg):
    from qddx.vocab import extractability
    # 40 notes of positive admissions mention "hfpef", "warfarin" and "ckd"; 160 others mention none of them
    notes = [{"hadm_id": h, "pmh": "hfpef ckd warfarin" if h < 40 else "asthma", "meds": "", "labs": ""} for h in range(200)]
    meta = pd.DataFrame([{"item": "hx10:N18", "source": "history", "code": "N18", "title": "Chronic kidney disease"},
                         {"item": "rx:warfarin", "source": "medication", "code": "warfarin", "title": "warfarin"}])
    labvals = pd.DataFrame(columns=["hadm_id", "itemid", "v"])
    terms = extractability(cfg, notes, ["hx10:N18"], meta, {"hx10:N18": set(range(40))}, labvals)["hx10:N18"]["terms"]
    assert "ckd" in terms and not any(w in t for t in terms for w in ("hfpef", "warfarin"))
