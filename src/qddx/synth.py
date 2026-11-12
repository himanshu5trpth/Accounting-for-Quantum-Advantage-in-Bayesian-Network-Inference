"""Synthetic MIMIC-shaped hosp tables and synthetic admission notes (no patient data) for tests and dry runs."""
from __future__ import annotations

import csv
import gzip
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

FIRST = ["Zorblat", "Quenby", "Marvo", "Tilda", "Osric", "Yvaine", "Brannoc", "Elspeth"]
LAST = ["Fennimore", "Ashgrove", "Pellucid", "Wraxall", "Dunmore", "Halloway", "Quimby", "Strand"]
ICD10 = {"ckd": "N183", "diabetes": "E119", "copd": "J449", "heart_failure": "I5023", "afib": "I480",
         "uti": "N390", "pneumonia": "J189", "aki": "N179", "anemia": "D649"}
ICD9 = {"ckd": "5853", "diabetes": "25000", "copd": "496", "heart_failure": "4280", "afib": "42731",
        "uti": "5990", "pneumonia": "486", "aki": "5849", "anemia": "2859"}
DRUG = {"loop_diuretic": "Furosemide", "anticoagulant": "Apixaban", "beta_blocker": "Metoprolol Tartrate"}
NOTE_TERM = {"ckd": "chronic kidney disease", "diabetes": "type 2 diabetes", "copd": "COPD",
             "heart_failure": "congestive heart failure", "afib": "atrial fibrillation", "uti": "urinary tract infection",
             "pneumonia": "pneumonia", "aki": "acute kidney injury", "anemia": "anemia"}


def _sig(x):
    return 1 / (1 + np.exp(-x))


def patient(rng) -> tuple[dict, dict]:
    """One admission from a hidden generative model -> (binary states, lab values)."""
    s = {"ckd": rng.random() < 0.2, "diabetes": rng.random() < 0.3, "copd": rng.random() < 0.15}
    s["heart_failure"] = rng.random() < _sig(-2 + 1.5 * s["ckd"] + 0.8 * s["diabetes"])
    s["afib"] = rng.random() < 0.1 + 0.3 * s["heart_failure"]
    s["uti"] = rng.random() < 0.12
    s["pneumonia"] = rng.random() < 0.08 + 0.25 * s["copd"]
    s["aki"] = rng.random() < 0.06 + 0.3 * s["ckd"] + 0.2 * s["heart_failure"]
    s["anemia"] = rng.random() < 0.1 + 0.3 * s["ckd"]
    s["loop_diuretic"] = rng.random() < 0.04 + 0.75 * s["heart_failure"]
    s["anticoagulant"] = rng.random() < 0.03 + 0.7 * s["afib"]
    s["beta_blocker"] = rng.random() < 0.08 + 0.4 * s["heart_failure"] + 0.4 * s["afib"]
    v = {"creat": max(0.3, rng.normal(0.9 + 1.2 * s["ckd"] + 1.0 * s["aki"], 0.25)),
         "na": rng.normal(139 - 6 * s["heart_failure"], 2.5),
         "wbc": max(1.0, rng.normal(8 + 6 * s["pneumonia"] + 4 * s["uti"], 1.8)),
         "hgb": max(4.0, rng.normal(13.8 - 3.5 * s["anemia"] - 1.0 * s["ckd"], 1.0))}
    return {k: int(b) for k, b in s.items()}, v


def _write(path: Path, header: list[str], rows, gz: bool = True):
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if gz else open
    with opener(path, "wt", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def make_mimic(out: Path, n_subjects: int = 600, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    pats, adms, dxs, rxs, labs = [], [], [], [], []
    hadm, lid = 20000000, 1
    lab_items = {"creat": (50912, 0.5, 1.2), "na": (50983, 135, 145), "wbc": (51301, 4.0, 11.0), "hgb": (51222, 12.0, 16.0)}
    for i in range(n_subjects):
        sid = 10000000 + i
        age = int(rng.integers(16, 90))
        era = ["2008 - 2010", "2011 - 2013", "2014 - 2016", "2017 - 2019", "2020 - 2022"][i % 5]
        pats.append([sid, "F" if rng.random() < 0.5 else "M", age, 2150, era, ""])
        for _ in range(int(rng.integers(1, 5))):
            hadm += 1
            t = datetime(2150, 1, 1) + timedelta(days=int(rng.integers(0, 2000)), hours=int(rng.integers(0, 24)))
            adms.append([sid, hadm, t.isoformat(sep=" "), (t + timedelta(days=4)).isoformat(sep=" "), "", "URGENT"])
            s, v = patient(rng)
            seq = 1
            for k, code in ICD10.items():
                if s[k]:
                    icd9 = rng.random() < 0.3
                    dxs.append([sid, hadm, seq, ICD9[k] if icd9 else code, 9 if icd9 else 10])
                    seq += 1
            dxs.append([sid, hadm, seq, "I10", 10])
            for k, drug in DRUG.items():
                if s[k]:
                    rxs.append([sid, hadm, (t + timedelta(hours=float(rng.uniform(0, 20)))).isoformat(sep=" "), drug, "PO"])
            rxs.append([sid, hadm, (t + timedelta(hours=30)).isoformat(sep=" "), "Furosemide", "IV"])  # outside window
            if hadm % 3 == 0:  # a two-word drug name (its proposition id has a space)
                rxs.append([sid, hadm, (t + timedelta(hours=2)).isoformat(sep=" "), "Potassium Chloride", "PO"])
            for k, (item, lo, hi) in lab_items.items():
                if rng.random() < 0.85:
                    ct = t + timedelta(hours=float(rng.uniform(-6, 20)))
                    labs.append([lid, sid, hadm if rng.random() < 0.8 else "", item, ct.isoformat(sep=" "),
                                 f"{v[k]:.2f}", f"{v[k]:.2f}", lo if rng.random() < 0.9 else "", hi if rng.random() < 0.9 else ""])
                    lid += 1
                late = t + timedelta(hours=60)  # outside the admission window
                labs.append([lid, sid, hadm, item, late.isoformat(sep=" "), "99", "99", lo, hi])
                lid += 1
    h = out / "hosp"
    _write(h / "patients.csv.gz", ["subject_id", "gender", "anchor_age", "anchor_year", "anchor_year_group", "dod"], pats)
    _write(h / "admissions.csv" / "admissions.csv", ["subject_id", "hadm_id", "admittime", "dischtime", "deathtime", "admission_type"], adms, gz=False)
    _write(h / "diagnoses_icd.csv.gz", ["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"], dxs)
    _write(h / "prescriptions.csv.gz", ["subject_id", "hadm_id", "starttime", "drug", "route"], rxs)
    titles = {"N183": "Chronic kidney disease, stage 3 (moderate)", "E119": "Type 2 diabetes mellitus without complications",
              "J449": "Chronic obstructive pulmonary disease, unspecified", "I5023": "Acute on chronic systolic heart failure",
              "I480": "Paroxysmal atrial fibrillation", "N390": "Urinary tract infection, site not specified",
              "J189": "Pneumonia, unspecified organism", "N179": "Acute kidney failure, unspecified",
              "D649": "Anemia, unspecified", "I10": "Essential (primary) hypertension"}
    d_icd = [[c, 10, t] for c, t in titles.items()] + [[ICD9[k], 9, titles[ICD10[k]]] for k in ICD10]
    _write(h / "d_icd_diagnoses.csv.gz", ["icd_code", "icd_version", "long_title"], d_icd)
    _write(h / "d_labitems.csv.gz", ["itemid", "label", "fluid", "category"],
           [[50912, "Creatinine", "Blood", "Chemistry"], [50983, "Sodium", "Blood", "Chemistry"],
            [51301, "White Blood Cells", "Blood", "Hematology"], [51222, "Hemoglobin", "Blood", "Hematology"]])
    _write(h / "labevents.csv.gz", ["labevent_id", "subject_id", "hadm_id", "itemid", "charttime", "value", "valuenum",
                                    "ref_range_lower", "ref_range_upper"], labs)
    return out


def note_text(rng, s: dict, v: dict, name: tuple[str, str]) -> str:
    first, last = name
    hist = [NOTE_TERM[k] for k in ("ckd", "diabetes", "copd") if s[k]]
    if s["heart_failure"] and rng.random() < 0.6:
        hist.append("congestive heart failure")
    if s["afib"] and rng.random() < 0.6:
        hist.append("atrial fibrillation")
    pmh = "\n".join(f"- {h}" for h in hist) or "- hypertension"
    if not s["copd"] and rng.random() < 0.3:
        pmh += "\n- denies COPD"
    meds = [d for k, d in DRUG.items() if s[k]]
    dd = [NOTE_TERM[k] for k in ("heart_failure", "afib", "uti", "pneumonia", "aki", "anemia") if s[k]] or ["chest pain"]
    hpi = (f"Mr. {last} is a 70 year old with the history below presenting with "
           f"{'dyspnea and leg swelling' if s['heart_failure'] else 'fever' if s['pneumonia'] or s['uti'] else 'weakness'}.")
    if s["pneumonia"] and rng.random() < 0.5:
        hpi += " Chest x-ray concerning for possible pneumonia."
    return f"""
Name:  {first} {last}                     Unit No:   {rng.integers(10, 99)}-Q{rng.integers(1000, 9999)}

Admission Date:   01/0{rng.integers(1, 9)}/2150              Discharge Date:   01/1{rng.integers(1, 9)}/2150

Date of Birth:   02/03/2080             Sex:   M
Telephone:   (555) 123-{rng.integers(1000, 9999)}
Fax:   (555) 987-{rng.integers(1000, 9999)}
Email:   {first.lower()}.{last.lower()}@example.org
SSN:   900-12-{rng.integers(1000, 9999)}
Health Plan Beneficiary No:   HP{rng.integers(10**9, 10**10)}
Certificate/License No:   LC-QQ-{rng.integers(10**6, 10**7)}

Service:  MEDICINE

Allergies:
No Known Allergies

Attending:  Dr. Halvard Quist

Chief Complaint:
shortness of breath

Major Surgical or Invasive Procedure:
none

History of Present Illness:
{hpi}

Past Medical History:
{pmh}

Social History:
Lives with wife.

Family History:
Noncontributory.

Physical Exam:
ADMISSION PHYSICAL EXAM:
Vitals: stable
DISCHARGE PHYSICAL EXAM:
improved

Pertinent Results:
ADMISSION LABS:
01/01/2150 WBC-{v['wbc']:.1f} Hgb-{v['hgb']:.1f}
01/01/2150 Creat-{v['creat']:.1f} Na-{v['na']:.0f}
DISCHARGE LABS:
01/05/2150 WBC-7.0 Hgb-14.0 Creat-0.8 Na-140

Brief Hospital Course:
Treated for {', '.join(dd)}.

Medications on Admission:
{chr(10).join(f'{i + 1}. {m} 20 mg PO daily' for i, m in enumerate(meds)) or 'None'}

Discharge Medications:
1. Furosemide 40 mg PO daily

Discharge Disposition:
Home

Discharge Diagnosis:
{chr(10).join(dd)}

Discharge Condition:
stable

Discharge Instructions:
Call Dr. Quist with questions.

Followup Instructions:
___
"""


def make_notes(path: Path, n: int = 40, seed: int = 1) -> Path:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        s, v = patient(rng)
        text = note_text(rng, s, v, (FIRST[i % len(FIRST)], LAST[(i // 2) % len(LAST)]))
        rows.append([f"{10000000 + i // 2}-DS-{i}", 10000000 + i // 2, 30000000 + i, "DS", 1,
                     "2150-01-05 00:00:00", "2150-01-05 01:00:00", text])
    _write(path, ["note_id", "subject_id", "hadm_id", "note_type", "note_seq", "charttime", "storetime", "text"], rows, gz=False)
    return path


def make_mimic_notes(path: Path, admissions: list[tuple[int, int]], seed: int = 2) -> Path:
    """Redacted MIMIC-IV-Note-shaped discharge.csv.gz for given (subject_id, hadm_id) pairs."""
    import re as _re
    rng = np.random.default_rng(seed)
    rows = []
    for i, (sid, hadm) in enumerate(admissions):
        s, v = patient(rng)
        text = note_text(rng, s, v, ("___", "___"))
        text = _re.sub(r"\b\d{1,2}/\d{1,2}/\d{4}\b", "___", text)
        rows.append([f"{sid}-DS-{i}", sid, hadm, "DS", 1, "2150-01-05 00:00:00", "2150-01-05 01:00:00", text])
    _write(path, ["note_id", "subject_id", "hadm_id", "note_type", "note_seq", "charttime", "storetime", "text"], rows, gz=True)
    return path
