"""Admission notes: loading, de-identification, admission-time sections, discharge-diagnosis labels,
the rule-free baseline extractor, and the shared mapping from findings to graded memberships."""
from __future__ import annotations

import csv
import math
import re
from pathlib import Path

HEADERS = ["Name", "Unit No", "Admission Date", "Discharge Date", "Date of Birth", "Sex", "Service", "Allergies",
           "Attending", "Chief Complaint", "Major Surgical or Invasive Procedure", "History of Present Illness",
           "Past Medical History", "Social History", "Family History", "Physical Exam", "Pertinent Results",
           "Brief Hospital Course", "Medications on Admission", "Discharge Medications", "Discharge Disposition",
           "Facility", "Discharge Diagnosis", "Discharge Condition", "Discharge Instructions", "Followup Instructions"]
PHI_FIELDS = ["Name", "Unit No", "Admission Date", "Discharge Date", "Date of Birth", "Telephone", "Fax", "Email",
              "SSN", "Health Plan Beneficiary No", "Certificate/License No", "Attending", "Facility"]
_TITLES = {"dr", "md", "mr", "mrs", "ms", "miss", "phd", "do", "rn", "np", "pa"}
_SEC_RE = re.compile(r"^[ \t]*(" + "|".join(re.escape(h) for h in HEADERS) + r")[ \t]*:", re.M | re.I)
_LABELS = "|".join(re.escape(f) for f in PHI_FIELDS + ["Sex"])
# a value ends at the next header label (any spacing), at a wide gap before any label, or at the end of the line
_FIELD_RE = re.compile(r"(?<![A-Za-z])(" + "|".join(re.escape(f) for f in PHI_FIELDS) + r")[ \t]*:[ \t]*(.*?)"
                       r"(?=[ \t]+(?:" + _LABELS + r")[ \t]*:|[ \t]{2,}[A-Z][A-Za-z/ ]{1,30}:|$)", re.M)
_NOT_NAMES = _TITLES | {w.lower() for f in PHI_FIELDS + ["Sex"] for w in re.findall(r"[A-Za-z]+", f)} | \
    {"no", "on", "of", "the", "and", "in", "at", "to", "for", "with", "by"}
_PATTERNS = [(re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"), "[DATE]"),
             (re.compile(r"\b\d{4}-\d{2}-\d{2}\b"), "[DATE]"),
             (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
             (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
             (re.compile(r"\(?\b\d{3}\)?[ .-]?\d{3}[ .-]\d{4}\b"), "[PHONE]"),
             (re.compile(r"\b[A-Za-z]{1,3}[- ]?[A-Za-z]{0,3}[- ]?\d{6,}\b"), "[ID]")]
STATUSES = ("present", "absent", "possible", "unknown")


def load_notes(path: Path) -> list[dict]:
    csv.field_size_limit(1 << 30)
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


# ------------------------------------------------------------------------------------- privacy
def phi_values(text: str) -> tuple[list[str], list[str]]:
    """-> (header PHI values, name tokens) as written in the note header."""
    vals, names = [], []
    for field, val in _FIELD_RE.findall(text):
        val = val.strip()
        if len(val) < 2 or field.lower() == "sex":
            continue
        vals.append(val)
        if field in ("Name", "Attending"):
            names += [t for t in re.findall(r"[A-Za-z][A-Za-z'-]+", val) if t.lower().strip(".") not in _NOT_NAMES]
    return vals, sorted(set(names), key=len, reverse=True)


def deidentify(text: str) -> tuple[str, dict]:
    vals, names = phi_values(text)
    out = text
    for v in sorted(set(vals), key=len, reverse=True):
        out = out.replace(v, "[PHI]")
    for t in names:
        out = re.sub(rf"(?<![A-Za-z]){re.escape(t)}(?![A-Za-z])", "[NAME]", out, flags=re.I)
    for rx, tag in _PATTERNS:
        out = rx.sub(tag, out)
    return out, {"values": vals, "names": names}


def leaked(text: str, phi: dict) -> list[str]:
    """PHI items (header values and name tokens) still present in `text`."""
    low = text.lower()
    hits = [v for v in phi["values"] if v.lower() in low]
    hits += [t for t in phi["names"] if re.search(rf"(?<![a-z]){re.escape(t.lower())}(?![a-z])", low)]
    return hits


# ------------------------------------------------------------------------------------- sections
def sections(text: str) -> dict[str, str]:
    ms = list(_SEC_RE.finditer(text))
    out: dict[str, str] = {}
    for i, m in enumerate(ms):
        key = m.group(1).lower()
        body = text[m.end(): ms[i + 1].start() if i + 1 < len(ms) else len(text)]
        out[key] = (out.get(key, "") + "\n" + body).strip() if key in out else body.strip()
    return out


def evidence_text(text: str, cfg: dict) -> str:
    """Admission-time information only: allowed sections, each cut at the first discharge-time marker."""
    secs = sections(text)
    stops = [s.lower() for s in cfg["notes"]["stop_markers"]]
    parts = []
    for name in cfg["notes"]["evidence_sections"]:
        body = secs.get(name)
        if not body:
            continue
        low = body.lower()
        cut = min([low.find(s) for s in stops if low.find(s) >= 0], default=len(body))
        parts.append(f"{name.title()}:\n{body[:cut].strip()}")
    return "\n\n".join(parts)


def _clauses(text: str) -> list[str]:
    return [c for c in re.split(r"[\n.;]+", text) if c.strip()]


def _term_rx(term: str) -> re.Pattern:
    return re.compile(rf"(?<![a-z0-9]){re.escape(term.lower())}(?![a-z0-9])")


def mention(text: str, terms: list[str], cfg: dict) -> tuple[str, str]:
    """NegEx-lite: -> (status, quote). Affirmed beats hedged beats negated; 'unknown' if never mentioned."""
    neg = [_term_rx(t) for t in cfg["notes"]["negation_terms"]]
    hed = [_term_rx(t) if t[0].isalnum() else re.compile(re.escape(t)) for t in cfg["notes"]["hedge_terms"]]
    rank = {"present": 3, "possible": 2, "absent": 1, "unknown": 0}
    best, quote = "unknown", ""
    rxs = [_term_rx(t) for t in terms]
    for cl in _clauses(text):
        low = cl.lower()
        for rx in rxs:
            m = rx.search(low)
            if not m:
                continue
            window = " ".join(low[: m.start()].split()[-6:])
            st = "absent" if any(r.search(window) for r in neg) else \
                 "possible" if any(r.search(window) or r.search(low[m.end(): m.end() + 3]) for r in hed) else "present"
            if rank[st] > rank[best]:
                best, quote = st, cl.strip()[:200]
    return best, quote


def labels(text: str, cfg: dict) -> dict[str, int] | None:
    """Discharge diagnoses (ground truth) from the note's own 'Discharge Diagnosis' section; None if absent."""
    body = sections(text).get(cfg["notes"]["label_section"], "")
    if not body.strip():
        return None
    out = {}
    for k, p in cfg["propositions"].items():
        if p["kind"] == "diagnosis":
            out[k] = int(mention(body, p["label_terms"], cfg)[0] in ("present", "possible"))
    return out


# ------------------------------------------------------------------------------------- findings
def lab_value(text: str, keys: list[str]) -> tuple[float | None, str]:
    rx = re.compile(r"(?<![A-Za-z])(" + "|".join(re.escape(k) for k in keys) + r")\s*[-:=]\s*(\d+(?:\.\d+)?)", re.I)
    m = rx.search(text)
    if not m:
        return None, ""
    line = text[text.rfind("\n", 0, m.start()) + 1: text.find("\n", m.end()) if text.find("\n", m.end()) >= 0 else len(text)]
    return float(m.group(2)), line.strip()[:200]


def section_text(ev_text: str, names: list[str] | None) -> str:
    """The named admission-time sections of an evidence text (None = the whole text)."""
    if not names:
        return ev_text
    secs = sections(ev_text)
    return "\n".join(secs.get(n.lower(), "") for n in names)


def baseline_findings(ev_text: str, cfg: dict) -> dict[str, dict]:
    """Keyword + negation/hedge extractor (no language model); same schema as the agent's findings. A proposition's
    optional `note_section` list restricts where it is looked for (e.g. history only in Past Medical History, so a
    relative's disease or the current illness described in the HPI is not read as the patient's history)."""
    out = {}
    for k, p in cfg["propositions"].items():
        if p["kind"] in ("diagnosis", "demographic"):  # demographics come from the admission record, not the note
            continue
        text = section_text(ev_text, p.get("note_section"))
        if p["kind"] == "lab":
            v, q = lab_value(text, p["note_keys"])
            out[k] = {"status": "unknown" if v is None else "present", "value": v, "quote": q}
        else:
            terms = p["note_terms"] if p["kind"] == "history" else p["drug_terms"]
            st, q = mention(text, terms, cfg)
            out[k] = {"status": st, "value": None, "quote": q}
    return out


def lab_membership(x: float, lim: dict, width: float) -> float | None:
    """Graded abnormality: logistic ramp centred on the data-derived limit, scale = width x reference span."""
    lo, hi = lim["lower"], lim["upper"]
    edge = hi if lim["direction"] == "high" else lo
    if edge is None:  # no reference limit on that side (e.g. NT-proBNP has only an upper one): unknown
        return None
    scale = max(width * ((hi - lo) if lo is not None and hi is not None else abs(edge)), 1e-9)
    z = (x - hi) / scale if lim["direction"] == "high" else (lo - x) / scale
    m = 1.0 / (1.0 + math.exp(-max(min(z, 50), -50)))
    return 1.0 if m > 0.99 else 0.0 if m < 0.01 else round(m, 4)


def memberships(findings: dict[str, dict], cfg: dict, limits: dict, closed_world: bool = True,
                crisp: bool = False, structured: dict | None = None) -> dict[str, float | None]:
    """Findings -> memberships in [0, 1] (None = unknown). Closed world: undocumented history/medication = 0,
    mirroring how the knowledge base codes them; undocumented labs stay unknown (as in MIMIC). Demographic
    propositions take their value from `structured` (the admission record) and are unknown without it."""
    h = cfg["notes"]["hedge_membership"]
    out: dict[str, float | None] = {}
    for k, p in cfg["propositions"].items():
        if p["kind"] == "diagnosis":
            continue
        if p["kind"] == "demographic":
            v = (structured or {}).get(k)
            out[k] = None if v is None or v < 0 else float(v)
            continue
        f = findings.get(k) or {"status": "unknown", "value": None}
        st = f.get("status", "unknown")
        if p["kind"] == "lab" and f.get("value") is not None:
            m = lab_membership(float(f["value"]), limits[k], cfg["knowledge"]["fuzzy_width"])
        elif st == "present":
            m = 1.0
        elif st == "possible":
            m = h
        elif st == "absent":
            m = 0.0
        else:
            m = 0.0 if closed_world and p["kind"] != "lab" else None
        out[k] = None if m is None else float(m >= 0.5) if crisp else m
    return out
