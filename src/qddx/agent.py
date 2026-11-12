"""Agentic neural layer: a local LLM (Ollama) grounds the de-identified admission note into propositions,
then answers the engine's follow-up questions by searching the note with a tool.

Every finding must carry a verbatim quote from the note (and a lab value must appear in its quote);
anything that fails validation is discarded and becomes 'unknown'. The LLM never writes the output report.
"""
from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.request
from pathlib import Path

from .bn import Evidence, dx_entropy, info_gain
from .notes import HEADERS, STATUSES, baseline_findings, memberships, mention, section_text

PROMPT_VERSION = "v3"     # grounding prompt
FOLLOWUP_VERSION = "f4"   # follow-up protocol (separate, so grounding caches survive follow-up changes)
_FINDING = {"type": "object", "properties": {
    "id": {"type": "string"}, "status": {"type": "string", "enum": list(STATUSES)},
    "value": {"type": ["number", "null"]}, "quote": {"type": "string"}}, "required": ["id", "status", "value", "quote"]}
_SYN = {"present": ("present", "yes", "true", "documented", "positive", "confirmed", "mentioned"),
        "possible": ("possible", "probable", "likely", "suspected", "hedged", "uncertain", "questionable"),
        "absent": ("absent", "no", "false", "negative", "denied", "denies", "ruled out"),
        "notfound": ("not found", "not documented", "not mentioned", "not present", "none", "no matches", "not stated"),
        "unknown": ("unknown", "unclear", "n/a", "na")}


def normalize_status(s, kind: str) -> str | None:
    """Map the model's wording to a status; 'not found' means absent for history/medications (closed world,
    as in the knowledge base) and unknown for labs."""
    t = " ".join(str(s if s is not None else "").lower().replace("_", " ").split())
    if not t:
        return None
    for st, words in _SYN.items():
        if t in words:
            return ("unknown" if kind == "lab" else "absent") if st == "notfound" else st
    return None


GROUND_SCHEMA = {"type": "object", "properties": {"findings": {"type": "array", "items": _FINDING}}, "required": ["findings"]}
SEARCH_TOOL = {"type": "function", "function": {
    "name": "search_note", "description": "Search the de-identified admission note. Every query word must match a complete word on the same line. Use separate searches for synonyms; no single-word fallback.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "words to look for"}},
                   "required": ["query"]}}}
SYSTEM = ("You read a de-identified hospital admission note for a diagnostic engine. Report only what the note states "
          "about this patient at admission (history, home medications or medications given on arrival, admission labs). "
          "Every finding that is present, possible, or has a value must quote the note verbatim (copy the exact text, "
          "at most 200 characters). Use 'possible' for hedged statements, 'absent' for explicit negations, and "
          "'unknown' when the note does not say. An unhedged diagnosis in the patient's history or a named drug "
          "listed in home/admission medications or explicitly given on arrival counts as 'present', not 'possible'. "
          "Uncertainty about dose, indication, or severity alone does not make the named item merely possible. "
          "Reserve 'possible' for actual uncertainty about whether the patient has the finding. Family-history "
          "findings belong to the relative, not the patient. A discontinued historical drug is not automatically "
          "a current medication; distinguish it from treatment given on arrival. Answer with JSON only.")


def describe(k: str, p: dict) -> str:
    if p["kind"] == "lab":
        return f"- {k}: first admission {p['label'].split()[0]} value (put the number in 'value')"
    what = {"history": "past medical history of", "medication": "taking or given on arrival:"}[p["kind"]]
    return f"- {k}: {what} {p['label']}"


class LLMError(RuntimeError):
    """The model server answered but failed (e.g. HTTP 500 on a malformed tool call)."""


def assert_local(model: str) -> None:
    """Patient notes never leave the machine: refuse Ollama cloud models (names with a 'cloud' tag or suffix)."""
    tag = model.lower().rsplit("/", 1)[-1]
    if "cloud" in tag:
        raise ValueError(f"{model!r} is an Ollama cloud model; patient data may only go to local models (DUA)")


def ollama_info(url: str, models: list[str]) -> dict:
    """Server version and local model digests, recorded with every result for reproducibility."""
    def get(path):
        with urllib.request.urlopen(url.rstrip("/") + path, timeout=10) as r:
            return json.loads(r.read())
    try:
        tags = {m["name"]: m.get("digest") for m in get("/api/tags").get("models", [])}
        return {"version": get("/api/version").get("version"),
                "digests": {m: tags.get(m) or tags.get(m + ":latest") for m in models}}
    except (urllib.error.URLError, OSError, ValueError) as e:
        return {"version": None, "error": str(e)[:200]}


class OllamaClient:
    def __init__(self, url: str, model: str, timeout: int = 300, think=None, seed: int = 0, retries: int = 3,
                 num_ctx: int | None = None):
        assert_local(model)
        self.url, self.model, self.timeout, self.think, self.seed = url.rstrip("/"), model, timeout, think, seed
        self.retries, self.num_ctx = retries, num_ctx

    def chat(self, messages: list[dict], tools: list | None = None, fmt=None) -> dict:
        body = {"model": self.model, "messages": messages, "stream": False,
                "options": {"temperature": 0, "seed": self.seed}}
        if self.num_ctx:
            body["options"]["num_ctx"] = int(self.num_ctx)
        if tools:
            body["tools"] = tools
        if fmt is not None:
            body["format"] = fmt
        if self.think is not None:
            body["think"] = self.think
        last = ""
        for attempt in range(self.retries):
            body["options"]["seed"] = self.seed + attempt  # a server error is often one bad generation; resample
            req = urllib.request.Request(self.url + "/api/chat", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read())["message"]
            except urllib.error.HTTPError as e:
                last = f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}"
                if e.code < 500:
                    break
            except urllib.error.URLError as e:
                if not isinstance(e.reason, TimeoutError):
                    raise RuntimeError(f"Ollama not reachable at {self.url} ({e}). Start it (`ollama serve`, "
                                       f"`ollama pull {self.model}`) or set agent.backend: baseline.") from e
                last = f"timeout: {e}"
            except TimeoutError as e:
                last = f"timeout: {e}"
        raise LLMError(last)


def parse_json(content: str) -> dict:
    """First JSON object in the reply ({} if none)."""
    for cand in (content, (re.search(r"\{.*\}", content or "", re.S) or [None])[0]):
        try:
            r = json.loads(cand)
            if isinstance(r, dict):
                return r
        except (json.JSONDecodeError, TypeError):
            pass
    return {}


def _norm(s: str) -> str:
    return " ".join(str(s).lower().split())


def _patient_note(note: str) -> str:
    """Do not let a relative's listed history veto the patient's negative finding."""
    headings = "|".join(re.escape(h) for h in HEADERS)
    return re.sub(r"^[ \t]*Family History[ \t]*:.*?(?=^[ \t]*(?:" + headings + r")[ \t]*:|\Z)",
                  "", note, flags=re.I | re.M | re.S)


def validate(f: dict, note: str, cfg: dict) -> tuple[dict, str | None]:
    """-> (finding, reason it was rejected or None). Rejected findings become 'unknown'."""
    k, P = f.get("id"), cfg["propositions"]
    unk = {"status": "unknown", "value": None, "quote": ""}
    if k not in P or P[k]["kind"] in ("diagnosis", "demographic"):
        return unk, "unknown proposition"
    st = normalize_status(f.get("status"), P[k]["kind"])
    if st is None:
        return unk, "bad status"
    if st == "absent" and P[k]["kind"] in ("history", "medication"):
        terms = (P[k].get("note_terms") or []) + (P[k].get("drug_terms") or [])
        where = section_text(note, P[k]["note_section"]) if P[k].get("note_section") else _patient_note(note)
        if mention(where, terms, cfg)[0] in ("present", "possible"):
            return unk, "vocabulary mention conflicts with absent"
    v = f.get("value")
    if P[k]["kind"] != "lab":
        v = None
    elif v is not None:
        try:
            v = float(v)
        except (TypeError, ValueError):
            return unk, "non-numeric value"
    q = str(f.get("quote") or "")
    if st in ("present", "possible") or v is not None:
        if not q.strip() or _norm(q) not in _norm(note):
            return unk, "quote not in note"
        if v is not None and not any(abs(float(x) - v) < 1e-9 for x in re.findall(r"\d+(?:\.\d+)?", q)):
            return unk, "value not in quote"
    return {"status": st, "value": v, "quote": q[:200]}, None


def search_note(note: str, query: str, limit: int = 8) -> str:
    words = re.findall(r"[a-z0-9]+", query.lower())
    lines = [ln.strip() for ln in note.splitlines() if ln.strip()]
    patterns = [re.compile(rf"(?<![a-z0-9]){re.escape(w)}(?![a-z0-9])") for w in words]
    hits = [ln for ln in lines if patterns and all(rx.search(ln.lower()) for rx in patterns)]
    return "\n".join(h[:300] for h in hits[:limit]) or "no matches"


class Agent:
    def __init__(self, client, cfg: dict, cache_dir: Path | None = None, protocol: str = "tools"):
        """protocol: 'tools' (native tool calling) or 'text' (the model asks for searches as JSON {"search": ...},
        for models without tool calling; same validation, cached separately)."""
        self.client, self.cfg, self.cache, self.protocol = client, cfg, cache_dir, protocol
        self.model = getattr(client, "model", "fake")

    def _cached(self, key: str, fn):
        if self.cache is None:
            return fn()
        p = self.cache / f"{key}.json"
        if p.exists():
            return json.loads(p.read_text())
        out = fn()
        if not out.get("error"):  # failures are not cached, so a rerun retries them
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(out))
        return out

    def _key(self, note: str, *extra) -> str:
        return hashlib.sha256("|".join([PROMPT_VERSION, self.model, note, *extra]).encode()).hexdigest()[:24]

    def ground(self, note: str) -> dict:
        """One pass over the note -> {prop: finding}, plus the rejection log."""
        def run():
            P = self.cfg["propositions"]
            props = "\n".join(describe(k, p) for k, p in P.items() if p["kind"] not in ("diagnosis", "demographic"))
            msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content":
                    f"Propositions:\n{props}\n\nReturn {{\"findings\": [{{\"id\", \"status\", \"value\", \"quote\"}}]}} "
                    f"with one entry per proposition.\n\nNOTE:\n<<<\n{note[: self.cfg['agent']['max_chars']]}\n>>>"}]
            try:
                got = parse_json(self.client.chat(msgs, fmt=GROUND_SCHEMA).get("content", "")).get("findings")
                if not got:  # some models return nothing under a schema; retry with plain JSON mode
                    got = parse_json(self.client.chat(msgs, fmt="json").get("content", "")).get("findings") or []
            except LLMError as e:
                return {"findings": {}, "rejected": [], "error": str(e)}
            found, rejected = {}, []
            for f in got if isinstance(got, list) else []:
                if not isinstance(f, dict):
                    continue
                ok, why = validate(f, note, self.cfg)
                if why:
                    rejected.append({"id": f.get("id"), "reason": why})
                elif f["id"] not in found or found[f["id"]]["status"] == "unknown":
                    found[f["id"]] = ok
            return {"findings": found, "rejected": rejected}
        return self._cached(self._key(note, "ground"), run)

    def investigate(self, note: str, k: str) -> dict:
        """Engine-requested follow-up on one proposition; the agent sees only what its searches return.
        It must search at least `min_searches` times with different wordings before answering."""
        def run():
            a, p = self.cfg["agent"], self.cfg["propositions"][k]
            hints = ", ".join((p.get("note_keys") or []) + (p.get("note_terms") or []) + (p.get("drug_terms") or []))
            absent = ("If the searches find nothing, answer status 'absent' with an empty quote."
                      if p["kind"] != "lab" else "If no value is found, answer status 'unknown'.")
            text = self.protocol == "text"
            how = ('search the note by replying with the JSON object {"search": "<words>"} (one search per reply; you '
                   'will get the matching lines back)' if text else "search the note with search_note")
            ask = (f"[id={k}] The diagnostic engine asks: {describe(k, p)[2:]}. To find out, {how}: at least "
                   f"{a['min_searches']} and at most {a['max_tool_calls']} searches, each with a different wording "
                   f"(abbreviations, brand and generic names, related terms; e.g. {hints}). {absent} Then answer with JSON "
                   f"{{\"id\": \"{k}\", \"status\", \"value\", \"quote\"}} only, quoting verbatim from the search results.")
            msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": ask}]
            calls, queries, nudged = 0, [], False
            for _ in range(a["max_tool_calls"] + 3):
                try:
                    m = self.client.chat(msgs, tools=[SEARCH_TOOL] if calls < a["max_tool_calls"] and not text else None)
                except LLMError as e:
                    return {"finding": {"status": "unknown", "value": None, "quote": ""}, "rejected": "llm error",
                            "queries": queries, "error": str(e)}
                msgs.append({k2: v for k2, v in m.items() if k2 in ("role", "content", "tool_calls")})
                tcs = m.get("tool_calls") or []
                if text and calls < a["max_tool_calls"]:
                    req = parse_json(m.get("content", ""))
                    if isinstance(req.get("search"), str):
                        tcs = [{"function": {"name": "search_note", "arguments": {"query": req["search"]}}}]
                if tcs and calls < a["max_tool_calls"]:
                    for tc in tcs[: a["max_tool_calls"] - calls]:
                        args = tc.get("function", {}).get("arguments", {})
                        try:
                            q = str((json.loads(args) if isinstance(args, str) else args).get("query", ""))
                        except (json.JSONDecodeError, AttributeError):
                            q = ""
                        queries.append(q)
                        calls += 1
                        res = search_note(note, q)
                        msgs.append({"role": "user", "content": f"Search results for '{q}':\n{res}"} if text else
                                    {"role": "tool", "tool_name": "search_note", "content": res})
                    continue
                if calls < a["min_searches"] and not nudged:
                    nudged = True
                    msgs.append({"role": "user", "content": "Search again with a different wording before answering."})
                    continue
                break
            f = parse_json(m.get("content", ""))
            if normalize_status(f.get("status"), p["kind"]) is None:
                # gpt-oss sometimes answers through the tool channel or in prose: ask once more, no tools, JSON schema
                msgs.append({"role": "user", "content": f"Answer now with the JSON object for {k} only."})
                try:
                    m = self.client.chat(msgs, fmt=_FINDING)
                except LLMError as e:
                    return {"finding": {"status": "unknown", "value": None, "quote": ""}, "rejected": "llm error",
                            "queries": queries, "error": str(e)}
                f = parse_json(m.get("content", ""))
            f["id"] = k
            ok, why = validate(f, note, self.cfg)
            out = {"finding": ok, "rejected": why, "queries": queries}
            if why:
                out["raw"] = str(m.get("content", ""))[:300]
            return out
        extra = ("investigate", FOLLOWUP_VERSION, k) + (("text",) if self.protocol == "text" else ())
        return self._cached(self._key(note, *extra), run)

    def run(self, note: str, bn, limits: dict, loop: bool = True) -> dict:
        """Grounding, then up to max_steps engine-chosen follow-ups (largest expected information gain over the
        diagnoses among still-unknown or graded findings)."""
        cfg = self.cfg
        g = self.ground(note)
        errors = [f"ground: {g['error']}"] if g.get("error") else []
        if errors:  # grounding failed after retries: fall back to the keyword extractor for this note (counted)
            g = {"findings": baseline_findings(note, cfg), "rejected": []}
        findings = dict(g["findings"])
        dx = [k for k, p in cfg["propositions"].items() if p["kind"] == "diagnosis"]
        trace, asked = [], set()
        for _ in range(cfg["agent"]["max_steps"] if loop else 0):
            m = memberships(findings, cfg, limits, closed_world=False)
            cand = [k for k, v in m.items() if k not in asked and (v is None or 0.0 < v < 1.0)
                    and cfg["propositions"][k]["kind"] != "demographic"]
            if not cand:
                break
            gains = info_gain(bn, Evidence.from_memberships(m), cand, dx)
            best = max(gains, key=gains.get)
            if gains[best] < cfg["agent"]["min_info_gain"]:
                break
            asked.add(best)
            r = self.investigate(note, best)
            before = memberships(findings, cfg, limits)[best]
            if r["finding"]["status"] != "unknown":
                findings[best] = r["finding"]
            if r.get("error"):
                errors.append(f"investigate {best}: {r['error']}")
            after = memberships(findings, cfg, limits, closed_world=False)
            trace.append({"asked": best, "info_gain": gains[best], "queries": r["queries"],
                          "status": r["finding"]["status"], "rejected": r["rejected"],
                          "changed": memberships(findings, cfg, limits)[best] != before,
                          # analysis only (selection unchanged): every candidate's expected gain, realised entropy
                          "candidate_gains": {c: float(v) for c, v in gains.items()},
                          "entropy_before": dx_entropy(bn, Evidence.from_memberships(m), dx),
                          "entropy_after": dx_entropy(bn, Evidence.from_memberships(after), dx)})
        return {"findings": findings, "ground_findings": g["findings"], "rejected": g["rejected"], "trace": trace,
                "errors": errors}


class FakeClient:
    """Deterministic stand-in for tests and dry runs: answers from the baseline extractor, through the same
    JSON / tool-calling protocol (two searches per follow-up)."""
    model = "fake"

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def chat(self, messages, tools=None, fmt=None):
        user = messages[1]["content"]
        if "NOTE:\n<<<\n" in user:
            note = user.split("NOTE:\n<<<\n", 1)[1].rsplit("\n>>>", 1)[0]
            fs = [{"id": k, **v} for k, v in baseline_findings(note, self.cfg).items()]
            return {"role": "assistant", "content": json.dumps({"findings": fs})}
        k = re.search(r"\[id=(\w+)\]", user).group(1)
        text = '{"search"' in user
        tool_out = [m["content"] for m in messages if m.get("role") == "tool"] + \
                   [m["content"].split("\n", 1)[1] for m in messages if m.get("role") == "user" and m["content"].startswith("Search results for")]
        p = self.cfg["propositions"][k]
        terms = (p.get("note_keys") or []) + (p.get("note_terms") or []) + (p.get("drug_terms") or [])
        if (tools or (text and fmt is None)) and len(tool_out) < min(2, len(terms)):
            q = terms[len(tool_out)]
            if text:
                return {"role": "assistant", "content": json.dumps({"search": q})}
            return {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "search_note", "arguments": {"query": q}}}]}
        f = baseline_findings("\n".join(o for o in tool_out if o != "no matches"), self.cfg)[k]
        if f["status"] == "unknown" and p["kind"] != "lab":
            f = {**f, "status": "absent"}
        return {"role": "assistant", "content": json.dumps({"id": k, **f})}
