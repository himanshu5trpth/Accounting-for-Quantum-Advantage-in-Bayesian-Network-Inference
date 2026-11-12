from qddx.agent import Agent, FakeClient, search_note, validate
from qddx.bn import BN, Struct
from qddx import notes as N
from qddx.synth import make_notes

import numpy as np


def test_validation_rejects_fabrication(cfg):
    note = "Past Medical History:\n- chronic kidney disease\nCreat-2.4"
    assert validate({"id": "ckd", "status": "present", "value": None, "quote": "chronic kidney disease"}, note, cfg)[1] is None
    assert validate({"id": "ckd", "status": "present", "value": None, "quote": "ESRD on dialysis"}, note, cfg)[1] == "quote not in note"
    assert validate({"id": "creat_high", "status": "present", "value": 3.1, "quote": "Creat-2.4"}, note, cfg)[1] == "value not in quote"
    assert validate({"id": "aki", "status": "present", "value": None, "quote": "Creat-2.4"}, note, cfg)[1] == "unknown proposition"
    ok, why = validate({"id": "copd", "status": "absent", "value": None, "quote": ""}, note, cfg)
    assert why is None and ok["status"] == "absent"


def test_search_tool():
    note = "PMH:\n- CKD stage 3\nMeds:\nfurosemide 40 mg"
    assert search_note(note, "furosemide dose") == "no matches"
    assert "furosemide" in search_note(note, "furosemide 40")
    assert search_note(note, "warfarin") == "no matches"


def test_agent_loop_with_fake_client(cfg, tmp_path):
    names = [k for k in cfg["propositions"]]
    bn = BN(Struct(names, [()] * len(names)), [np.array([[0.7, 0.3]])] * len(names))
    lim = {k: {"lower": 1.0, "upper": 2.0, "direction": p["direction"]} for k, p in cfg["propositions"].items() if p["kind"] == "lab"}
    r = N.load_notes(make_notes(tmp_path / "n.csv", n=2))[0]
    ev = N.evidence_text(N.deidentify(r["text"])[0], cfg)
    agent = Agent(FakeClient(cfg), cfg, tmp_path / "cache")
    out = agent.run(ev, bn, lim)
    assert set(out["findings"]) <= set(names) and out["rejected"] == []
    assert list((tmp_path / "cache").glob("*.json"))
    assert agent.run(ev, bn, lim) == out  # cached and deterministic


def test_server_errors_are_retried_then_tolerated(cfg, tmp_path):
    import json as _json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from qddx.agent import LLMError, OllamaClient
    hits = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            hits.append(1)
            ok = len(hits) == 2
            self.send_response(200 if ok else 500)
            self.end_headers()
            self.wfile.write(_json.dumps({"message": {"role": "assistant", "content": "{}"}}).encode() if ok else b"error parsing tool call")

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    c = OllamaClient(f"http://127.0.0.1:{srv.server_port}", "m", timeout=10)
    assert c.chat([{"role": "user", "content": "x"}])["content"] == "{}" and len(hits) == 2
    hits.clear()
    hits.append(1)  # next calls all fail (count never equals 2 again)
    hits.append(1)
    try:
        c.chat([{"role": "user", "content": "x"}])
        raise AssertionError("expected LLMError")
    except LLMError as e:
        assert "500" in str(e)

    class Flaky(FakeClient):
        def chat(self, messages, tools=None, fmt=None):
            if "NOTE:\n<<<\n" not in messages[1]["content"]:
                raise LLMError("HTTP 500: error parsing tool call")
            return super().chat(messages, tools, fmt)

    names = list(cfg["propositions"])
    bn = BN(Struct(names, [()] * len(names)), [np.array([[0.7, 0.3]])] * len(names))
    lim = {k: {"lower": 1.0, "upper": 2.0, "direction": p["direction"]} for k, p in cfg["propositions"].items() if p["kind"] == "lab"}
    ev = N.evidence_text(N.load_notes(make_notes(tmp_path / "n.csv", n=2))[0]["text"], cfg)
    cfg["agent"]["min_info_gain"] = -1.0  # force follow-ups on this uninformative network
    out = Agent(Flaky(cfg), cfg, tmp_path / "cache").run(ev, bn, lim)
    assert out["errors"] and all(t["rejected"] == "llm error" for t in out["trace"])
    assert len(list((tmp_path / "cache").glob("*.json"))) == 1  # only the successful grounding is cached
    srv.shutdown()


def test_status_wording_and_final_json_answer(cfg, tmp_path):
    from qddx.agent import normalize_status
    assert normalize_status("Not documented", "history") == "absent" and normalize_status("not found", "lab") == "unknown"
    assert normalize_status("Present", "medication") == "present" and normalize_status("banana", "history") is None

    class Prose(FakeClient):  # answers in prose first; must be asked again for JSON
        def chat(self, messages, tools=None, fmt=None):
            out = super().chat(messages, tools, fmt)
            if fmt is None and not out.get("tool_calls") and "NOTE:\n<<<\n" not in messages[1]["content"]:
                return {"role": "assistant", "content": "The searches show nothing relevant."}
            return out

    note = "Past Medical History:\n- chronic kidney disease"
    r = Agent(Prose(cfg), cfg, None).investigate(note, "ckd")
    assert r["finding"]["status"] == "present" and r["rejected"] is None and len(r["queries"]) >= 2


def test_search_requires_every_complete_query_word():
    note = "chronic foley\nchronic prednisone\nrenal artery stenosis\nchronic renal disease\nNABS"
    assert search_note(note, "chronic renal") == "chronic renal disease"
    assert search_note(note, "chronic kidney disease") == "no matches"
    assert search_note(note, "Na") == "no matches"
    assert search_note("adrenal incidentaloma", "renal") == "no matches"
    assert search_note(note, "") == "no matches"


def test_absence_guard_covers_the_whole_configured_vocabulary(cfg):
    cases = [("anticoagulant", "Rivaroxaban 20 mg daily"),
             ("beta_blocker", "Carvedilol 12.5 mg twice daily"),
             ("loop_diuretic", "Torsemide 40 mg daily"),
             ("loop_diuretic", "Lasix 40 mg daily"),
             ("ckd", "ESRD not on dialysis"), ("copd", "COPD")]
    for k, note in cases:
        finding, reason = validate({"id": k, "status": "absent", "value": None, "quote": ""}, note, cfg)
        assert finding["status"] == "unknown" and reason == "vocabulary mention conflicts with absent"


def test_absence_guard_preserves_explicit_negation_and_family_context(cfg):
    for note in ("Past Medical History:\nNo CKD", "Past Medical History:\nNo renal disease\nFamily History:\nMother had CKD"):
        finding, reason = validate({"id": "ckd", "status": "absent", "value": None, "quote": ""}, note, cfg)
        assert reason is None and finding["status"] == "absent"
    finding, reason = validate({"id": "ckd", "status": "absent", "value": None, "quote": ""},
                               "Family History:\nMother had CKD\nPast Medical History:\nCKD stage 3", cfg)
    assert reason == "vocabulary mention conflicts with absent"


def test_absence_guard_does_not_erase_a_hedged_finding(cfg):
    finding, reason = validate({"id": "copd", "status": "absent", "value": None, "quote": ""},
                               "Past Medical History:\nPossible COPD", cfg)
    assert reason == "vocabulary mention conflicts with absent"


def test_grounding_prompt_distinguishes_listing_from_hedging():
    from qddx.agent import SYSTEM, PROMPT_VERSION, FOLLOWUP_VERSION
    assert "listed" in SYSTEM.lower() and "present" in SYSTEM.lower()
    assert PROMPT_VERSION != "v2" and FOLLOWUP_VERSION != "f3"


def test_preflight_detects_missing_tools_and_think(cfg):
    import json as _json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from qddx.experiments import preflight

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = _json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            err = ("model does not support thinking" if "think" in body else
                   "model does not support tools" if body.get("tools") else None)
            self.send_response(400 if err else 200)
            self.end_headers()
            self.wfile.write(_json.dumps({"error": err} if err else {"message": {"role": "assistant", "content": "{}"}}).encode())

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    cfg["agent"]["url"] = f"http://127.0.0.1:{srv.server_port}"
    info = preflight(cfg, "some-medical-model")
    assert info == {"ok": True, "think": None, "tools": False, "reason": None}
    srv.shutdown()


def test_cloud_models_refused_and_num_ctx_sent():
    import json as _json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import pytest
    from qddx.agent import OllamaClient, assert_local, ollama_info
    for bad in ("gpt-oss:120b-cloud", "kimi-k2.6:cloud", "deepseek-v4-flash:0731-cloud"):
        with pytest.raises(ValueError):
            assert_local(bad)
        with pytest.raises(ValueError):
            OllamaClient("http://127.0.0.1:1", bad)
    assert_local("medgemma:27b")
    seen = []

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            out = {"version": "0.40.13"} if self.path == "/api/version" else {"models": [{"name": "m:latest", "digest": "abc"}]}
            self.send_response(200)
            self.end_headers()
            self.wfile.write(_json.dumps(out).encode())

        def do_POST(self):
            seen.append(_json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(_json.dumps({"message": {"role": "assistant", "content": "{}"}}).encode())

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}"
    OllamaClient(url, "m", num_ctx=16384).chat([{"role": "user", "content": "x"}])
    assert seen[-1]["options"]["num_ctx"] == 16384
    assert ollama_info(url, ["m"]) == {"version": "0.40.13", "digests": {"m": "abc"}}
    srv.shutdown()


def test_text_search_protocol_for_models_without_tools(cfg):
    note = "Past Medical History:\n- chronic kidney disease\nMedications on Admission:\n1. Apixaban 5 mg"
    a = Agent(FakeClient(cfg), cfg, None, protocol="text")
    r = a.investigate(note, "ckd")
    assert r["finding"]["status"] == "present" and len(r["queries"]) >= 2 and r["rejected"] is None
    r2 = a.investigate(note, "anticoagulant")  # fake searches warfarin, coumadin only: its "absent" is vetoed
    assert r2["finding"]["status"] == "unknown" and r2["rejected"] == "vocabulary mention conflicts with absent"
