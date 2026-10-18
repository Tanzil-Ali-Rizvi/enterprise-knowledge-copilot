from fastapi.testclient import TestClient

from kcopilot import api
from kcopilot.evaluation import check_gates, evaluate, load_cases
from kcopilot.llm import ABSTAIN_MESSAGE, LLMResult


def test_answer_has_citations_and_timings(copilot):
    ans = copilot.ask("How many days of earned leave do employees get each year?", "priya")
    assert "24" in ans.answer
    assert ans.citations and ans.citations[0].source == "hr/leave-policy.md"
    assert {"retrieve_ms", "rerank_ms", "generate_ms", "total_ms"} <= ans.timings_ms.keys()


def test_out_of_scope_abstains_without_llm_call(copilot):
    ans = copilot.ask("What is the capital of France?", "priya")
    assert ans.abstained and ans.answer == ABSTAIN_MESSAGE
    assert ans.timings_ms["generate_ms"] == 0.0


def test_llm_abstention_clears_citations(settings):
    from kcopilot.pipeline import Copilot

    class AlwaysAbstain:
        name = "stub"

        def generate(self, *a, **k):
            return LLMResult(ABSTAIN_MESSAGE + " [1]")

    cp = Copilot(settings=settings, llm=AlwaysAbstain(), persist=False)
    ans = cp.ask("How many days of earned leave do employees get each year?", "priya")
    assert ans.abstained and ans.citations == []


def test_eval_gates_pass(copilot):
    result = evaluate(copilot, load_cases("evals/golden.jsonl"))
    assert check_gates(result["metrics"]) == []
    assert result["metrics"]["acl_leak_rate"] == 0.0


def test_api_end_to_end(copilot):
    api._state["copilot"] = copilot
    client = TestClient(api.app)  # no context manager: reuse the session copilot
    assert client.get("/health").json()["status"] == "ok"

    r = client.post("/ask", json={"question": "What is the hotel limit in Mumbai?"}, headers={"X-User-Id": "priya"})
    assert r.status_code == 200 and "9,000" in r.json()["answer"]

    assert client.post("/ask", json={"question": "hello"}, headers={"X-User-Id": "nobody"}).status_code == 401
    assert client.post("/ask", json={"question": "hi there"}).status_code == 422
    assert client.post("/admin/reindex", headers={"X-User-Id": "priya"}).status_code == 403

    docs = client.get("/documents", headers={"X-User-Id": "priya"}).json()
    sources = {d["source"] for d in docs}
    assert "exec/q3-2026-board-summary.md" not in sources
    assert "hr/compensation-bands-2026.md" not in sources
    assert "engineering/incident-response-runbook.md" in sources
