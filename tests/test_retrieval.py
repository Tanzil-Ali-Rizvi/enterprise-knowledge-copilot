from kcopilot.generation import extract_citations
from kcopilot.models import Chunk, ScoredChunk
from kcopilot.retrieval import reciprocal_rank_fusion


def _sc(cid: str, score: float = 1.0) -> ScoredChunk:
    chunk = Chunk(
        chunk_id=cid,
        doc_id="d",
        title="T",
        source=f"{cid}.md",
        section="S",
        text=f"text {cid}",
        allowed_roles=["all"],
        position=0,
    )
    return ScoredChunk(chunk=chunk, score=score, retriever="x")


def test_rrf_rewards_agreement_between_retrievers():
    fused = reciprocal_rank_fusion([[_sc("a"), _sc("b"), _sc("c")], [_sc("b"), _sc("c"), _sc("a")]])
    assert [f.chunk.chunk_id for f in fused][0] == "b"
    assert len(fused) == 3


def test_hybrid_finds_expected_source(copilot):
    user = copilot.users.get("priya")
    items, _ = copilot.retrieve("What is the hotel limit in Mumbai?", user)
    assert items[0].chunk.source == "finance/travel-expense-policy.md"
    assert items[0].chunk.section == "Hotel Limits"


def test_bm25_and_vector_modes_work(copilot):
    user = copilot.users.get("priya")
    for mode in ("bm25", "vector"):
        items, _ = copilot.retrieve("How many approvals does a pull request need?", user, mode=mode)
        assert any(i.chunk.source == "engineering/engineering-onboarding.md" for i in items)


def test_citations_drop_hallucinated_numbers():
    items = [_sc("a"), _sc("b")]
    text, cites = extract_citations("Fact one [1]. Fact two [7]. Fact three [2][1].", items)
    assert "[7]" not in text
    assert [c.index for c in cites] == [1, 2]


def test_hyphenated_terms_match_both_spellings(copilot):
    ans = copilot.ask("What is the guest wifi network called?", "priya")
    assert "NW-Guest" in ans.answer
