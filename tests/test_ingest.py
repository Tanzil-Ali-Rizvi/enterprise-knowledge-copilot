from pathlib import Path

from kcopilot.ingest import chunk_document, load_directory, load_document

ROOT = Path(__file__).resolve().parents[1] / "data" / "corpus"


def test_front_matter_parsed():
    doc = load_document(ROOT / "exec" / "q3-2026-board-summary.md", ROOT)
    assert doc.title == "Q3 FY2026 Board Summary"
    assert doc.allowed_roles == ["exec"]
    assert doc.source == "exec/q3-2026-board-summary.md"
    assert not doc.text.startswith("---")


def test_multi_role_access_list():
    doc = load_document(ROOT / "hr" / "compensation-bands-2026.md", ROOT)
    assert doc.allowed_roles == ["hr", "exec"]


def test_chunks_keep_section_paths_and_roles():
    doc = load_document(ROOT / "hr" / "leave-policy.md", ROOT)
    chunks = chunk_document(doc)
    sections = {c.section for c in chunks}
    assert {"Earned Leave", "Sick Leave", "Parental Leave"} <= sections
    assert all(c.allowed_roles == ["all"] for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_long_sections_are_windowed_with_overlap():
    doc = load_document(ROOT / "hr" / "leave-policy.md", ROOT)
    doc = doc.model_copy(update={"text": "# T\n## S\n" + " ".join(f"w{i}" for i in range(500))})
    chunks = chunk_document(doc, size=100, overlap=20)
    assert len(chunks) == 6
    assert chunks[0].text.split()[-20:] == chunks[1].text.split()[:20]


def test_whole_corpus_loads():
    docs = load_directory(ROOT)
    assert len(docs) >= 12
    assert len({d.doc_id for d in docs}) == len(docs)
