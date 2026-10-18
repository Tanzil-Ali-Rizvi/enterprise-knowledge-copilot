"""Document loading and heading-aware chunking.

Supported formats: Markdown (.md), plain text (.txt) and PDF (.pdf).
Markdown/text files may start with a simple front-matter block:

    ---
    title: Leave Policy
    department: HR
    access: all            # comma-separated roles, or "all"
    updated: 2026-04-01
    ---
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from .guardrails import sanitize
from .models import Chunk, Document

SUPPORTED = {".md", ".txt", ".pdf"}
_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_HEADING = re.compile(r"^(#{1,4})\s+(.+?)\s*$", re.MULTILINE)


def _parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    match = _FRONT_MATTER.match(raw)
    if not match:
        return {}, raw
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip().lower()] = value.split("#", 1)[0].strip()
    return meta, raw[match.end() :]


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def load_document(path: Path, root: Path) -> Document:
    raw = _read_pdf(path) if path.suffix.lower() == ".pdf" else path.read_text(encoding="utf-8")
    meta, body = _parse_front_matter(raw)
    body, _ = sanitize(body, str(path.name))
    rel = path.relative_to(root).as_posix()
    roles = [r.strip().lower() for r in meta.get("access", "all").split(",") if r.strip()] or ["all"]
    title = meta.get("title") or path.stem.replace("-", " ").replace("_", " ").title()
    return Document(
        doc_id=hashlib.sha1(rel.encode()).hexdigest()[:12],
        title=title,
        source=rel,
        text=body.strip(),
        department=meta.get("department", "general"),
        allowed_roles=roles,
        updated=meta.get("updated"),
    )


def load_directory(root: Path) -> list[Document]:
    root = Path(root)
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)
    return [load_document(p, root) for p in files]


def _sections(text: str) -> list[tuple[str, str]]:
    """Split markdown into (heading path, body) pairs so every chunk knows where it came from."""
    matches = list(_HEADING.finditer(text))
    if not matches:
        return [("", text)]
    sections: list[tuple[str, str]] = []
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(("", preamble))
    stack: list[tuple[int, str]] = []
    for i, m in enumerate(matches):
        level, heading = len(m.group(1)), m.group(2).strip()
        stack = [(lvl, h) for lvl, h in stack if lvl < level] + [(level, heading)]
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end() : end].strip()
        if body:
            # Skip the top-level document title in the path; it duplicates the doc title.
            path = " > ".join(h for lvl, h in stack if lvl > 1) or heading
            sections.append((path, body))
    return sections


def _window(words: list[str], size: int, overlap: int) -> list[list[str]]:
    if len(words) <= size:
        return [words]
    step = max(1, size - overlap)
    windows = []
    for start in range(0, len(words), step):
        windows.append(words[start : start + size])
        if start + size >= len(words):
            break
    return windows


def chunk_document(doc: Document, size: int = 180, overlap: int = 40) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section, body in _sections(doc.text):
        for piece in _window(body.split(), size, overlap):
            text = " ".join(piece)
            pos = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}-{pos:03d}",
                    doc_id=doc.doc_id,
                    title=doc.title,
                    source=doc.source,
                    section=section,
                    text=text,
                    allowed_roles=doc.allowed_roles,
                    position=pos,
                )
            )
    return chunks
