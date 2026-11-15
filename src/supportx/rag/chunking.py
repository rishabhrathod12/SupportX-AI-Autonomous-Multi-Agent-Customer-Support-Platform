"""Split Markdown documents into chunks that keep track of where they came from."""

import re
from dataclasses import dataclass

_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_BLANK_LINE = re.compile(r"\n\s*\n")


@dataclass(frozen=True)
class Chunk:
    """A piece of a document. `heading` is the section path ('Refund Policy > Return window')."""

    id: str
    source: str
    heading: str
    text: str

    @property
    def embedding_text(self) -> str:
        """What gets embedded: the heading gives the chunk context (a lone 'Yes' means nothing)."""
        return f"{self.heading}\n{self.text}" if self.heading else self.text


def split_markdown(text: str, source: str, max_chars: int = 800, overlap: int = 100) -> list[Chunk]:
    """Split by headings first, then pack paragraphs up to `max_chars`.

    A single paragraph longer than `max_chars` is cut into overlapping windows so that a
    sentence at the cut point still appears whole in one of the pieces.
    """
    if max_chars < 50:
        raise ValueError("max_chars must be at least 50")
    if not 0 <= overlap <= max_chars // 2:
        raise ValueError("overlap must be between 0 and max_chars // 2")

    chunks: list[Chunk] = []
    for heading, body in _sections(text):
        for piece in _pack(body, max_chars, overlap):
            chunks.append(
                Chunk(id=f"{source}#{len(chunks)}", source=source, heading=heading, text=piece)
            )
    return chunks


def _sections(text: str) -> list[tuple[str, str]]:
    """(heading path, body) for every section that has body text."""
    sections: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []  # (level, title)
    body: list[str] = []

    def flush() -> None:
        content = "\n".join(body).strip()
        if content:
            sections.append((" > ".join(title for _, title in stack), content))
        body.clear()

    for line in text.splitlines():
        match = _HEADING.match(line)
        if match is None:
            body.append(line)
            continue
        flush()
        level = len(match.group(1))
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, match.group(2)))
    flush()
    return sections


def _pack(body: str, max_chars: int, overlap: int) -> list[str]:
    pieces: list[str] = []
    current = ""
    for paragraph in (p.strip() for p in _BLANK_LINE.split(body)):
        if not paragraph:
            continue
        if len(paragraph) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(_hard_split(paragraph, max_chars, overlap))
        elif current and len(current) + 2 + len(paragraph) > max_chars:
            pieces.append(current)
            current = paragraph
        else:
            current = f"{current}\n\n{paragraph}" if current else paragraph
    if current:
        pieces.append(current)
    return pieces


def _hard_split(text: str, max_chars: int, overlap: int) -> list[str]:
    """Overlapping windows, preferring to cut at a space instead of in the middle of a word."""
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            space = text.rfind(" ", start + max_chars // 2, end)
            if space != -1:
                end = space
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)  # always move forward
    return pieces
