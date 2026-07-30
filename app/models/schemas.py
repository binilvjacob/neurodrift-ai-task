from typing import Literal

from pydantic import BaseModel

LocatorKind = Literal["page", "sheet_rows", "heading", "text"]


class Locator(BaseModel):
    """Where a piece of text came from in its source document, precise enough to cite."""

    kind: LocatorKind
    page_start: int | None = None
    page_end: int | None = None
    sheet: str | None = None
    row_start: int | None = None
    row_end: int | None = None
    heading: str | None = None
    para_start: int | None = None
    para_end: int | None = None
    line_start: int | None = None
    line_end: int | None = None

    def display(self) -> str:
        if self.kind == "page":
            if self.page_start == self.page_end:
                return f"p.{self.page_start}"
            return f"p.{self.page_start}-{self.page_end}"
        if self.kind == "sheet_rows":
            if self.row_start == self.row_end:
                return f"{self.sheet}!row {self.row_start}"
            return f"{self.sheet}!rows {self.row_start}-{self.row_end}"
        if self.kind == "heading":
            return self.heading or "(no heading)"
        if self.kind == "text":
            if self.line_start == self.line_end:
                return f"line {self.line_start}"
            return f"lines {self.line_start}-{self.line_end}"
        raise ValueError(f"unhandled locator kind: {self.kind}")

    @staticmethod
    def merge(locators: list["Locator"]) -> "Locator":
        if not locators:
            raise ValueError("cannot merge an empty list of locators")
        kind = locators[0].kind
        if kind == "page":
            pages = [l.page_start for l in locators if l.page_start is not None] + [
                l.page_end for l in locators if l.page_end is not None
            ]
            return Locator(kind=kind, page_start=min(pages), page_end=max(pages))
        if kind == "sheet_rows":
            rows = [l.row_start for l in locators if l.row_start is not None] + [
                l.row_end for l in locators if l.row_end is not None
            ]
            return Locator(kind=kind, sheet=locators[0].sheet, row_start=min(rows), row_end=max(rows))
        if kind == "heading":
            heading = next((l.heading for l in locators if l.heading), None)
            paras = [l.para_start for l in locators if l.para_start is not None] + [
                l.para_end for l in locators if l.para_end is not None
            ]
            return Locator(
                kind=kind,
                heading=heading,
                para_start=min(paras) if paras else None,
                para_end=max(paras) if paras else None,
            )
        if kind == "text":
            lines = [l.line_start for l in locators if l.line_start is not None] + [
                l.line_end for l in locators if l.line_end is not None
            ]
            return Locator(kind=kind, line_start=min(lines) if lines else None, line_end=max(lines) if lines else None)
        raise ValueError(f"unhandled locator kind: {kind}")


class DocumentMetadata(BaseModel):
    tenant_id: str
    document_id: str
    filename: str
    source_type: Literal["pdf", "docx", "txt", "xlsx"]


class TextSegment(BaseModel):
    """A structural unit extracted from a document (a page, a row, a paragraph...), in document order."""

    text: str
    order: int
    locator: Locator


class Chunk(BaseModel):
    chunk_id: str
    tenant_id: str
    document_id: str
    chunk_index: int
    text: str
    token_count: int
    locator: Locator
    metadata: DocumentMetadata


class SourceChunk(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    text: str
    score: float
    locator: str


class IngestResponse(BaseModel):
    tenant_id: str
    document_id: str
    filename: str
    chunks_indexed: int


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None


class QueryResponse(BaseModel):
    tenant_id: str
    question: str
    answer: str
    grounded: bool
    sources: list[SourceChunk]
    refusal_reason: str | None = None
