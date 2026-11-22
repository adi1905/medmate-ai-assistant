import json
import hashlib
from itertools import groupby
 
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
 
from .helpers.base_chunker import BaseChunker
from .helpers.chunk_utils import copy_metadata, generate_chunk_id
from .helpers.parsed_document import ParsedDocument
from collections import OrderedDict
 
class PDFChunker(BaseChunker):
 
    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        max_table_chars: int = 1500,
        max_table_rows: int = 30,
        max_table_cols: int = 15,
        max_image_chars: int = 800,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_table_chars = max_table_chars
        self.max_table_rows = max_table_rows
        self.max_image_chars = max_image_chars
        self.max_table_cols = max_table_cols
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        )
 
    def _make_parent_id(self, doc: Document, kind: str) -> str:
        source = doc.metadata.get("source", "unknown")
        idx = doc.metadata.get(f"{kind}_index")
        stem = source.rsplit(".", 1)[0] if "." in source else source
        return f"{stem}-{kind}-{idx}"
 
    def chunk(self, parsed_document: ParsedDocument) -> list[Document]:
        chunks = []
 
        text_docs = [d for d in parsed_document.documents if d.metadata.get("element_type") == "text"]
        non_text_docs = [d for d in parsed_document.documents if d.metadata.get("element_type") != "text"]
        # formula_docs = [d for d in parsed_document.documents if d.metadata.get("element_type") == "formula"]
 
        # all_docs = parsed_document.documents
 
        chunks.extend(self._chunk_text_groups(text_docs))
 
        for doc in non_text_docs:
            element_type = doc.metadata.get("element_type")
            if element_type == "table":
                chunks.extend(self._chunk_table(doc))
            elif element_type == "image":
                chunks.extend(self._chunk_image(doc))
            elif element_type == "formula":
                chunks.extend(self._chunk_formula(doc))   # simple preserve
 
        # chunks.extend(self._chunk_formula_groups(formula_docs, all_docs))
 
        chunks.sort(key=lambda d: (d.metadata.get("page_start") or d.metadata.get("page") or 0))
 
        return chunks
 
    # ==========================================================
    # TEXT
    # ==========================================================
 
    def _group_key(self, doc: Document) -> str:
        return doc.metadata.get("header_path_str", "")
 
    def _chunk_text_groups(self, text_docs: list[Document]) -> list[Document]:
        chunks = []
        for header_path_str, group in groupby(text_docs, key=self._group_key):
            chunks.extend(self._merge_and_split(list(group)))
        return chunks
 
    def _merge_and_split(self, docs: list[Document]) -> list[Document]:
        if not docs:
            return []
 
        base_meta = copy_metadata(docs[0].metadata)
 
        merged_text = "\n\n".join(
            doc.page_content.strip()
            for doc in docs
            if doc.page_content.strip()
        )
 
        if not merged_text:
            return []
 
        pages = [d.metadata.get("page") for d in docs if d.metadata.get("page")]
        page_start = min(pages) if pages else None
        page_end = max(pages) if pages else None
 
        header_path_str = base_meta.get("header_path_str", "")
        group_id = hashlib.md5(
            f"{base_meta.get('source', '')}::{header_path_str}".encode()
        ).hexdigest()[:12]
 
        splits = self.text_splitter.split_text(merged_text)
 
        result = []
        for idx, split in enumerate(splits):
            meta = copy_metadata(base_meta)
            meta["chunk_index"] = idx
            meta["chunk_total"] = len(splits)
            meta["page_start"] = page_start
            meta["page_end"] = page_end
            meta["group_id"] = group_id
            meta["chunk_id"] = generate_chunk_id(split, meta)
 
            result.append(Document(page_content=split, metadata=meta))
 
        return result
 
    # ==========================================================
    # TABLE
    # ==========================================================
 
    def _chunk_table(self, doc: Document) -> list[Document]:
        content = doc.page_content
        if not content:
            return []

        table_id = self._make_parent_id(doc, "table")
        table_json = doc.metadata.get("table_json")

        # Check size purely on data dimensions, not rendered string length
        if not table_json or (
            len(table_json) <= self.max_table_rows
            and len(table_json[0].keys()) <= self.max_table_cols  # add this param
        ):
            meta = copy_metadata(doc.metadata)
            meta["table_id"] = table_id
            meta["chunk_index"] = 0
            meta["chunk_total"] = 1
            meta["page_start"] = doc.metadata.get("page")
            meta["page_end"] = doc.metadata.get("page")
            meta["chunk_id"] = generate_chunk_id(content, meta)
            return [Document(page_content=content, metadata=meta)]

        # Large table: reconstruct chunks with proper multi-header markdown
        columns = list(table_json[0].keys())
        caption = doc.metadata.get("caption", "")
        row_groups = self._build_row_groups(table_json, columns)

        chunks = []
        row_cursor = 1
        for idx, rows in enumerate(row_groups):
            chunk_md = self._rows_to_markdown(columns, rows)  # now handles multi-headers
            row_start, row_end = row_cursor, row_cursor + len(rows) - 1
            row_cursor = row_end + 1

            range_note = f"(rows {row_start}-{row_end} of {len(table_json)})"
            chunk_content = (
                f"{caption} {range_note}\n\n{chunk_md}"
                if caption else f"Table {range_note}\n\n{chunk_md}"
            )

            meta = copy_metadata(doc.metadata)
            meta["table_id"] = table_id
            meta["chunk_index"] = idx
            meta["chunk_total"] = len(row_groups)
            meta["page_start"] = doc.metadata.get("page")
            meta["page_end"] = doc.metadata.get("page")
            meta["table_json"] = rows
            meta["row_range"] = [row_start, row_end]
            meta["full_table_row_count"] = len(table_json)
            meta["columns"] = columns
            meta["chunk_id"] = generate_chunk_id(chunk_content, meta)
            chunks.append(Document(page_content=chunk_content, metadata=meta))

        return chunks
 
    def _build_row_groups(self, table_json: list[dict], columns: list[str]) -> list[list[dict]]:
        """Group rows respecting both max_table_rows and an approximate char budget,
        never splitting a row itself."""
        row_groups: list[list[dict]] = []
        current_group: list[dict] = []
        current_len = 0
        header_overhead = len(" | ".join(columns)) * 2  # header + separator line estimate
 
        for row in table_json:
            row_str = " | ".join(str(v) for v in row.values())
            if current_group and (
                len(current_group) >= self.max_table_rows
                or current_len + len(row_str) + header_overhead > self.max_table_chars
            ):
                row_groups.append(current_group)
                current_group, current_len = [], 0
            current_group.append(row)
            current_len += len(row_str)
 
        if current_group:
            row_groups.append(current_group)
        return row_groups
 
    @staticmethod
    def _parse_columns(columns: list[str]) -> tuple[list[tuple[str, str, str]], bool]:
        """
        Returns (parsed_cols, is_multiheader).
        Each parsed_col is (parent, child, original_key).
        is_multiheader=True when at least one key has a meaningful parent.child split.
        """
        parsed = []
        is_multiheader = False
        for key in columns:
            if '.' in key:
                parent, child = key.split('.', 1)
                # 'Symbol.Symbol' -> not truly multi-header (parent == child)
                if parent != child:
                    is_multiheader = True
                parsed.append((parent, child, key))
            else:
                parsed.append((key, key, key))
        return parsed, is_multiheader

    @staticmethod
    def _rows_to_markdown(columns: list[str], rows: list[dict]) -> str:
        parsed_cols, is_multiheader = PDFChunker._parse_columns(columns)

        sep = "| " + " | ".join(["---"] * len(parsed_cols)) + " |"

        def make_row(cells: list[str]) -> str:
            return "| " + " | ".join(cells) + " |"

        def make_body(rows, parsed_cols) -> str:
            lines = []
            for row in rows:
                lines.append(make_row([str(row.get(key, "")) for _, _, key in parsed_cols]))
                lines.append(sep)  # separator after EVERY row, matching docling style
            return "\n".join(lines)

        if not is_multiheader:
            child_headers = [child for _, child, _ in parsed_cols]
            return "\n".join([
                make_row(child_headers),
                sep,
                make_body(rows, parsed_cols),
            ])

        # Multi-level: build two header rows
        groups = OrderedDict()
        for parent, child, key in parsed_cols:
            groups.setdefault(parent, []).append((child, key))

        top_cells = []
        for parent, children in groups.items():
            top_cells.append(parent)
            top_cells.extend([""] * (len(children) - 1))

        mid_cells = [child for _, child, _ in parsed_cols]

        return "\n".join([
            make_row(top_cells),
            sep,
            make_row(mid_cells),
            sep,
            make_body(rows, parsed_cols),
        ])
 
    # ==========================================================
    # IMAGE
    # ==========================================================
 
    def _chunk_image(self, doc: Document) -> list[Document]:
        content = doc.page_content

        if not content or content == "[Image]":
            return []

        image_id = self._make_parent_id(doc, "image")

        caption = (doc.metadata.get("caption") or "").strip()
        section = (doc.metadata.get("section") or "").strip()

        has_caption = bool(caption)
        has_description = bool(content)

        prefix_parts = []

        if section:
            prefix_parts.append(f"Section: {section}")

        if caption:
            prefix_parts.append(f"Caption: {caption}")

        prefix = "\n\n".join(prefix_parts)

        def build_chunk_text(body: str) -> str:
            body = body.strip()

            if prefix:
                return f"{prefix}\n\n{body}"

            return body

        # --------------------------------------------------
        # Small image description
        # --------------------------------------------------

        if len(content) <= self.max_image_chars:
            chunk_content = build_chunk_text(content)

            meta = copy_metadata(doc.metadata)
            meta["image_id"] = image_id
            meta["chunk_index"] = 0
            meta["chunk_total"] = 1
            meta["page_start"] = doc.metadata.get("page")
            meta["page_end"] = doc.metadata.get("page")
            meta["has_caption"] = has_caption
            meta["has_description"] = has_description
            meta["chunk_id"] = generate_chunk_id(chunk_content, meta)

            return [
                Document(
                    page_content=chunk_content,
                    metadata=meta,
                )
            ]

        # --------------------------------------------------
        # Large image description
        # --------------------------------------------------

        splits = self.text_splitter.split_text(content)

        chunks = []

        for idx, split in enumerate(splits):
            chunk_content = build_chunk_text(split)

            meta = copy_metadata(doc.metadata)
            meta["image_id"] = image_id
            meta["chunk_index"] = idx
            meta["chunk_total"] = len(splits)
            meta["page_start"] = doc.metadata.get("page")
            meta["page_end"] = doc.metadata.get("page")
            meta["has_caption"] = has_caption
            meta["has_description"] = has_description
            meta["chunk_id"] = generate_chunk_id(chunk_content, meta)

            chunks.append(
                Document(
                    page_content=chunk_content,
                    metadata=meta,
                )
            )

        return chunks
    # ==========================================================
    # FORMULA
    # ==========================================================
 
    def _chunk_formula(self, doc: Document) -> list[Document]:
        content = doc.page_content
        if not content:
            return []
        meta = copy_metadata(doc.metadata)
        meta["chunk_index"] = 0
        meta["chunk_total"] = 1
        meta["page_start"] = doc.metadata.get("page")
        meta["page_end"] = doc.metadata.get("page")
        meta["chunk_id"] = generate_chunk_id(content, meta)
        return [Document(page_content=content, metadata=meta)]