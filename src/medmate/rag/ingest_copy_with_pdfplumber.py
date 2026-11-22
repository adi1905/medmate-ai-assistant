from pathlib import Path
from typing import Optional
import json
from langchain_core.documents import Document
from .helpers.parsed_document import ParsedDocument
import base64
from openai import OpenAI
import pdfplumber

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions, CodeFormulaVlmOptions, PictureDescriptionApiOptions
from docling_core.types.doc import TextItem, ListItem, TableItem, PictureItem, SectionHeaderItem, FormulaItem, DocItemLabel

class PDFParser:
    def __init__(
        self, 
        images_output_dir:str = "storage/images",
        page_range: tuple[int, int] | None = None,
        process_images: bool = False,
        process_formulas: bool = False,
        formula_preset: str | None = None,
        describe_images: bool = False,
        openai_api_key: str | None = None,
        picture_description_model: str = "gpt-4o-mini",
        picture_description_prompt: str = "Describe this image in detail, focusing on any data, charts, diagrams, or text it contains.",
        figure_idx_offset: int = 0,
        table_idx_offset: int = 0,
        formula_idx_offset: int = 0,
        document_metadata: dict | None = None,
    ):
        self.images_output_dir = Path(images_output_dir)

        self.images_output_dir.mkdir(parents=True, exist_ok=True)
        self.page_range = page_range
        self.process_images = process_images
        self.process_formulas = process_formulas
        self.formula_preset = formula_preset if process_formulas else None

        self.describe_images = describe_images and process_images
        self.openai_api_key = openai_api_key
        self.openai_client = OpenAI(api_key=self.openai_api_key) if self.openai_api_key else None
        self.picture_description_model = picture_description_model
        self.picture_description_prompt = picture_description_prompt
        self.figure_idx_offset = figure_idx_offset
        self.table_idx_offset = table_idx_offset
        self.formula_idx_offset = formula_idx_offset
        self.document_metadata = document_metadata or {}
    

        if self.describe_images and not self.openai_api_key:
            raise ValueError("describe_images=True requires an openai_api_key")
        self.converter = self.build_converter()

    def build_converter(self) -> DocumentConverter:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options = TableStructureOptions(do_cell_matching=True)
        pipeline_options.generate_picture_images = self.process_images
        pipeline_options.images_scale = 2.5

        if self.process_formulas and self.formula_preset:
            print(f"[PDFParser] Formula enrichment ENABLED (VLM preset: {self.formula_preset})")
            pipeline_options.do_formula_enrichment = True
            pipeline_options.code_formula_options = (
                CodeFormulaVlmOptions.from_preset(self.formula_preset)
            )
        elif self.process_formulas:
            print("[PDFParser] Formula detection ENABLED (fast, layout-only — no VLM)")
            pipeline_options.do_formula_enrichment = False
        else:
            print("[PDFParser] Formula processing DISABLED — formulas will be stored as plain text")
            pipeline_options.do_formula_enrichment = False

        # if self.describe_images:
        #     print(f"[PDFParser] Picture description ENABLED (OpenAI model: {self.picture_description_model})")
        #     pipeline_options.do_picture_description = False
        #     pipeline_options.enable_remote_services = True
        #     pipeline_options.picture_description_options = PictureDescriptionApiOptions(
        #         url="https://api.openai.com/v1/chat/completions",
        #         headers={"Authorization": f"Bearer {self.openai_api_key}"},
        #         params={"model": self.picture_description_model},
        #         prompt=self.picture_description_prompt,
        #         timeout=60,
        #     )
        # else:
        #     print("[PDFParser] Picture description DISABLED")

        print(
            f"[PDFParser] Pipeline options: "
            f"process_images={self.process_images}, "
            f"process_formulas={self.process_formulas}, "
            f"do_formula_enrichment={pipeline_options.do_formula_enrichment}, "
            f"describe_images={self.describe_images}"
        )

        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                    # backend=PyPdfiumDocumentBackend
                )
            }
        )
    
    def _get_page(self, element) -> Optional[int]:
        try:
            if element.prov:
                return element.prov[0].page_no
        except Exception as e:
            pass
        return None
    
    @staticmethod
    def _safe_str(value) -> str:
        if value is None:
            return ""
        try:
            if callable(value):
                value = value()
            return str(value)
        except Exception:
            return ""
        
    @staticmethod
    def _row_fill_signature(row: list[str]) -> tuple[bool, ...]:
        """Boolean signature of which cells are non-empty in this row."""
        return tuple(bool(c.strip()) for c in row)

    @classmethod
    def _is_label_only_row(cls, row: list[str]) -> bool:
        """
        True if exactly the first non-empty run of cells is filled and the
        rest of the row is blank -- i.e. this row looks like a standalone
        section label rather than a data row with values in every column.
        """
        sig = cls._row_fill_signature(row)
        if not any(sig):
            return False
        first_empty_after_fill = False
        seen_fill = False
        for filled in sig:
            if filled:
                seen_fill = True
            elif seen_fill:
                first_empty_after_fill = True
        # only col(s) at the start are filled, rest is empty
        filled_count = sum(sig)
        return seen_fill and first_empty_after_fill and filled_count <= max(1, len(sig) // 2)

    @classmethod
    def _is_banner_row(cls, row: list[str], header_fill_sig: tuple[bool, ...]) -> bool:
        """
        True if this row's fill pattern is a full-width single phrase that
        doesn't match the shape of the table's header row at all (e.g. every
        cell filled with fragments of one long all-caps title), which signals
        a new top-level record/section rather than continuation data.
        """
        sig = cls._row_fill_signature(row)
        non_empty = [c for c in row if c.strip()]
        if len(non_empty) < 2:
            return False
        joined = " ".join(non_empty)
        looks_titleish = joined.isupper() and not any(ch.isdigit() for ch in joined)
        shape_mismatch = sig != header_fill_sig
        return looks_titleish and shape_mismatch
    

    def _generate_image_description(
        self,
        image_path: str,
        caption: str | None = None,
        section: str | None = None,
    ) -> str:
        """
        Generate an engineering-focused semantic description
        using GPT-4o-mini Vision.
        """

        if not self.openai_client:
            return ""

        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()

            image_b64 = base64.b64encode(image_bytes).decode("utf-8")

            context_parts = []

            if caption:
                context_parts.append(f"Figure Caption:\n{caption}")

            if section:
                context_parts.append(f"Document Section:\n{section}")

            context_parts.append(self.picture_description_prompt)

            prompt = "\n\n".join(context_parts)

            response = self.openai_client.chat.completions.create(
                model=self.picture_description_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_b64}"
                                },
                            },
                        ],
                    }
                ],
                temperature=0,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"[IMAGE DESCRIPTION ERROR] {image_path}: {e}")
            return ""
        
    # ==========================================================
    # PDFPLUMBER FALLBACK TABLE EXTRACTION
    # ==========================================================

    @staticmethod
    def _clean_cell(val) -> str:
        if val is None:
            return ""
        return str(val).replace("\n", " ").strip()

    def _pdfplumber_table_to_grid(self, raw_table: list[list]) -> list[list[str]]:
        """
        Clean a raw pdfplumber table (list of rows, each a list of cells,
        cells possibly None) into a normalized grid of strings, dropping
        fully-empty rows/columns.
        """
        grid = [[self._clean_cell(c) for c in row] for row in raw_table]

        # drop fully empty rows
        grid = [row for row in grid if any(cell for cell in row)]
        if not grid:
            return []

        # drop fully empty columns
        n_cols = max(len(r) for r in grid)
        grid = [row + [""] * (n_cols - len(row)) for row in grid]
        keep_cols = [c for c in range(n_cols) if any(row[c] for row in grid)]
        grid = [[row[c] for c in keep_cols] for row in grid]

        return grid
    
    def _grid_to_markdown(self, grid: list[list[str]]) -> str:
        if not grid:
            return ""

        header, *rows = grid
        n_cols = len(header)

        def esc(cell: str) -> str:
            return cell.replace("|", "\\|")

        lines = []
        lines.append("| " + " | ".join(esc(c) or f"Col{i+1}" for i, c in enumerate(header)) + " |")
        lines.append("| " + " | ".join(["---"] * n_cols) + " |")
        for row in rows:
            lines.append("| " + " | ".join(esc(c) for c in row) + " |")

        return "\n".join(lines)
    
    def _grid_to_json_records(self, grid: list[list[str]]) -> list[dict]:
        if not grid:
            return []
        header, *rows = grid
        # de-dupe blank/duplicate headers
        seen = {}
        clean_header = []
        for i, h in enumerate(header):
            h = h or f"col_{i+1}"
            if h in seen:
                seen[h] += 1
                h = f"{h}_{seen[h]}"
            else:
                seen[h] = 0
            clean_header.append(h)

        records = []
        for row in rows:
            records.append({clean_header[i]: row[i] for i in range(len(clean_header))})
        return records
    

    def _split_grid_into_sections(self, grid: list[list[str]]) -> list[dict]:
        """
        Splits a merged raw_grid into logical sub-tables by detecting shifts
        in row fill-pattern (structure), not by matching specific header text.
        Works regardless of what column labels say, since FDA-style pages
        vary in what each section is called.
        """
        if not grid:
            return []

        sections: list[dict] = []
        current_rows: list[list[str]] = []
        current_header_sig: Optional[tuple] = None
        current_label: Optional[str] = None
        pending_new_record = False

        def flush():
            nonlocal current_rows, current_label, pending_new_record
            if len(current_rows) >= 2:
                sections.append({
                    "header_path_suffix": current_label,
                    "grid": current_rows,
                    "is_new_drug_section": pending_new_record,
                })
            current_rows = []
            current_label = None
            pending_new_record = False

        for row in grid:
            sig = self._row_fill_signature(row)

            # first row ever: establishes the initial "header shape"
            if current_header_sig is None and not current_rows:
                current_header_sig = sig
                current_rows = [row]
                if self._is_label_only_row(row):
                    current_label = " ".join(c for c in row if c.strip()).strip()
                continue

            # banner row -> new top-level record (e.g. a different drug entry)
            if self._is_banner_row(row, current_header_sig):
                flush()
                current_label = " ".join(c for c in row if c.strip()).strip()
                pending_new_record = True
                current_rows = [row]
                current_header_sig = None  # reset; next row re-establishes shape
                continue

            # label-only row -> new sub-section under current record
            if self._is_label_only_row(row) and current_rows:
                # only treat as boundary if it differs from an established data shape
                if sig != current_header_sig:
                    flush()
                    current_label = " ".join(c for c in row if c.strip()).strip()
                    current_rows = [row]
                    current_header_sig = None
                    continue

            if current_header_sig is None:
                current_header_sig = sig  # lock in shape from first real data row after a boundary

            current_rows.append(row)

        flush()

        if not sections:
            sections = [{"header_path_suffix": None, "grid": grid, "is_new_drug_section": False}]

        return sections
    
    
    def _extract_tables_pdfplumber_for_page(
        self,
        pdf_path: str,
        page_no: int,  # 1-indexed, matching docling's page_no
        table_settings: dict | None = None,
    ) -> list[dict]:
        """
        Run pdfplumber on a single page and return a list of parsed table dicts:
        {markdown, json_records, shape, raw_grid}
        """
        results = []
        settings = table_settings or {
            "vertical_strategy": "lines_strict",
            "horizontal_strategy": "lines_strict",
            "intersection_tolerance": 5,
        }

        try:
            with pdfplumber.open(pdf_path) as pdf:
                if page_no - 1 >= len(pdf.pages) or page_no - 1 < 0:
                    return results
                page = pdf.pages[page_no - 1]

                raw_tables = page.extract_tables(table_settings=settings)

                # fallback strategy if strict lines find nothing (common for
                # tables with weak/no visible borders, e.g. FDA label tables)
                if not raw_tables:
                    loose_settings = {
                        "vertical_strategy": "text",
                        "horizontal_strategy": "text",
                    }
                    raw_tables = page.extract_tables(table_settings=loose_settings)

                for raw_table in raw_tables:
                    grid = self._pdfplumber_table_to_grid(raw_table)
                    if not grid or len(grid) < 2:  # need header + at least 1 row
                        continue

                    for section in self._split_grid_into_sections(grid):
                        sub_grid = section["grid"]
                        if not sub_grid or len(sub_grid) < 2:
                            continue
                        md = self._grid_to_markdown(sub_grid)
                        json_records = self._grid_to_json_records(sub_grid)
                        results.append({
                            "markdown": md,
                            "json_records": json_records,
                            "shape": {"rows": len(sub_grid) - 1, "cols": len(sub_grid[0]) if sub_grid else 0},
                            "raw_grid": sub_grid,
                            "header_path_suffix": section["header_path_suffix"],
                            "is_new_drug_section": section["is_new_drug_section"],
                        })
        except Exception as e:
            print(f"[PDFParser] WARN: pdfplumber extraction failed on page {page_no}: {e}")

        return results

    def _pdfplumber_fallback_tables(
        self,
        file_path: str,
        pages_with_docling_tables: set[int],
        header_path_by_page: dict[int, list[str]],
        source: str,
        source_path: str,
        table_idx_start: int,
    ) -> list[Document]:
        """
        For pages that docling found NO tables on, try pdfplumber. If it finds
        tables there, convert each into a proper table Document.
        This targets the specific failure mode: docling misses a table
        entirely on a page (not: docling table is low quality).
        """
        fallback_docs: list[Document] = []
        table_idx = table_idx_start

        try:
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
        except Exception as e:
            print(f"[PDFParser] WARN: could not open with pdfplumber for fallback pass: {e}")
            return fallback_docs

        pages_to_check = self._pages_in_range(total_pages)

        for page_no in pages_to_check:
            if page_no in pages_with_docling_tables:
                continue  # docling already got a table here, skip

            page_tables = self._extract_tables_pdfplumber_for_page(file_path, page_no)
            print("Page tables content: ", page_tables)
            if not page_tables:
                continue


            base_header_path = header_path_by_page.get(page_no, ["Document"])
            current_drug_section: Optional[str] = None

            for t in page_tables:
                table_idx += 1

                # track drug-section context across sub-tables on the same page
                if t.get("is_new_drug_section") and t.get("header_path_suffix"):
                    current_drug_section = t["header_path_suffix"]

                header_path = list(base_header_path)
                if current_drug_section:
                    header_path.append(current_drug_section)
                if t.get("header_path_suffix") and t["header_path_suffix"] != current_drug_section:
                    header_path.append(t["header_path_suffix"])

                print(f"[PDFParser] pdfplumber FALLBACK table found on page {page_no} "
                      f"(missed by docling) -> table_index={table_idx} "
                      f"section='{t.get('header_path_suffix')}'")

                meta = self._base_metadata(
                    source=source, source_path=source_path,
                    header_path=header_path, page=page_no,
                )
                meta.update({
                    "element_type": "table",
                    "chunk_strategy": "preserve",
                    "table_index": table_idx,
                    "table_shape": t["shape"],
                    "table_json": t["json_records"],
                    "caption": "",
                    "extraction_method": "pdfplumber_fallback",
                })

                fallback_docs.append(Document(page_content=t["markdown"], metadata=meta))

        return fallback_docs

    def _pages_in_range(self, total_pages: int) -> list[int]:
        if not self.page_range:
            return list(range(1, total_pages + 1))
        start, end = self.page_range
        return list(range(max(start, 1), min(end, total_pages) + 1))
        
    def _should_describe_image(
        self,
        width: int | None,
        height: int | None,
        caption: str | None,
    ) -> bool:

        width = width or 0
        height = height or 0

        area = width * height

        # Tiny logos/icons
        if width < 100 or height < 100:
            return False

        # Very small overall area
        if area < 10000:
            return False

        # Tiny image with no caption
        if not caption and area < 40000:
            return False

        return True
        

    def _extract_caption(self, element, doc, all_items_seq=None, item_index=None) -> str:
        """
        1. Try docling's built-in caption resolution (works when captions are linked via RefItem)
        2. Fall back to scanning adjacent items for label=caption (handles unlinked captions)
        """
        caption = ""

        # --- Strategy 1: docling native ---
        try:
            caption = element.caption_text(doc) or ""
        except Exception:
            pass
        if not caption:
            try:
                caption = " ".join(
                    cap.resolve(doc).text
                    for cap in element.captions
                    if hasattr(cap.resolve(doc), "text") and cap.resolve(doc).text
                ).strip()
            except Exception:
                pass

        if caption:
            return caption

        # --- Strategy 2: scan adjacent items for unlinked captions ---
        if all_items_seq is None or item_index is None:
            return caption

        # Check 3 items before and after the table/figure
        for offset in [-3, -2, -1, 1, 2, 3]:
            idx = item_index + offset
            if idx < 0 or idx >= len(all_items_seq):
                continue
            adj_item, _ = all_items_seq[idx]
            if (
                isinstance(adj_item, TextItem)
                and getattr(adj_item, "label", None) == DocItemLabel.CAPTION
            ):
                text = (getattr(adj_item, "text", "") or "").strip()
                if text:
                    caption = text
                    break

        return caption
    
    # ==========================================================
    # METADATA + DOCUMENT BUILDERS
    # ==========================================================

    def _base_metadata(
        self,
            *,
        source: str,
        source_path: str,
        header_path: list[str],
        page: Optional[int],
    ) -> dict:
        metadata = {
            "source": source,
            "source_path": source_path,
            "doc_type": "pdf",
            "parser": "docling",
            "parser_version": "2.0",
            "section": header_path[-1] if header_path else "Document",
            "parent_section": header_path[-2] if len(header_path) >= 2 else None,
            "header_path": header_path,
            "header_path_str": " > ".join(header_path),
            "page": page,
        }

        metadata.update(self.document_metadata)
        return metadata
    
    def _create_text_document(
        self,
        *,
        text: str,
        source: str,
        source_path: str,
        header_path: list[str],
        page: Optional[int],
    ) -> Document:
        
        meta = self._base_metadata(
            source=source, source_path=source_path, header_path=header_path, page=page,
        )
        meta.update({
            "element_type": "text",
            "chunk_strategy": "split",
        })
        return Document(page_content=text, metadata=meta)
    
    def _create_table_document(
        self,
        *,
        source: str,
        source_path: str,
        header_path: list[str],
        page: Optional[int],
        table_idx: int,
        table_md: str,
        table_json: list | None,
        table_shape: dict | None,
        caption: str,
    ) -> Document:
        meta = self._base_metadata(
            source=source, source_path=source_path, header_path=header_path, page=page,
        )
        meta.update({
            "element_type": "table",
            "chunk_strategy": "preserve",
            "table_index": table_idx,
            "table_shape": table_shape,
            "table_json": table_json,
            "caption": caption,
        })
        
        # carries real semantic content that bare cell values don't.
        body = table_md or (json.dumps(table_json, ensure_ascii=False, default=str) if table_json else "")
        content = f"{caption}\n\n{body}" if caption else body
        return Document(page_content=content, metadata=meta)
    
    def _create_image_document(
        self,
        *,
        source: str,
        source_path: str,
        header_path: list[str],
        page: Optional[int],
        image_idx: int,
        image_path: str | None,
        has_image: bool,
        caption: str,
        width: int | None,
        height: int | None,
        description: str | None,
    ) -> Document:
        meta = self._base_metadata(
            source=source, source_path=source_path, header_path=header_path, page=page,
        )
        meta.update({
            "element_type": "image",
            "chunk_strategy": "preserve",
            "image_index": image_idx,
            "image_path": image_path,
            "has_image": has_image,
            "caption": caption,
            "image_width": width,
            "image_height": height,
        })
        content_parts = []
        # if caption:
        #     content_parts.append(f"Caption: {caption}")
        if description:
            content_parts.append(description)
        content = "\n\n".join(content_parts) if content_parts else "[Image]"

        return Document(page_content=content, metadata=meta)

    
    def _create_formula_document(
        self,
        *,
        source, source_path, header_path, page,
        formula_idx, formula_text, formula_quality,
        preceding_context: str = "",   # NEW
        following_context: str = "",   # NEW
    ) -> Document:
        meta = self._base_metadata(
            source=source, source_path=source_path, header_path=header_path, page=page,
        )
        meta.update({
            "element_type": "formula",
            "chunk_strategy": "preserve",
            "formula_quality": formula_quality,
            "formula_index": formula_idx,
            "formula_preset": self.formula_preset,
            "preceding_context": preceding_context,   # NEW
            "following_context": following_context,   # NEW
        })

        parts = []
        if preceding_context:
            parts.append(preceding_context)
        parts.append(formula_text)
        if following_context:
            parts.append(following_context)
        return Document(page_content="\n\n".join(parts), metadata=meta)
    
    def _extract_table(self, element: TableItem, doc, all_items_seq=None, item_index=None) -> tuple[str, list | None, dict | None, str]:
        table_md, table_json, table_shape = "", None, None
        try:
            table_md = self._safe_str(element.export_to_markdown())
        except Exception as e:
            print(f"[PDFParser] WARN: table markdown export failed: {e}")
        try:
            df = element.export_to_dataframe()
            table_shape = {"rows": int(df.shape[0]), "cols": int(df.shape[1])}
            table_json = df.to_dict(orient="records")
        except Exception as e:
            print(f"[PDFParser] WARN: table dataframe export failed: {e}")

        caption = self._extract_caption(element, doc, all_items_seq=all_items_seq, item_index=item_index)
        return table_md, table_json, table_shape, caption
    
    def _extract_image(
        self,
        element,
        doc,
        image_path,
        all_items_seq=None,
        item_index=None,
    ):
        has_image, caption, width, height = False, "", None, None
        

        try:
            image = element.get_image(doc)
            if image:
                width, height = image.size
                image.save(image_path, format="PNG")
                has_image = True
        except Exception as e:
            print(f"[PDFParser] WARN: image extraction failed: {e}")

        caption = self._extract_caption(element, doc, all_items_seq=all_items_seq, item_index=item_index)

        try:
            description = (
                element.meta.description.text
                if element.meta
                and element.meta.description
                and element.meta.description.text
                else ""
            )
        except Exception:
            pass

        return has_image, caption, width, height, description
    
    # ==========================================================
    # PARSE
    # ==========================================================

    def parse(self, file_path: str) -> ParsedDocument:
        source = Path(file_path).name
        source_path = str(Path(file_path).resolve())
        stem = Path(file_path).stem

        kwargs = {"page_range": self.page_range} if self.page_range else {}
        result = self.converter.convert(file_path, **kwargs)
        doc = result.document

        # ----------------------------------------
        # DEBUG: Item type summary
        # ----------------------------------------

        all_items_seq = list(doc.iterate_items())  # single flat list, reused everywhere
        type_counts: dict[str, int] = {}
        label_counts: dict[str, int] = {}
        for item, _ in all_items_seq:
            t = type(item).__name__
            type_counts[t] = type_counts.get(t, 0) + 1
            label = str(getattr(item, "label", "none"))
            label_counts[label] = label_counts.get(label, 0) + 1

        print(f"\n[PDFParser] File     : {source}")
        print(f"[PDFParser] Pages    : {self.page_range or 'all'}")
        print(f"[PDFParser] Images   : {'enabled' if self.process_images else 'disabled'}")
        print(f"[PDFParser] Formulas : {self.formula_preset or ('enabled (no VLM)' if self.process_formulas else 'disabled')}")
        print(f"[PDFParser] Types    : {type_counts}")
        print(f"[PDFParser] Labels   : {label_counts}")

        print(f"[PDFParser] Labels   : {label_counts}")

        # Print sample text for each label
        label_samples: dict[str, list[str]] = {}
        for item, _ in all_items_seq:
            label = str(getattr(item, "label", "none"))
            text = (getattr(item, "text", "") or "").strip()
            if text and label not in label_samples:
                label_samples[label] = []
            if text and len(label_samples.get(label, [])) < 2:  # max 2 samples per label
                label_samples[label].append(text[:80])  # truncate to 80 chars

        print(f"[PDFParser] Label samples:")
        for label, samples in label_samples.items():
            for i, sample in enumerate(samples):
                preview = sample.replace("\n", " ")
                print(f"  [{label}] ({i+1}): {preview}{'...' if len(sample) == 80 else ''}")

        if self.process_formulas and self.formula_preset and type_counts.get("FormulaItem", 0) == 0:
            formula_label_count = label_counts.get("formula", 0)
            if formula_label_count > 0:
                print(f"[PDFParser] WARN: VLM preset set but 0 FormulaItems found; "
                    f"{formula_label_count} TextItem(s) with label=formula will be used instead (quality=ocr).")
            else:
                print("[PDFParser] WARN: VLM preset set but no formulas detected at all in this PDF.")

        # ----------------------------------------
        # FORMULA CONTEXT HELPER (closure over all_items_seq)
        # ----------------------------------------

        def get_formula_context(item_index: int) -> tuple[str, str]:
            """
            Walk backwards and forwards in the sequential item list from a formula's
            position to find the nearest surrounding prose.
            Stops at section boundaries, tables, images, or after 5 hops.
            Returns (preceding_text, following_text).
            """
            preceding, following = "", ""

            # look back
            for i in range(item_index - 1, max(item_index - 6, -1), -1):
                el, _ = all_items_seq[i]
                if isinstance(el, SectionHeaderItem):
                    break  # don't cross a section boundary
                if isinstance(el, (TableItem, PictureItem)):
                    break  # structural break
                if isinstance(el, FormulaItem):
                    continue  # skip adjacent formulas, keep looking back
                if isinstance(el, (TextItem, ListItem)):
                    text = (getattr(el, "text", "") or "").strip()
                    if text and getattr(el, "label", None) != DocItemLabel.CAPTION:
                        preceding = text[-400:] if len(text) > 400 else text
                        break

            # look forward
            for i in range(item_index + 1, min(item_index + 6, len(all_items_seq))):
                el, _ = all_items_seq[i]
                if isinstance(el, SectionHeaderItem):
                    break
                if isinstance(el, (TableItem, PictureItem)):
                    break
                if isinstance(el, FormulaItem):
                    continue
                if isinstance(el, (TextItem, ListItem)):
                    text = (getattr(el, "text", "") or "").strip()
                    if text and getattr(el, "label", None) != DocItemLabel.CAPTION:
                        following = text[:400] if len(text) > 400 else text
                        break

            return preceding, following

        # ----------------------------------------
        # MAIN LOOP
        # ----------------------------------------
        documents: list[Document] = []
        image_paths: list[str] = []

        header_stack: list[str] = []
        table_idx = self.table_idx_offset
        figure_idx = self.figure_idx_offset
        formula_idx = self.formula_idx_offset
        skipped_images = 0
        skipped_formulas_as_text = 0

        pages_with_docling_tables: set[int] = set()
        header_path_by_page: dict[int, list[str]] = {}

        pending_intro: Optional[str] = None
        pending_list: list[str] = []
        pending_list_path: list[str] = []
        pending_list_page: Optional[int] = None

        doc_type_counts: dict[str, int] = {}

        def _flush_list():
            nonlocal pending_intro, pending_list, pending_list_path, pending_list_page
            if not pending_list:
                pending_intro = None
                return
            parts = []
            if pending_intro:
                parts.append(pending_intro)
            parts.extend(pending_list)
            content = "\n".join(parts)
            doc_type_counts["text"] = doc_type_counts.get("text", 0) + 1
            documents.append(self._create_text_document(
                text=content, source=source, source_path=source_path,
                header_path=pending_list_path, page=pending_list_page,
            ))
            pending_intro = None
            pending_list = []
            pending_list_path = []
            pending_list_page = None

        # enumerate so we have the index available for context lookup
        for enum_idx, (element, level) in enumerate(all_items_seq):
            try:
                print("Element metadata : ",element.metadata)
            except:
                pass
            page = self._get_page(element)
            current_path = list(header_stack) or ["Document"]
            if page is not None and page not in header_path_by_page:
                header_path_by_page[page] = current_path

            # ----------------------------------------
            # SECTION HEADER
            # ----------------------------------------
            if isinstance(element, SectionHeaderItem):
                text = (element.text or "").strip()
                if not text:
                    continue
                header_stack = header_stack[:max(level - 1, 0)]
                if not header_stack or header_stack[-1] != text:
                    header_stack.append(text)
                continue

            # ----------------------------------------
            # FORMULA — must come BEFORE TextItem/ListItem (FormulaItem subclasses TextItem)
            # ----------------------------------------
            is_formula_item = isinstance(element, FormulaItem)
            is_formula_label = (
                isinstance(element, TextItem)
                and not is_formula_item
                and getattr(element, "label", None) == DocItemLabel.FORMULA
            )

            if is_formula_item or is_formula_label:
                # print("Formula Item found:", element)

                latex_text = (getattr(element, "text", None) or "").strip()
                ocr_text = (getattr(element, "orig", None) or "").strip()

                if latex_text:
                    formula_text = latex_text
                    quality = "latex"
                elif ocr_text:
                    formula_text = ocr_text
                    quality = "ocr"
                else:
                    continue

                if not self.process_formulas:
                    # Formula processing disabled: store as plain text so content isn't lost
                    skipped_formulas_as_text += 1
                    doc_type_counts["text"] = doc_type_counts.get("text", 0) + 1
                    documents.append(self._create_text_document(
                        text=formula_text, source=source, source_path=source_path,
                        header_path=current_path, page=page,
                    ))
                    continue

                formula_idx += 1
                doc_type_counts["formula"] = doc_type_counts.get("formula", 0) + 1

                # grab surrounding prose NOW, while we have the doc in memory
                preceding_context, following_context = get_formula_context(enum_idx)

                documents.append(self._create_formula_document(
                    source=source, source_path=source_path, header_path=current_path,
                    page=page, formula_idx=formula_idx, formula_text=formula_text,
                    formula_quality=quality,
                    preceding_context=preceding_context,
                    following_context=following_context,
                ))
                continue

            # ----------------------------------------
            # TEXT / LIST
            # ----------------------------------------

            elif isinstance(element, (TextItem, ListItem)):
                text = (getattr(element, "text", "") or "").strip()
                if not text:
                    continue
                if isinstance(element, TextItem) and getattr(element, "label", None) == DocItemLabel.CAPTION:
                    continue

                if isinstance(element, ListItem):
                    if not pending_list:
                        pending_list_path = current_path
                        pending_list_page = page
                    pending_list.append(f"• {text}")
                    continue

                if text.endswith(":"):
                    _flush_list()
                    pending_intro = text
                    continue

                _flush_list()
                doc_type_counts["text"] = doc_type_counts.get("text", 0) + 1
                documents.append(self._create_text_document(
                    text=text, source=source, source_path=source_path,
                    header_path=current_path, page=page,
                ))
                continue

            # ----------------------------------------
            # TABLE
            # ----------------------------------------

            elif isinstance(element, TableItem):
                table_idx += 1
                table_md, table_json, table_shape, caption = self._extract_table(
                    element, doc, all_items_seq=all_items_seq, item_index=enum_idx
                )
                if not table_md and not table_json:
                    print(f"[PDFParser] SKIP TableItem #{table_idx} page={page}: no content extracted")
                    continue

                # NEW: mark this page as already having a docling table
                if page is not None:
                    pages_with_docling_tables.add(page)

                doc_type_counts["table"] = doc_type_counts.get("table", 0) + 1
                documents.append(self._create_table_document(
                    source=source, source_path=source_path, header_path=current_path,
                    page=page, table_idx=table_idx, table_md=table_md,
                    table_json=table_json, table_shape=table_shape, caption=caption,
                ))
                continue

            # ----------------------------------------
            # IMAGE
            # ----------------------------------------

            elif isinstance(element, PictureItem):

                figure_idx += 1
                image_path = self.images_output_dir / f"{stem}-figure-{figure_idx}.png"
                has_image, caption, width, height, _ = self._extract_image(
                    element,
                    doc,
                    image_path,
                    all_items_seq=all_items_seq,
                    item_index=enum_idx,
                )
                section_name = current_path[-1] if current_path else ""
                description = ""
                if (
                    self.describe_images and
                    has_image
                    and image_path.exists()
                    and self._should_describe_image(
                        width,
                        height,
                        caption,
                    )
                ):
                    description = self._generate_image_description(
                        image_path=str(image_path),
                        caption=caption,
                        section=section_name,
                    )
                else:
                    print("not describing image")
                doc_type_counts["image"] = doc_type_counts.get("image", 0) + 1
                documents.append(self._create_image_document(
                    source=source, source_path=source_path, header_path=current_path,
                    page=page, image_idx=figure_idx,
                    image_path=str(image_path) if has_image else None,
                    has_image=has_image, caption=caption, width=width, height=height, description=description
                ))

        _flush_list()

        fallback_table_docs = self._pdfplumber_fallback_tables(
            file_path=file_path,
            pages_with_docling_tables=pages_with_docling_tables,
            header_path_by_page=header_path_by_page,
            source=source,
            source_path=source_path,
            table_idx_start=table_idx,
        )
        if fallback_table_docs:
            print(f"[PDFParser] pdfplumber fallback added {len(fallback_table_docs)} table(s) "
                  f"missed by docling")
            documents.extend(fallback_table_docs)
            doc_type_counts["table"] = doc_type_counts.get("table", 0) + len(fallback_table_docs)

        # ----------------------------------------
        # DEBUG: Final summary
        # ----------------------------------------
        print(f"\n[PDFParser] Stored documents       : {doc_type_counts}")
        print(f"[PDFParser] Total documents         : {len(documents)}")
        print(f"[PDFParser] Images on disk          : {len(image_paths)}")
        if skipped_images:
            print(f"[PDFParser] Images skipped (process_images=False): {skipped_images}")
        if skipped_formulas_as_text:
            print(f"[PDFParser] Formulas stored as plain text (process_formulas=False): {skipped_formulas_as_text}")

        return ParsedDocument(
            documents=documents,
            image_paths=image_paths,
            source_type="pdf",
        )


