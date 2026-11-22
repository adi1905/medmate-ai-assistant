from pathlib import Path
from typing import Optional
import json
from langchain_core.documents import Document
from .helpers.parsed_document import ParsedDocument
import base64
from openai import OpenAI

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

        # pages_with_docling_tables: set[int] = set()
        # header_path_by_page: dict[int, list[str]] = {}

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
            # if page is not None and page not in header_path_by_page:
            #     header_path_by_page[page] = current_path

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
                # if page is not None:
                #     pages_with_docling_tables.add(page)

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


