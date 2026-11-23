"""OpsPilot AI - Document Ingestion, Text Extraction & Chunking Service Tests (Step 7.2B).

Comprehensive unit and integration test suite covering:
1. Successful TXT ingestion.
2. Successful Markdown ingestion.
3. Successful multi-page PDF ingestion.
4. Correct PDF page-number preservation.
5. SHA-256 content hash correctness.
6. Duplicate-content rejection.
7. Unsupported file extension rejection.
8. Empty file rejection.
9. File-size limit enforcement (10 MB).
10. Invalid filename and path traversal protection.
11. Filename collision handling (safe suffixing).
12. Empty/scanned PDF rejection.
13. UTF-8 BOM handling.
14. Safe fallback decoding & binary rejection.
15. Chunk size and overlap behavior.
16. Paragraph/sentence-aware splitting.
17. Long text splitting without content loss.
18. Short document handling.
19. Sequential chunk indexes.
20. Filesystem cleanup after database failure (compensating rollback).
21. No misleading 'Indexed' record after failed persistence.
22. Destination resolution handles SQLite error defensively (defensive exception handling).
23. Production database isolation check (data/opspilot.db remains 0 rows).

Uses isolated temporary directories and isolated temporary SQLite databases for all tests.
Generates synthetic PDFs using pypdf. No live external network calls or Gemini APIs.
"""

import hashlib
import io
import os
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pypdf

from app import config
from app.database import db
from app.services import rag_service


def create_synthetic_pdf(page_texts):
    """Generate a valid multi-page PDF in memory with text content using pypdf."""
    writer = pypdf.PdfWriter()
    for text in page_texts:
        page = writer.add_blank_page(width=300, height=300)
        # Content stream in PDF: BT /F1 12 Tf 10 200 Td (text) Tj ET
        escaped_text = text.replace("(", r"\(").replace(")", r"\)")
        stream_data = f"BT /F1 12 Tf 10 200 Td ({escaped_text}) Tj ET".encode("latin-1")
        stream_obj = pypdf.generic.DecodedStreamObject()
        stream_obj.set_data(stream_data)
        page[pypdf.generic.NameObject("/Contents")] = writer._add_object(stream_obj)

        font_dict = pypdf.generic.DictionaryObject({
            pypdf.generic.NameObject("/F1"): pypdf.generic.DictionaryObject({
                pypdf.generic.NameObject("/Type"): pypdf.generic.NameObject("/Font"),
                pypdf.generic.NameObject("/Subtype"): pypdf.generic.NameObject("/Type1"),
                pypdf.generic.NameObject("/BaseFont"): pypdf.generic.NameObject("/Helvetica"),
            })
        })
        res_dict = pypdf.generic.DictionaryObject({
            pypdf.generic.NameObject("/Font"): font_dict
        })
        page[pypdf.generic.NameObject("/Resources")] = res_dict

    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def create_blank_pdf(num_pages=1):
    """Generate a blank PDF with no text stream (simulating a scanned/empty PDF)."""
    writer = pypdf.PdfWriter()
    for _ in range(num_pages):
        writer.add_blank_page(width=300, height=300)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


class TestRAGIngestionService(unittest.TestCase):
    """Test suite for document ingestion, text extraction, chunking, and persistence."""

    @classmethod
    def setUpClass(cls):
        cls._prod_db_counts = {}
        prod_db = config.DATABASE_PATH
        if os.path.exists(prod_db):
            conn = sqlite3.connect(prod_db)
            cursor = conn.cursor()
            for tbl in ["incidents", "investigations", "knowledge_documents", "knowledge_chunks", "audit_logs"]:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {tbl}")
                    cls._prod_db_counts[tbl] = cursor.fetchone()[0]
                except sqlite3.OperationalError:
                    pass
            conn.close()

    def setUp(self):
        """Create isolated temporary storage directory and temporary SQLite database."""
        self.temp_dir = tempfile.mkdtemp(prefix="opspilot_rag_test_")
        self.temp_storage = os.path.join(self.temp_dir, "storage")
        os.makedirs(self.temp_storage, exist_ok=True)

        self.db_temp_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_temp_file.close()
        self.test_db_path = self.db_temp_file.name

        # Initialize schema, indexes, FTS5 virtual table, and triggers
        db.initialize_database(db_path=self.test_db_path)

    def tearDown(self):
        """Clean up temporary directories and database file."""
        if os.path.exists(self.test_db_path):
            try:
                os.remove(self.test_db_path)
            except OSError:
                pass

        if os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except OSError:
                pass

    # --------------------------------------------------------------------------
    # 1. Successful TXT Ingestion
    # --------------------------------------------------------------------------
    def test_01_successful_txt_ingestion(self):
        """Verify successful ingestion of a standard text runbook."""
        content = (
            "PostgreSQL Database Recovery Runbook.\n\n"
            "Step 1: Check replication lag across read replicas.\n"
            "Step 2: If primary is deadlocked, inspect pg_stat_activity."
        )
        file_bytes = content.encode("utf-8")
        result = rag_service.ingest_document(
            file_name="postgres_recovery.txt",
            file_bytes=file_bytes,
            storage_dir=self.temp_storage,
            db_path=self.test_db_path,
        )

        self.assertIsInstance(result, dict)
        self.assertGreater(result["document_id"], 0)
        self.assertEqual(result["document_name"], "postgres_recovery.txt")
        self.assertEqual(result["document_type"], "TXT")
        self.assertEqual(result["file_size_bytes"], len(file_bytes))
        self.assertEqual(result["status"], "Indexed")
        self.assertGreaterEqual(result["chunk_count"], 1)

        # Confirm file exists on disk with identical content
        dest_path = result["file_path"]
        self.assertTrue(os.path.exists(dest_path))
        with open(dest_path, "rb") as f:
            self.assertEqual(f.read(), file_bytes)

        # Confirm chunks searchable via FTS5
        search_hits = db.search_knowledge_chunks_fts("pg_stat_activity", db_path=self.test_db_path)
        self.assertEqual(len(search_hits), 1)
        self.assertEqual(search_hits[0]["document_id"], result["document_id"])

    # --------------------------------------------------------------------------
    # 2. Successful Markdown Ingestion
    # --------------------------------------------------------------------------
    def test_02_successful_markdown_ingestion(self):
        """Verify successful ingestion of Markdown preserving code and headers."""
        md_content = (
            "# Kubernetes Ingress Troubleshooting\n\n"
            "Check ingress controller logs:\n"
            "```bash\n"
            "kubectl logs -n ingress-nginx deployment/ingress-nginx-controller\n"
            "```\n\n"
            "| Error Code | Meaning |\n"
            "| --- | --- |\n"
            "| 502 | Bad Gateway |\n"
            "| 504 | Gateway Timeout |\n"
        )
        file_bytes = md_content.encode("utf-8")
        result = rag_service.ingest_document(
            file_name="k8s_troubleshooting.md",
            file_bytes=file_bytes,
            storage_dir=self.temp_storage,
            db_path=self.test_db_path,
        )

        self.assertEqual(result["document_type"], "Markdown")
        self.assertEqual(result["status"], "Indexed")

        # Confirm code block content is retained and searchable
        hits = db.search_knowledge_chunks_fts("kubectl logs ingress", db_path=self.test_db_path)
        self.assertGreaterEqual(len(hits), 1)
        self.assertIn("kubectl logs", hits[0]["content"])

    # --------------------------------------------------------------------------
    # 3. Successful Multi-Page PDF Ingestion
    # --------------------------------------------------------------------------
    def test_03_successful_multipage_pdf_ingestion(self):
        """Verify successful ingestion of a synthetic multi-page PDF."""
        pdf_bytes = create_synthetic_pdf([
            "Page 1: Architecture overview of billing microservices.",
            "Page 2: Kafka event broker configuration and consumer groups.",
            "Page 3: Disaster recovery failover procedures for US-East.",
        ])

        result = rag_service.ingest_document(
            file_name="Architecture_Guide.pdf",
            file_bytes=pdf_bytes,
            storage_dir=self.temp_storage,
            db_path=self.test_db_path,
        )

        self.assertEqual(result["document_type"], "PDF")
        self.assertEqual(result["chunk_count"], 3)
        self.assertEqual(result["status"], "Indexed")

    # --------------------------------------------------------------------------
    # 4. Correct PDF Page Number Preservation
    # --------------------------------------------------------------------------
    def test_04_correct_pdf_page_number_preservation(self):
        """Verify that chunks extracted from PDF preserve 1-based page numbers."""
        pdf_bytes = create_synthetic_pdf([
            "Section A on page one.",
            "Section B on page two.",
            "Section C on page three.",
        ])

        result = rag_service.ingest_document(
            file_name="Paged_Doc.pdf",
            file_bytes=pdf_bytes,
            storage_dir=self.temp_storage,
            db_path=self.test_db_path,
        )

        chunks = db.get_chunks_for_document(result["document_id"], db_path=self.test_db_path)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0]["page_number"], 1)
        self.assertIn("Section A", chunks[0]["content"])
        self.assertEqual(chunks[1]["page_number"], 2)
        self.assertIn("Section B", chunks[1]["content"])
        self.assertEqual(chunks[2]["page_number"], 3)
        self.assertIn("Section C", chunks[2]["content"])

    # --------------------------------------------------------------------------
    # 5. SHA-256 Content Hash Correctness
    # --------------------------------------------------------------------------
    def test_05_sha256_content_hash_correctness(self):
        """Verify stored content_hash exactly matches SHA-256 of input bytes."""
        file_bytes = b"Unique content for SHA-256 hash validation test."
        expected_hash = hashlib.sha256(file_bytes).hexdigest()

        result = rag_service.ingest_document(
            file_name="hash_test.txt",
            file_bytes=file_bytes,
            storage_dir=self.temp_storage,
            db_path=self.test_db_path,
        )

        self.assertEqual(result["content_hash"], expected_hash)

        # Lookup by hash
        found = db.get_knowledge_document_by_hash(expected_hash, db_path=self.test_db_path)
        self.assertIsNotNone(found)
        self.assertEqual(found["id"], result["document_id"])

    # --------------------------------------------------------------------------
    # 6. Duplicate Content Rejection
    # --------------------------------------------------------------------------
    def test_06_duplicate_content_rejection(self):
        """Verify uploading identical file bytes raises DuplicateDocumentError."""
        file_bytes = b"Standard operating procedure content."

        # First ingestion succeeds
        rag_service.ingest_document(
            file_name="sop_original.txt",
            file_bytes=file_bytes,
            storage_dir=self.temp_storage,
            db_path=self.test_db_path,
        )

        # Duplicate ingestion with same name fails
        with self.assertRaises(rag_service.DuplicateDocumentError):
            rag_service.ingest_document(
                file_name="sop_original.txt",
                file_bytes=file_bytes,
                storage_dir=self.temp_storage,
                db_path=self.test_db_path,
            )

        # Duplicate ingestion with different name also fails
        with self.assertRaises(rag_service.DuplicateDocumentError):
            rag_service.ingest_document(
                file_name="sop_copy.txt",
                file_bytes=file_bytes,
                storage_dir=self.temp_storage,
                db_path=self.test_db_path,
            )

    # --------------------------------------------------------------------------
    # 7. Unsupported File Extension
    # --------------------------------------------------------------------------
    def test_07_unsupported_file_extension(self):
        """Verify unsupported extensions (.docx, .exe, .py, etc.) are rejected."""
        unsupported = ["script.py", "malware.exe", "document.docx", "data.json", "image.png"]
        for fname in unsupported:
            with self.assertRaises(rag_service.UnsupportedFileTypeError):
                rag_service.ingest_document(
                    file_name=fname,
                    file_bytes=b"dummy bytes",
                    storage_dir=self.temp_storage,
                    db_path=self.test_db_path,
                )

    # --------------------------------------------------------------------------
    # 8. Empty File Rejection
    # --------------------------------------------------------------------------
    def test_08_empty_file_rejection(self):
        """Verify 0-byte file raises EmptyFileError."""
        with self.assertRaises(rag_service.EmptyFileError):
            rag_service.ingest_document(
                file_name="empty.txt",
                file_bytes=b"",
                storage_dir=self.temp_storage,
                db_path=self.test_db_path,
            )

    # --------------------------------------------------------------------------
    # 9. File Size Limit Enforcement
    # --------------------------------------------------------------------------
    def test_09_file_size_limit_enforcement(self):
        """Verify file exceeding 10 MB raises FileSizeLimitError."""
        oversized_bytes = b"A" * (config.MAX_DOCUMENT_SIZE_BYTES + 1)
        with self.assertRaises(rag_service.FileSizeLimitError):
            rag_service.ingest_document(
                file_name="oversized.txt",
                file_bytes=oversized_bytes,
                storage_dir=self.temp_storage,
                db_path=self.test_db_path,
            )

    # --------------------------------------------------------------------------
    # 10. Invalid Filename and Path Traversal Protection
    # --------------------------------------------------------------------------
    def test_10_invalid_filename_and_path_traversal(self):
        """Verify directory traversal and invalid characters are sanitized safely."""
        traversal_names = [
            "../../etc/passwd.txt",
            "..\\..\\system.txt",
            "/absolute/root/file.txt",
            "C:\\Windows\\System32\\driver.txt",
            "sub/folder/runbook.txt",
        ]

        for unsafe_name in traversal_names:
            result = rag_service.ingest_document(
                file_name=unsafe_name,
                file_bytes=f"Content for {unsafe_name}".encode("utf-8"),
                storage_dir=self.temp_storage,
                db_path=self.test_db_path,
            )
            # Stored path must remain strictly within temp_storage
            dest_path = Path(result["file_path"]).resolve()
            storage_path = Path(self.temp_storage).resolve()
            self.assertTrue(dest_path.is_relative_to(storage_path))
            self.assertNotIn("..", dest_path.name)

        # Empty or dot-only filenames raise InvalidFilenameError
        for invalid_name in ["", "   ", "...", ".txt"]:
            with self.assertRaises(rag_service.InvalidFilenameError):
                rag_service.ingest_document(
                    file_name=invalid_name,
                    file_bytes=b"Some content",
                    storage_dir=self.temp_storage,
                    db_path=self.test_db_path,
                )

    # --------------------------------------------------------------------------
    # 11. Filename Collision Handling
    # --------------------------------------------------------------------------
    def test_11_filename_collision_handling(self):
        """Verify uploading different content with same filename generates safe suffix."""
        result1 = rag_service.ingest_document(
            file_name="Incident_Runbook.txt",
            file_bytes=b"Version 1 content for runbook.",
            storage_dir=self.temp_storage,
            db_path=self.test_db_path,
        )
        self.assertEqual(result1["document_name"], "Incident_Runbook.txt")

        # Second upload with different content and same original name
        result2 = rag_service.ingest_document(
            file_name="Incident_Runbook.txt",
            file_bytes=b"Version 2 different content for runbook.",
            storage_dir=self.temp_storage,
            db_path=self.test_db_path,
        )
        self.assertEqual(result2["document_name"], "Incident_Runbook_1.txt")

        # Third upload with different content and same original name
        result3 = rag_service.ingest_document(
            file_name="Incident_Runbook.txt",
            file_bytes=b"Version 3 third variation content for runbook.",
            storage_dir=self.temp_storage,
            db_path=self.test_db_path,
        )
        self.assertEqual(result3["document_name"], "Incident_Runbook_2.txt")

        # Confirm all 3 files exist on disk and in SQLite
        for r in [result1, result2, result3]:
            self.assertTrue(os.path.exists(r["file_path"]))
            doc = db.get_knowledge_document_by_id(r["document_id"], db_path=self.test_db_path)
            self.assertEqual(doc["document_name"], r["document_name"])

    # --------------------------------------------------------------------------
    # 12. Empty / Scanned PDF Rejection
    # --------------------------------------------------------------------------
    def test_12_empty_scanned_pdf_rejection(self):
        """Verify blank PDF with no extractable text raises EmptyExtractedContentError."""
        blank_pdf_bytes = create_blank_pdf(num_pages=2)
        with self.assertRaises(rag_service.EmptyExtractedContentError):
            rag_service.ingest_document(
                file_name="blank_scanned.pdf",
                file_bytes=blank_pdf_bytes,
                storage_dir=self.temp_storage,
                db_path=self.test_db_path,
            )

        # Confirm no file was left on disk and 0 docs in SQLite
        self.assertEqual(len(os.listdir(self.temp_storage)), 0)
        self.assertEqual(len(db.get_all_knowledge_documents(db_path=self.test_db_path)), 0)

    # --------------------------------------------------------------------------
    # 13. UTF-8 BOM Handling
    # --------------------------------------------------------------------------
    def test_13_utf8_bom_handling(self):
        """Verify UTF-8 BOM is stripped cleanly without artifact characters."""
        content = "Operational handbook with UTF-8 byte order mark."
        bom_bytes = b"\xef\xbb\xbf" + content.encode("utf-8")

        result = rag_service.ingest_document(
            file_name="bom_doc.txt",
            file_bytes=bom_bytes,
            storage_dir=self.temp_storage,
            db_path=self.test_db_path,
        )

        chunks = db.get_chunks_for_document(result["document_id"], db_path=self.test_db_path)
        self.assertNotIn("\ufeff", chunks[0]["content"])
        self.assertTrue(chunks[0]["content"].startswith("Operational handbook"))

    # --------------------------------------------------------------------------
    # 14. Safe Fallback Decoding & Binary Rejection
    # --------------------------------------------------------------------------
    def test_14_safe_fallback_decoding(self):
        """Verify CP1252 text decodes safely and binary garbage is rejected."""
        # Windows-1252 text with curly quotes (\x93 and \x94)
        cp1252_bytes = b"Error: \x93Connection dropped by peer\x94 during handshake."
        result = rag_service.ingest_document(
            file_name="cp1252_log.txt",
            file_bytes=cp1252_bytes,
            storage_dir=self.temp_storage,
            db_path=self.test_db_path,
        )
        self.assertEqual(result["status"], "Indexed")

        # Arbitrary binary file with null bytes disguised as .txt
        binary_bytes = b"\x00\x01\x02\x03\xff\xfe\x00\x05binary_data"
        with self.assertRaises(rag_service.TextExtractionError):
            rag_service.ingest_document(
                file_name="binary_corrupt.txt",
                file_bytes=binary_bytes,
                storage_dir=self.temp_storage,
                db_path=self.test_db_path,
            )

    # --------------------------------------------------------------------------
    # 15. Chunk Size and Overlap Behavior
    # --------------------------------------------------------------------------
    def test_15_chunk_size_and_overlap_behavior(self):
        """Verify chunk size and overlap parameters are strictly respected."""
        # Generate text with multiple sentences
        sentences = [f"Sentence {i}: Operational procedures for cluster scaling node group {i}." for i in range(20)]
        long_text = " ".join(sentences)

        chunk_size = 200
        chunk_overlap = 50

        chunks = rag_service.chunk_document_content([(1, long_text)], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.assertGreater(len(chunks), 1)

        for c in chunks:
            self.assertLessEqual(len(c["content"]), chunk_size, f"Chunk exceeded size limit: {len(c['content'])}")

        # Verify overlap exists between consecutive chunks
        for i in range(len(chunks) - 1):
            curr_tail_words = set(chunks[i]["content"].split()[-5:])
            next_head_words = set(chunks[i + 1]["content"].split()[:10])
            # Common words should exist due to overlap
            self.assertTrue(len(curr_tail_words.intersection(next_head_words)) > 0, "Expected word overlap between chunks.")

    # --------------------------------------------------------------------------
    # 16. Paragraph and Sentence-Aware Splitting
    # --------------------------------------------------------------------------
    def test_16_paragraph_sentence_aware_splitting(self):
        """Verify chunking splits cleanly at paragraph or sentence boundaries."""
        p1 = "Paragraph one details the incident escalation path for critical severity alerts."
        p2 = "Paragraph two describes the notification channels including Slack, PagerDuty, and email."
        text = f"{p1}\n\n{p2}"

        # Size large enough for one paragraph, but not both
        chunks = rag_service.chunk_document_content([(1, text)], chunk_size=120, chunk_overlap=20)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["content"], p1)
        self.assertEqual(chunks[1]["content"], p2)

    # --------------------------------------------------------------------------
    # 17. Long Text Splitting Without Content Loss
    # --------------------------------------------------------------------------
    def test_17_long_text_splitting_without_content_loss(self):
        """Verify long unbroken text is chunked safely without dropping content."""
        unbroken_text = "ABCDEFGHIJ" * 50  # 500 characters unbroken string
        chunks = rag_service.chunk_document_content([(1, unbroken_text)], chunk_size=150, chunk_overlap=30)
        self.assertGreaterEqual(len(chunks), 3)

        # Confirm all chunks respect limit
        for c in chunks:
            self.assertLessEqual(len(c["content"]), 150)

        # First chunk starts with ABCDEFGHIJ and last chunk ends with ABCDEFGHIJ
        self.assertTrue(chunks[0]["content"].startswith("ABCDEFGHIJ"))
        self.assertTrue(chunks[-1]["content"].endswith("ABCDEFGHIJ"))

    # --------------------------------------------------------------------------
    # 18. Short Document Handling
    # --------------------------------------------------------------------------
    def test_18_short_document_handling(self):
        """Verify short documents create exactly 1 chunk matching the text."""
        short_text = "Brief operational notice: maintenance completed."
        chunks = rag_service.chunk_document_content([(1, short_text)], chunk_size=600, chunk_overlap=100)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["content"], short_text)
        self.assertEqual(chunks[0]["chunk_index"], 0)

    # --------------------------------------------------------------------------
    # 19. Sequential Chunk Indexes
    # --------------------------------------------------------------------------
    def test_19_sequential_chunk_indexes(self):
        """Verify chunk indexes are strictly sequential (0, 1, 2, ...)."""
        multi_page = [
            (1, "Page 1: " + "alpha beta gamma delta " * 30),
            (2, "Page 2: " + "epsilon zeta eta theta " * 30),
            (3, "Page 3: " + "iota kappa lambda mu " * 30),
        ]
        chunks = rag_service.chunk_document_content(multi_page, chunk_size=200, chunk_overlap=40)
        self.assertGreater(len(chunks), 5)

        for expected_idx, c in enumerate(chunks):
            self.assertEqual(c["chunk_index"], expected_idx)

    # --------------------------------------------------------------------------
    # 20. Filesystem Cleanup After Database Failure
    # --------------------------------------------------------------------------
    def test_20_filesystem_cleanup_after_database_failure(self):
        """Verify compensating cleanup: written file is unlinked if DB persistence fails."""
        file_bytes = b"Temporary content destined for persistence failure."

        # Mock db.create_knowledge_document_with_chunks to raise an exception
        with patch("app.database.db.create_knowledge_document_with_chunks", side_effect=sqlite3.OperationalError("Simulated DB lock")):
            with self.assertRaises(rag_service.PersistenceError):
                rag_service.ingest_document(
                    file_name="failure_cleanup.txt",
                    file_bytes=file_bytes,
                    storage_dir=self.temp_storage,
                    db_path=self.test_db_path,
                )

        # Confirm destination file does NOT remain on disk
        dest_file = os.path.join(self.temp_storage, "failure_cleanup.txt")
        self.assertFalse(os.path.exists(dest_file), "Written file must be unlinked after DB persistence failure.")

    # --------------------------------------------------------------------------
    # 21. No Misleading 'Indexed' Record After Failed Persistence
    # --------------------------------------------------------------------------
    def test_21_no_misleading_indexed_record_after_failed_persistence(self):
        """Verify database rollback ensures no orphaned or empty 'Indexed' doc remains."""
        file_bytes = b"Content to test DB integrity on chunk failure."

        # Pass invalid chunk data via monkeypatching chunk_document_content
        with patch("app.services.rag_service.chunk_document_content", return_value=[{"page_number": 1, "content": ""}]):
            with self.assertRaises(rag_service.PersistenceError):
                rag_service.ingest_document(
                    file_name="rollback_doc.txt",
                    file_bytes=file_bytes,
                    storage_dir=self.temp_storage,
                    db_path=self.test_db_path,
                )

        # Confirm 0 documents and 0 chunks in SQLite
        all_docs = db.get_all_knowledge_documents(db_path=self.test_db_path)
        self.assertEqual(len(all_docs), 0, "Database must not contain any document record after failed persistence.")

    # --------------------------------------------------------------------------
    # 22. Destination Resolution Handles SQLite Error Defensively
    # --------------------------------------------------------------------------
    def test_22_destination_resolution_handles_sqlite_error(self):
        """Verify that resolve_unique_destination handles sqlite3.Error defensively without NameError."""
        with patch("app.database.db.get_connection", side_effect=sqlite3.OperationalError("Simulated locked DB")):
            unique_name, dest_path = rag_service.resolve_unique_destination(
                storage_dir=Path(self.temp_storage),
                safe_filename="resilient_doc.txt",
                db_path=self.test_db_path,
            )
            self.assertEqual(unique_name, "resilient_doc.txt")
            self.assertEqual(dest_path.name, "resilient_doc.txt")

    # --------------------------------------------------------------------------
    # 23. Production Database Isolation
    # --------------------------------------------------------------------------
    def test_23_production_database_isolation(self):
        """Confirm that data/opspilot.db was not modified or accessed during tests."""
        prod_db = config.DATABASE_PATH
        self.assertNotEqual(
            os.path.abspath(self.test_db_path),
            os.path.abspath(prod_db),
            "Test database path must not be the production database path!",
        )
        if os.path.exists(prod_db):
            conn = sqlite3.connect(prod_db)
            cursor = conn.cursor()
            for tbl, initial_count in getattr(self, "_prod_db_counts", {}).items():
                cursor.execute(f"SELECT COUNT(*) FROM {tbl};")
                current_count = cursor.fetchone()[0]
                self.assertEqual(
                    current_count,
                    initial_count,
                    f"Table '{tbl}' in production DB was modified during test run! (before: {initial_count}, after: {current_count})",
                )
            conn.close()


if __name__ == "__main__":
    unittest.main()
