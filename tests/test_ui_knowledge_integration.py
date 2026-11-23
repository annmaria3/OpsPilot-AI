"""OpsPilot AI - Knowledge Base UI & End-to-End RAG Integration Tests (Step 8.2).

Verifies the Streamlit Knowledge Base interface, document ingestion,
chunk inspection, FTS5 BM25 retrieval, safe deletion, and Incident Journey reflection.

Guarantees 100% test isolation: operates strictly on temporary SQLite databases
and temporary filesystem directories. Never touches or mutates production data.
"""

import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

import app.config
from app.database import db
from app.services import incident_service, rag_service
from app.utils import helpers


class TestUIKnowledgeIntegration(unittest.TestCase):
    """Automated integration tests for the Knowledge Base UI and RAG pipeline."""

    @classmethod
    def setUpClass(cls):
        """Locate script path and record production database baseline."""
        cls.script_path = str(Path(__file__).resolve().parent.parent / "app" / "main.py")
        cls._prod_db_path = "data/opspilot.db"
        cls._prod_baseline = {}
        if os.path.exists(cls._prod_db_path):
            conn = sqlite3.connect(cls._prod_db_path)
            c = conn.cursor()
            for tbl in ["incidents", "investigations", "audit_logs", "knowledge_documents", "knowledge_chunks"]:
                try:
                    c.execute(f"SELECT count(*) FROM {tbl};")
                    cls._prod_baseline[tbl] = c.fetchone()[0]
                except sqlite3.OperationalError:
                    pass
            conn.close()

    def setUp(self):
        """Create isolated temporary database and storage directory."""
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db.close()
        self.test_db_path = self.temp_db.name
        db.initialize_database(db_path=self.test_db_path)

        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_storage_dir = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.test_db_path):
            try:
                os.remove(self.test_db_path)
            except OSError:
                pass
        self.temp_dir.cleanup()

    # --------------------------------------------------------------------------
    # 1. KNOWLEDGE BASE EMPTY STATE RENDERING
    # --------------------------------------------------------------------------
    def test_01_knowledge_base_empty_state_rendering(self):
        """Verify Knowledge Base renders truthful empty state when 0 documents are indexed."""
        with patch.object(app.config, "DATABASE_PATH", self.test_db_path), \
             patch.object(db, "DATABASE_PATH", self.test_db_path), \
             patch.object(app.config, "KNOWLEDGE_BASE_DIR", self.test_storage_dir), \
             patch.object(rag_service, "KNOWLEDGE_BASE_DIR", self.test_storage_dir):

            at = AppTest.from_file(self.script_path)
            at.run(timeout=30)
            self.assertEqual(len(at.exception), 0, f"App threw exception on startup: {at.exception}")

            # Navigate to Knowledge Base
            at.sidebar.radio[0].set_value("Knowledge Base")
            at.run(timeout=30)
            self.assertEqual(len(at.exception), 0, f"Exceptions on KB page: {at.exception}")

            # Verify titles and empty state info messages
            all_markdown = " ".join(m.value for m in at.markdown)
            all_info = " ".join(inf.value for inf in at.info)
            self.assertIn("Knowledge Base", all_markdown)
            self.assertIn("TOTAL RUNBOOKS", all_markdown)
            self.assertIn("INDEXED CHUNKS", all_markdown)
            self.assertIn("The Knowledge Base is currently empty (0 indexed documents)", all_info)

    # --------------------------------------------------------------------------
    # 2. DOCUMENT INVENTORY & CHUNK INSPECTOR RENDERING
    # --------------------------------------------------------------------------
    def test_02_document_inventory_and_chunk_inspection(self):
        """Verify document inventory, metadata, and chunk inspector render correctly when documents are indexed."""
        # Ingest a sample markdown runbook into the isolated test database
        sample_content = (
            "# Payment Gateway Runbook\n\n"
            "This procedure handles HTTP 504 Gateway Timeout errors.\n\n"
            "Check connection pool allocation and trip the circuit breaker if upstream is unresponsive."
        ).encode("utf-8")

        doc_record = rag_service.ingest_document(
            file_name="payment_gateway_runbook.md",
            file_bytes=sample_content,
            storage_dir=self.test_storage_dir,
            db_path=self.test_db_path,
        )
        self.assertIsNotNone(doc_record)
        self.assertEqual(doc_record["document_name"], "payment_gateway_runbook.md")
        self.assertGreater(doc_record["chunk_count"], 0)

        with patch.object(app.config, "DATABASE_PATH", self.test_db_path), \
             patch.object(db, "DATABASE_PATH", self.test_db_path), \
             patch.object(app.config, "KNOWLEDGE_BASE_DIR", self.test_storage_dir), \
             patch.object(rag_service, "KNOWLEDGE_BASE_DIR", self.test_storage_dir):

            at = AppTest.from_file(self.script_path)
            at.run(timeout=30)

            # Navigate to Knowledge Base
            at.sidebar.radio[0].set_value("Knowledge Base")
            at.run(timeout=30)
            self.assertEqual(len(at.exception), 0)

            all_markdown = " ".join(m.value for m in at.markdown)
            self.assertIn("payment_gateway_runbook.md", all_markdown)
            self.assertIn("Chunk Inspector", all_markdown)
            self.assertIn("DOCUMENT ID", all_markdown)

            # Verify chunk inspector text area exists and contains extracted content
            self.assertGreater(len(at.text_area), 0)
            self.assertIn("Payment Gateway Runbook", at.text_area[0].value)

    # --------------------------------------------------------------------------
    # 3. FTS5 RETRIEVAL SEARCH PREVIEW & BM25 ACCURACY
    # --------------------------------------------------------------------------
    def test_03_retrieval_search_preview_and_accurate_bm25(self):
        """Verify FTS5 retrieval search preview displays raw BM25 score accurately without percentage notation."""
        sample_content = (
            "# Database Connection Pool Management\n\n"
            "Diagnose connection starvation and max_connections exhaustion.\n\n"
            "Scale max_connections to 250 and terminate idle-in-transaction sessions."
        ).encode("utf-8")

        rag_service.ingest_document(
            file_name="db_pool_runbook.md",
            file_bytes=sample_content,
            storage_dir=self.test_storage_dir,
            db_path=self.test_db_path,
        )

        with patch.object(app.config, "DATABASE_PATH", self.test_db_path), \
             patch.object(db, "DATABASE_PATH", self.test_db_path), \
             patch.object(app.config, "KNOWLEDGE_BASE_DIR", self.test_storage_dir), \
             patch.object(rag_service, "KNOWLEDGE_BASE_DIR", self.test_storage_dir):

            at = AppTest.from_file(self.script_path)
            at.run(timeout=30)

            at.sidebar.radio[0].set_value("Knowledge Base")
            at.run(timeout=30)

            # Find search input in Tab 3
            search_inputs = [ti for ti in at.text_input if ti.key == "kb_fts_search_query"]
            self.assertEqual(len(search_inputs), 1)

            # Execute search
            search_inputs[0].set_value("connection starvation")
            at.run(timeout=30)
            self.assertEqual(len(at.exception), 0)

            all_markdown = " ".join(m.value for m in at.markdown)
            self.assertIn("Retrieved", all_markdown)
            self.assertIn("BM25 Score:", all_markdown)
            # Ensure BM25 score is NOT formatted as a percentage
            self.assertNotIn("% relevant", all_markdown)
            self.assertNotIn("% match", all_markdown)
            self.assertIn("[db_pool_runbook.md, p.1 #", all_markdown)

    # --------------------------------------------------------------------------
    # 4. SAFE DOCUMENT DELETION WORKFLOW
    # --------------------------------------------------------------------------
    def test_04_safe_document_deletion_cascades_and_cleans_file(self):
        """Verify delete_document removes DB record, cascades chunks, clears FTS5, and unlinks file."""
        sample_content = b"# Ephemeral Guide\n\nTemporary content that should be cleanly removed."
        doc_record = rag_service.ingest_document(
            file_name="ephemeral_guide.md",
            file_bytes=sample_content,
            storage_dir=self.test_storage_dir,
            db_path=self.test_db_path,
        )
        doc_id = doc_record["document_id"]
        saved_file = Path(doc_record["file_path"])

        # Confirm file and database records exist
        self.assertTrue(saved_file.exists())
        self.assertIsNotNone(db.get_knowledge_document_by_id(doc_id, db_path=self.test_db_path))
        self.assertGreater(len(db.get_chunks_for_document(doc_id, db_path=self.test_db_path)), 0)
        self.assertEqual(len(db.search_knowledge_chunks_fts("Ephemeral", db_path=self.test_db_path)), 1)

        # Execute safe deletion via service layer
        deleted = rag_service.delete_document(doc_id, db_path=self.test_db_path)
        self.assertTrue(deleted)

        # Confirm database records and chunks are gone
        self.assertIsNone(db.get_knowledge_document_by_id(doc_id, db_path=self.test_db_path))
        self.assertEqual(len(db.get_chunks_for_document(doc_id, db_path=self.test_db_path)), 0)

        # Confirm FTS5 records were cleared via trigger
        self.assertEqual(len(db.search_knowledge_chunks_fts("Ephemeral", db_path=self.test_db_path)), 0)

        # Confirm physical file was unlinked from disk
        self.assertFalse(saved_file.exists())

    # --------------------------------------------------------------------------
    # 5. INCIDENT JOURNEY REFLECTS DYNAMIC KNOWLEDGE STATE
    # --------------------------------------------------------------------------
    def test_05_incident_journey_reflects_dynamic_knowledge_state(self):
        """Verify Incident Journey Stage 3 transitions truthfully when documents exist vs empty."""
        from app.main import compute_journey_stage_statuses

        inc = incident_service.create_incident(
            title="Checkout gateway timeout",
            description="504 errors on payment checkout endpoint",
            category="Payment / Transaction Failure",
            severity="High",
            department="Payments & Billing",
            db_path=self.test_db_path,
        )

        # 1. Zero documents in Knowledge Base -> Stage 3 is attention / 0 Docs Indexed
        stages_empty = compute_journey_stage_statuses(
            incident=inc,
            investigations=[],
            audit_logs=[],
            total_docs=0
        )
        self.assertEqual(stages_empty[3]["pill"], "0 Docs Indexed")
        self.assertEqual(stages_empty[3]["label"], "0 Runbooks Indexed")
        self.assertEqual(stages_empty[3]["state"], "attention")
        self.assertFalse(stages_empty[3]["is_completed"])
        self.assertIn("0 indexed documents", stages_empty[3]["evidence_summary"])

        # 2. Ingest document -> Stage 3 transitions to ready / Runbooks Available
        rag_service.ingest_document(
            file_name="payment_sop.md",
            file_bytes=b"# Payment SOP\n\nTimeout procedures.",
            storage_dir=self.test_storage_dir,
            db_path=self.test_db_path,
        )

        total_docs = len(db.get_all_knowledge_documents(db_path=self.test_db_path))
        self.assertEqual(total_docs, 1)

        stages_populated = compute_journey_stage_statuses(
            incident=inc,
            investigations=[],
            audit_logs=[],
            total_docs=total_docs
        )
        self.assertEqual(stages_populated[3]["pill"], "Ready")
        self.assertEqual(stages_populated[3]["label"], "1 Runbooks Available")
        self.assertEqual(stages_populated[3]["state"], "ready")

        # 3. Investigation with retrieved chunks -> Stage 3 completes with chunk count
        mock_chunks = [{"chunk_id": 1, "citation": "[payment_sop.md, p.1 #0]"}]
        stages_retrieved = compute_journey_stage_statuses(
            incident=inc,
            investigations=[{"retrieved_evidence": mock_chunks}],
            audit_logs=[],
            total_docs=total_docs
        )
        self.assertEqual(stages_retrieved[3]["pill"], "Retrieved")
        self.assertEqual(stages_retrieved[3]["label"], "1 Runbook Chunks")
        self.assertEqual(stages_retrieved[3]["state"], "completed")
        self.assertTrue(stages_retrieved[3]["is_completed"])

    # --------------------------------------------------------------------------
    # 7. JOURNEY STAGE 3 READY STATE & QUERY PREPARATION
    # --------------------------------------------------------------------------
    def test_07_journey_stage3_ready_state_and_query_preparation(self):
        """Verify Stage 3 displays Ready status and pre-populates query from incident symptoms."""
        # Ingest a sample runbook
        rag_service.ingest_document(
            file_name="payment_gateway_runbook.md",
            file_bytes=b"# Payment Gateway Runbook\n\nHandles 504 gateway timeout and circuit breaker.",
            storage_dir=self.test_storage_dir,
            db_path=self.test_db_path,
        )

        inc = incident_service.create_incident(
            title="Checkout gateway timeout",
            description="504 errors on payment checkout endpoint",
            category="Payment / Transaction Failure",
            severity="High",
            department="Payments & Billing",
            db_path=self.test_db_path,
        )
        inc_id = inc["id"]

        with patch.object(app.config, "DATABASE_PATH", self.test_db_path), \
             patch.object(db, "DATABASE_PATH", self.test_db_path), \
             patch.object(app.config, "KNOWLEDGE_BASE_DIR", self.test_storage_dir), \
             patch.object(rag_service, "KNOWLEDGE_BASE_DIR", self.test_storage_dir):

            at = AppTest.from_file(self.script_path)
            at.run(timeout=30)
            self.assertEqual(len(at.exception), 0)

            # Journey is the active page
            all_markdown = " ".join(m.value for m in at.markdown)
            self.assertIn("Incident Journey", all_markdown)
            self.assertIn("Ready for evidence retrieval", all_markdown)
            self.assertIn("1 standard runbooks indexed", all_markdown)

            # Check query input in Stage 3
            query_inputs = [ti for ti in at.text_input if ti.key == f"journey_query_input_{inc_id}"]
            self.assertEqual(len(query_inputs), 1)
            # Pre-populated with extracted keywords from incident
            self.assertIn("checkout", query_inputs[0].value.lower())
            self.assertIn("504", query_inputs[0].value)

    # --------------------------------------------------------------------------
    # 8. JOURNEY STAGE 3 RETRIEVE EVIDENCE INTERACTION & ACCURATE BM25
    # --------------------------------------------------------------------------
    def test_08_journey_stage3_retrieve_evidence_interaction(self):
        """Verify operator can click Retrieve Evidence, see chunks, citations, raw BM25 score, and completed pill."""
        rag_service.ingest_document(
            file_name="payment_gateway_runbook.md",
            file_bytes=b"# Payment Gateway Runbook\n\nProcedures for 504 gateway timeout and circuit breaker trips.",
            storage_dir=self.test_storage_dir,
            db_path=self.test_db_path,
        )

        inc = incident_service.create_incident(
            title="Checkout gateway timeout",
            description="504 errors on payment checkout endpoint",
            category="Payment / Transaction Failure",
            severity="High",
            department="Payments & Billing",
            db_path=self.test_db_path,
        )
        inc_id = inc["id"]

        with patch.object(app.config, "DATABASE_PATH", self.test_db_path), \
             patch.object(db, "DATABASE_PATH", self.test_db_path), \
             patch.object(app.config, "KNOWLEDGE_BASE_DIR", self.test_storage_dir), \
             patch.object(rag_service, "KNOWLEDGE_BASE_DIR", self.test_storage_dir):

            at = AppTest.from_file(self.script_path)
            at.run(timeout=30)

            # Find Retrieve Evidence button
            btn_key = f"btn_retrieve_{inc_id}"
            retrieve_btns = [b for b in at.button if b.key == btn_key]
            self.assertEqual(len(retrieve_btns), 1)

            # Click Retrieve Evidence
            retrieve_btns[0].click()
            at.run(timeout=30)
            self.assertEqual(len(at.exception), 0)

            all_markdown = " ".join(m.value for m in at.markdown)
            # Stage 3 now shows retrieved status
            self.assertIn("STATUS: RETRIEVED", all_markdown)
            self.assertIn("relevant runbook chunks", all_markdown)
            self.assertIn("Retrieved Evidence Packets", all_markdown)
            self.assertIn("BM25 Score:", all_markdown)
            # Strict requirement: No percentage conversion
            self.assertNotIn("% relevant", all_markdown)
            self.assertNotIn("% match", all_markdown)
            # Provenance & Citation displayed
            self.assertIn("payment_gateway_runbook.md", all_markdown)
            self.assertIn("[payment_gateway_runbook.md, p.1 #", all_markdown)
            # Next stage CTA button
            proceed_btns = [b for b in at.button if b.key == f"journey_btn_proceed_to_ai_{inc_id}"]
            self.assertEqual(len(proceed_btns), 1)

    # --------------------------------------------------------------------------
    # 9. JOURNEY STAGE 3 QUERY STALENESS & NO MATCH HANDLING
    # --------------------------------------------------------------------------
    def test_09_journey_stage3_query_staleness_and_no_match(self):
        """Verify query staleness detection and No Match handling when search returns 0 chunks."""
        from app.main import compute_journey_stage_statuses

        inc = {"id": 10, "formatted_id": "INC-010", "title": "Test", "category": "General", "severity": "Low"}

        # 1. Staleness test: is_stale=True causes Query Modified status
        stale_stages = compute_journey_stage_statuses(
            incident=inc,
            investigations=[],
            audit_logs=[],
            total_docs=2,
            retrieved_chunks=[{"chunk_id": 1}],
            is_stale=True,
        )
        self.assertEqual(stale_stages[3]["pill"], "Query Modified")
        self.assertEqual(stale_stages[3]["label"], "Re-Run Needed")
        self.assertEqual(stale_stages[3]["state"], "attention")
        self.assertFalse(stale_stages[3]["is_completed"])

        # 2. No Match test: retrieval_executed=True with 0 chunks causes No Match status
        no_match_stages = compute_journey_stage_statuses(
            incident=inc,
            investigations=[],
            audit_logs=[],
            total_docs=2,
            retrieved_chunks=[],
            retrieval_executed=True,
            is_stale=False,
        )
        self.assertEqual(no_match_stages[3]["pill"], "No Match")
        self.assertEqual(no_match_stages[3]["label"], "0 Chunks Matched")
        self.assertEqual(no_match_stages[3]["state"], "attention")
        self.assertFalse(no_match_stages[3]["is_completed"])

        # 3. Live AppTest No Match check with non-matching query
        rag_service.ingest_document(
            file_name="payment_gateway_runbook.md",
            file_bytes=b"# Payment Gateway Runbook\n\nSpecific payment instructions.",
            storage_dir=self.test_storage_dir,
            db_path=self.test_db_path,
        )
        live_inc = incident_service.create_incident(
            title="Incident with runbooks",
            description="Payment errors",
            db_path=self.test_db_path,
        )
        live_id = live_inc["id"]

        with patch.object(app.config, "DATABASE_PATH", self.test_db_path), \
             patch.object(db, "DATABASE_PATH", self.test_db_path), \
             patch.object(app.config, "KNOWLEDGE_BASE_DIR", self.test_storage_dir), \
             patch.object(rag_service, "KNOWLEDGE_BASE_DIR", self.test_storage_dir):

            at = AppTest.from_file(self.script_path)
            at.run(timeout=30)

            # Edit query to something with 0 matches
            q_inputs = [ti for ti in at.text_input if ti.key == f"journey_query_input_{live_id}"]
            self.assertEqual(len(q_inputs), 1)
            q_inputs[0].set_value("quantum_nonexistent_xyz123")
            at.run(timeout=30)

            # Click retrieve
            btn = [b for b in at.button if b.key == f"btn_retrieve_{live_id}"][0]
            btn.click()
            at.run(timeout=30)

            all_markdown = " ".join(m.value for m in at.markdown)
            self.assertIn("No matching runbook evidence found", all_markdown)
            self.assertIn("0 MATCHES", all_markdown)

    # --------------------------------------------------------------------------
    # 10. STAGE 4 RECEIVES STAGE 3 RETRIEVED EVIDENCE BUNDLE
    # --------------------------------------------------------------------------
    def test_10_stage4_receives_stage3_retrieved_evidence(self):
        """Verify that AI Investigation (Stage 4) receives the exact pre-retrieved chunks from Stage 3."""
        from app.services import ai_service

        inc = incident_service.create_incident(
            title="Database pool exhausted",
            description="500 connection starvation",
            category="Database Connectivity / Query Failure",
            severity="Critical",
            department="Engineering & DevOps",
            db_path=self.test_db_path,
        )
        inc_id = inc["id"]

        test_chunks = [{
            "chunk_id": 42,
            "document_name": "db_pool_runbook.md",
            "page_number": 2,
            "chunk_index": 1,
            "rank": 1,
            "bm25_score": -2.4512,
            "citation": "[db_pool_runbook.md, p.2 #1]",
            "content": "Scale max_connections to 250."
        }]

        captured_call = {}

        def mock_investigate(incident_data, retrieved_evidence=None, timeout=45, db_path=None):
            captured_call["incident_data"] = incident_data
            captured_call["retrieved_evidence"] = retrieved_evidence
            return {
                "incident_id": inc_id,
                "ai_recommendation": "Execute connection scale procedure per runbook [db_pool_runbook.md, p.2 #1].",
                "confirmed_facts": ["Database pool exhausted"],
                "root_cause_hypotheses": [{"hypothesis": "Connection pool exhaustion", "confidence": 0.9}],
                "evidence_citations": ["[db_pool_runbook.md, p.2 #1]"],
                "retrieved_evidence": retrieved_evidence.get("chunks", []) if retrieved_evidence else [],
                "confidence": 0.9,
                "mode": "evidence_grounded",
            }

        with patch.object(app.config, "DATABASE_PATH", self.test_db_path), \
             patch.object(db, "DATABASE_PATH", self.test_db_path), \
             patch.object(ai_service, "investigate_incident", side_effect=mock_investigate):

            at = AppTest.from_file(self.script_path)
            at.run(timeout=30)

            # Pre-set Stage 3 retrieved evidence in session state for this incident
            at.session_state[f"incident_retrieved_evidence_{inc_id}"] = {
                "incident_id": inc_id,
                "search_query": "database pool exhausted",
                "total_chunks_retrieved": 1,
                "chunks": test_chunks,
                "executed": True,
            }

            # Navigate to AI Investigation
            at.sidebar.radio[0].set_value("AI Investigation")
            at.run(timeout=30)
            self.assertEqual(len(at.exception), 0)

            # Trigger investigation
            inv_btn = [b for b in at.button if b.key == f"btn_run_inv_{inc_id}"]
            self.assertEqual(len(inv_btn), 1)
            inv_btn[0].click()
            at.run(timeout=30)
            self.assertEqual(len(at.exception), 0)

            # Verify AI service received the exact Stage 3 chunks
            self.assertIsNotNone(captured_call.get("retrieved_evidence"))
            passed_chunks = captured_call["retrieved_evidence"].get("chunks", [])
            self.assertEqual(len(passed_chunks), 1)
            self.assertEqual(passed_chunks[0]["chunk_id"], 42)
            self.assertEqual(passed_chunks[0]["citation"], "[db_pool_runbook.md, p.2 #1]")

    # --------------------------------------------------------------------------
    # 6. PRODUCTION DATABASE ISOLATION VERIFICATION
    # --------------------------------------------------------------------------
    def test_06_production_database_isolation(self):
        """Verify that the production database remains completely untouched by test executions."""
        if not os.path.exists(self._prod_db_path):
            return

        conn = sqlite3.connect(self._prod_db_path)
        c = conn.cursor()
        current_counts = {}
        for tbl in ["incidents", "investigations", "audit_logs", "knowledge_documents", "knowledge_chunks"]:
            try:
                c.execute(f"SELECT count(*) FROM {tbl};")
                current_counts[tbl] = c.fetchone()[0]
            except sqlite3.OperationalError:
                pass
        conn.close()

        for tbl, baseline_cnt in self._prod_baseline.items():
            self.assertEqual(
                current_counts.get(tbl),
                baseline_cnt,
                f"Production table '{tbl}' count changed! Baseline: {baseline_cnt}, Current: {current_counts.get(tbl)}"
            )


if __name__ == "__main__":
    unittest.main()

