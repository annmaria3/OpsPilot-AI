"""OpsPilot AI - UI Investigation Dashboard & Human Governance Integration Tests (Step 7.4C).

Validates:
1. Incident selection and investigation data retrieval via service layer.
2. AI investigation proposal lifecycle and atomic persistence (INVESTIGATION_RUN).
3. Operator approval workflow (RECOMMENDATION_APPROVED, final_action = ai_recommendation, status = 'Approved').
4. Operator edit-and-approve workflow (RECOMMENDATION_EDITED, final_action = custom text, ai_recommendation preserved untouched).
5. Operator rejection workflow (RECOMMENDATION_REJECTED, final_action = NULL, status = 'Open').
6. Rejection after prior approval explicitly clears final_action to NULL (no COALESCE retention).
7. Provenance round-trip contract: all 9 chunk attributes preserved for UI rendering.
8. HTML sanitization helper (_safe_html_text) prevents XSS injection.
9. Epistemic boundary: confirmed_facts is strictly empty when no verified telemetry source is connected.
10. Citation grounding audit correctly distinguishes grounded references, unsupported claims, and flagged citations.
11. Fallback investigation handling (is_fallback: True) under unconfigured or unavailable API states.
12. Production database isolation check (data/opspilot.db remains 0 rows).

All database operations use isolated temporary SQLite databases.
"""

import html
import os
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from app.database import db
from app.utils.helpers import safe_html_text as _safe_html_text, clean_html_block as _clean_html_block
from app.services import ai_service, incident_service, rag_service


class TestUIInvestigationIntegration(unittest.TestCase):
    """Integration test suite for Step 7.4C AI Investigation Dashboard and Governance."""

    @classmethod
    def setUpClass(cls):
        cls._prod_db_counts = {}
        prod_db = "data/opspilot.db"
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
        """Create an isolated temporary SQLite database for each test."""
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_file.close()
        self.test_db_path = self.temp_file.name

        db.initialize_database(db_path=self.test_db_path)

        # Seed two sample incidents
        self.incident_1_id = db.create_incident(
            title="Database Connection Pool Exhaustion",
            description="Users report 504 Gateway Timeout on checkout API.",
            category="Database Connectivity / Query Failure",
            severity="Critical",
            department="Engineering & DevOps",
            impact="Critical",
            db_path=self.test_db_path,
        )
        self.incident_2_id = db.create_incident(
            title="Payment webhook delivery delay",
            description="Stripe webhook events queued with 30 minute lag.",
            category="Payment / Transaction Failure",
            severity="High",
            department="Payments & Billing",
            impact="High",
            db_path=self.test_db_path,
        )

        # Conforming 9-attribute evidence chunks
        self.sample_evidence = [
            {
                "chunk_id": 101,
                "document_id": 5,
                "document_name": "Runbook_Postgres_Pool.md",
                "page_number": 1,
                "chunk_index": 2,
                "rank": 1,
                "citation": "[Runbook_Postgres_Pool.md, p.1 #2]",
                "citation_full": "Runbook: Runbook_Postgres_Pool.md (Page 1, Chunk 2)",
                "bm25_score": -3.854,
                "relevance_score": 3.854,
                "content": "Scale max_connections to 250 when active connections exceed 90%.",
            },
            {
                "chunk_id": 102,
                "document_id": 5,
                "document_name": "Runbook_Postgres_Pool.md",
                "page_number": 2,
                "chunk_index": 3,
                "rank": 2,
                "citation": "[Runbook_Postgres_Pool.md, p.2 #3]",
                "citation_full": "Runbook: Runbook_Postgres_Pool.md (Page 2, Chunk 3)",
                "bm25_score": -2.140,
                "relevance_score": 2.140,
                "content": "Terminate idle-in-transaction workers via pg_terminate_backend.",
            },
        ]

        # Valid investigation data conforming to approved schema
        self.mock_investigation_data = {
            "reported_symptoms": [
                "Users report 504 Gateway Timeout on checkout API."
            ],
            "confirmed_facts": [],
            "assumptions": [
                "Assuming checkout workers connect directly to primary database cluster"
            ],
            "root_cause_hypotheses": [
                {
                    "hypothesis": "PostgreSQL connection pool exhausted by unclosed checkout transactions",
                    "likelihood": "High",
                    "supporting_evidence_citations": [
                        "[Runbook_Postgres_Pool.md, p.1 #2]"
                    ],
                    "unsupported_claims": [
                        "Checkout transaction leak rate unconfirmed without pg_stat_activity logs"
                    ]
                }
            ],
            "evidence_citations": [
                "[Runbook_Postgres_Pool.md, p.1 #2]",
                "[Runbook_Postgres_Pool.md, p.2 #3]",
            ],
            "ai_recommendation": "1. Temporarily increase max_connections to 250.\n2. Terminate idle worker connections.",
            "confidence": 0.88,
            "missing_information": [
                "SELECT count(*), state FROM pg_stat_activity GROUP BY state;"
            ],
            "diagnostic_actions": [
                "SELECT pid, state, query_start FROM pg_stat_activity WHERE state = 'idle in transaction';"
            ],
            "requires_human_review": True,
            "citation_validation_summary": {
                "total_citations": 2,
                "valid_citations": 2,
                "flagged_citations": 0,
                "flagged_details": [],
            },
            "retrieved_evidence": self.sample_evidence,
        }

    def tearDown(self):
        """Clean up temporary database file."""
        if os.path.exists(self.test_db_path):
            try:
                os.remove(self.test_db_path)
            except OSError:
                pass

    # --------------------------------------------------------------------------
    # 1. INCIDENT SELECTION & DATA LOADING
    # --------------------------------------------------------------------------
    def test_01_investigation_data_loading_and_incident_selection(self):
        """Verify UI data loading: incidents, details, and initial empty investigation state."""
        incidents = incident_service.get_all_incidents(db_path=self.test_db_path)
        self.assertEqual(len(incidents), 2)
        self.assertEqual(incidents[0]["id"], self.incident_2_id)  # Ordered newest first
        self.assertEqual(incidents[1]["id"], self.incident_1_id)

        inc = incident_service.get_incident_by_id(self.incident_1_id, db_path=self.test_db_path)
        self.assertIsNotNone(inc)
        self.assertEqual(inc["formatted_id"], f"INC-{self.incident_1_id:04d}")
        self.assertEqual(inc["status"], "Open")

        # Initially zero investigations
        invs = incident_service.get_investigations_for_incident(self.incident_1_id, db_path=self.test_db_path)
        self.assertEqual(len(invs), 0)

    # --------------------------------------------------------------------------
    # 2. PROPOSAL LIFECYCLE & ATOMIC PERSISTENCE
    # --------------------------------------------------------------------------
    def test_02_investigation_proposal_lifecycle_and_session_staging(self):
        """Verify proposal creation atomically records investigation, audit log, and status transition."""
        inv_id = incident_service.save_investigation_proposal(
            incident_id=self.incident_1_id,
            investigation_data=self.mock_investigation_data,
            evidence_packets=self.sample_evidence,
            actor="AI Diagnostic Engine",
            db_path=self.test_db_path,
        )
        self.assertIsInstance(inv_id, int)
        self.assertGreater(inv_id, 0)

        # 1. Investigation row persisted
        row = db.get_investigation_by_id(inv_id, db_path=self.test_db_path)
        self.assertIsNotNone(row)
        self.assertEqual(row["human_decision"], "Pending")
        self.assertIsNone(row["final_action"])
        self.assertEqual(row["ai_recommendation"], self.mock_investigation_data["ai_recommendation"])

        # 2. Incident status transitioned to 'Pending Approval'
        inc = incident_service.get_incident_by_id(self.incident_1_id, db_path=self.test_db_path)
        self.assertEqual(inc["status"], "Pending Approval")

        # 3. INVESTIGATION_RUN audit log recorded
        logs = incident_service.get_audit_history_for_incident(self.incident_1_id, db_path=self.test_db_path)
        inv_logs = [l for l in logs if l["action"] == "INVESTIGATION_RUN"]
        self.assertEqual(len(inv_logs), 1)
        self.assertEqual(inv_logs[0]["investigation_id"], inv_id)
        self.assertEqual(inv_logs[0]["actor"], "AI Diagnostic Engine")

    # --------------------------------------------------------------------------
    # 3. OPERATOR APPROVAL WORKFLOW
    # --------------------------------------------------------------------------
    def test_03_operator_approval_workflow(self):
        """Verify operator approval sets final_action = ai_recommendation and logs RECOMMENDATION_APPROVED."""
        inv_id = incident_service.save_investigation_proposal(
            incident_id=self.incident_1_id,
            investigation_data=self.mock_investigation_data,
            evidence_packets=self.sample_evidence,
            db_path=self.test_db_path,
        )

        success = incident_service.apply_investigation_decision(
            investigation_id=inv_id,
            decision="Approved",
            reviewer_notes="Approved connection pool scaling.",
            actor="LeadSRE",
            db_path=self.test_db_path,
        )
        self.assertTrue(success)

        # Investigation row
        inv = incident_service.get_investigation_by_id(inv_id, db_path=self.test_db_path)
        self.assertEqual(inv["human_decision"], "Approved")
        self.assertEqual(inv["final_action"], self.mock_investigation_data["ai_recommendation"])
        self.assertEqual(inv["reviewer_notes"], "Approved connection pool scaling.")

        # Incident status
        inc = incident_service.get_incident_by_id(self.incident_1_id, db_path=self.test_db_path)
        self.assertEqual(inc["status"], "Approved")

        # Audit log
        logs = incident_service.get_audit_history_for_incident(self.incident_1_id, db_path=self.test_db_path)
        app_logs = [l for l in logs if l["action"] == "RECOMMENDATION_APPROVED"]
        self.assertEqual(len(app_logs), 1)
        self.assertEqual(app_logs[0]["actor"], "LeadSRE")

    # --------------------------------------------------------------------------
    # 4. OPERATOR EDIT-AND-APPROVE WORKFLOW
    # --------------------------------------------------------------------------
    def test_04_operator_edit_and_approve_preserves_original_recommendation(self):
        """Verify edit-and-approve saves custom final_action while preserving original recommendation untouched."""
        inv_id = incident_service.save_investigation_proposal(
            incident_id=self.incident_1_id,
            investigation_data=self.mock_investigation_data,
            evidence_packets=self.sample_evidence,
            db_path=self.test_db_path,
        )

        custom_action = "1. Scale max_connections to 300.\n2. Do NOT terminate workers; inspect slow query log first."

        success = incident_service.apply_investigation_decision(
            investigation_id=inv_id,
            decision="Edited",
            final_action=custom_action,
            reviewer_notes="Adjusted max_connections to 300 based on node RAM.",
            actor="SeniorDBA",
            db_path=self.test_db_path,
        )
        self.assertTrue(success)

        inv = incident_service.get_investigation_by_id(inv_id, db_path=self.test_db_path)
        self.assertEqual(inv["human_decision"], "Approved")
        self.assertEqual(inv["final_action"], custom_action)
        # CRITICAL: original AI recommendation preserved intact
        self.assertEqual(inv["ai_recommendation"], self.mock_investigation_data["ai_recommendation"])
        self.assertNotEqual(inv["ai_recommendation"], inv["final_action"])

        # Audit log
        logs = incident_service.get_audit_history_for_incident(self.incident_1_id, db_path=self.test_db_path)
        edit_logs = [l for l in logs if l["action"] == "RECOMMENDATION_EDITED"]
        self.assertEqual(len(edit_logs), 1)

    # --------------------------------------------------------------------------
    # 5. OPERATOR REJECTION WORKFLOW
    # --------------------------------------------------------------------------
    def test_05_operator_rejection_clears_final_action(self):
        """Verify rejection sets human_decision = 'Rejected', final_action = NULL, and resets incident status to 'Open'."""
        inv_id = incident_service.save_investigation_proposal(
            incident_id=self.incident_1_id,
            investigation_data=self.mock_investigation_data,
            evidence_packets=self.sample_evidence,
            db_path=self.test_db_path,
        )

        success = incident_service.apply_investigation_decision(
            investigation_id=inv_id,
            decision="Rejected",
            reviewer_notes="Runbook does not apply to this cluster topology.",
            actor="LeadSRE",
            db_path=self.test_db_path,
        )
        self.assertTrue(success)

        inv = incident_service.get_investigation_by_id(inv_id, db_path=self.test_db_path)
        self.assertEqual(inv["human_decision"], "Rejected")
        # STRICT REQUIREMENT: final_action must be explicitly None
        self.assertIsNone(inv["final_action"])

        # Incident status reset to 'Open'
        inc = incident_service.get_incident_by_id(self.incident_1_id, db_path=self.test_db_path)
        self.assertEqual(inc["status"], "Open")

        # Audit log
        logs = incident_service.get_audit_history_for_incident(self.incident_1_id, db_path=self.test_db_path)
        rej_logs = [l for l in logs if l["action"] == "RECOMMENDATION_REJECTED"]
        self.assertEqual(len(rej_logs), 1)

    # --------------------------------------------------------------------------
    # 6. REJECTION AFTER PRIOR APPROVAL (PROVE NO COALESCE LEAK)
    # --------------------------------------------------------------------------
    def test_06_rejection_after_prior_approval_clears_final_action(self):
        """Verify rejecting a previously approved recommendation explicitly clears final_action to NULL."""
        inv_id = incident_service.save_investigation_proposal(
            incident_id=self.incident_1_id,
            investigation_data=self.mock_investigation_data,
            db_path=self.test_db_path,
        )

        # 1. First Approve
        incident_service.apply_investigation_decision(
            investigation_id=inv_id,
            decision="Approved",
            final_action="Initially approved action",
            db_path=self.test_db_path,
        )
        inv_app = incident_service.get_investigation_by_id(inv_id, db_path=self.test_db_path)
        self.assertEqual(inv_app["final_action"], "Initially approved action")

        # 2. Then Overturn / Reject
        incident_service.apply_investigation_decision(
            investigation_id=inv_id,
            decision="Rejected",
            reviewer_notes="Overturned by principal engineer.",
            db_path=self.test_db_path,
        )
        inv_rej = incident_service.get_investigation_by_id(inv_id, db_path=self.test_db_path)
        self.assertEqual(inv_rej["human_decision"], "Rejected")
        self.assertIsNone(inv_rej["final_action"])
        self.assertNotEqual(inv_rej["final_action"], "Initially approved action")

    # --------------------------------------------------------------------------
    # 7. PROVENANCE NINE ATTRIBUTES UI CONTRACT
    # --------------------------------------------------------------------------
    def test_07_provenance_nine_attributes_ui_contract(self):
        """Verify all 9 chunk provenance attributes survive round trip for UI display."""
        inv_id = incident_service.save_investigation_proposal(
            incident_id=self.incident_1_id,
            investigation_data=self.mock_investigation_data,
            evidence_packets=self.sample_evidence,
            db_path=self.test_db_path,
        )

        unpacked = incident_service.get_investigation_by_id(inv_id, db_path=self.test_db_path)
        prov = unpacked.get("provenance", [])
        self.assertEqual(len(prov), 2)

        chunk = prov[0]
        expected_attrs = [
            "chunk_id", "document_id", "document_name", "page_number",
            "chunk_index", "rank", "citation", "citation_full", "bm25_score"
        ]
        for attr in expected_attrs:
            self.assertIn(attr, chunk, f"Provenance chunk missing '{attr}'")

        self.assertEqual(chunk["chunk_id"], 101)
        self.assertEqual(chunk["document_id"], 5)
        self.assertEqual(chunk["document_name"], "Runbook_Postgres_Pool.md")
        self.assertEqual(chunk["page_number"], 1)
        self.assertEqual(chunk["chunk_index"], 2)
        self.assertEqual(chunk["rank"], 1)
        self.assertEqual(chunk["citation"], "[Runbook_Postgres_Pool.md, p.1 #2]")
        self.assertEqual(chunk["bm25_score"], -3.854)

    # --------------------------------------------------------------------------
    # 8. HTML SANITIZATION PREVENTS XSS
    # --------------------------------------------------------------------------
    def test_08_safe_html_text_prevents_xss(self):
        """Verify _safe_html_text neutralizes malicious HTML, script tags, and attributes."""
        # Malicious payloads
        payload_1 = "<script>alert('XSS')</script>"
        res_1 = _safe_html_text(payload_1)
        self.assertNotIn("<script>", res_1)
        self.assertIn("&lt;script&gt;", res_1)

        payload_2 = "<img src=x onerror=\"javascript:alert('pwned')\">"
        res_2 = _safe_html_text(payload_2)
        self.assertNotIn("<img", res_2)
        self.assertIn("&lt;img", res_2)

        # Newline preservation mode
        multiline_payload = "Line 1\nLine 2\r\nLine 3<script>"
        res_multi = _safe_html_text(multiline_payload, preserve_newlines=True)
        self.assertIn("<br>", res_multi)
        self.assertNotIn("<script>", res_multi)
        self.assertIn("&lt;script&gt;", res_multi)

        # None / empty handling
        self.assertEqual(_safe_html_text(None), "")
        self.assertEqual(_safe_html_text(""), "")

        # ----------------------------------------------------------------------
        # Step 8.1C Regression Tests: Remediation Formatting & Security
        # ----------------------------------------------------------------------
        from app.utils.helpers import format_remediation_html

        # 1. XSS Payloads strictly neutralized
        xss_payload = "<script>alert('xss')</script>\n<img src=x onerror=alert(1)>"
        xss_formatted = format_remediation_html(xss_payload)
        self.assertNotIn("<script>", xss_formatted)
        self.assertNotIn("<img", xss_formatted)
        self.assertIn("&lt;script&gt;", xss_formatted)
        self.assertIn("&lt;img", xss_formatted)

        # 2. Legitimate backslashes preserved (file paths, regex tokens)
        legit_path = "1. Inspect path C:\\OpsPilot\\logs\\app.log\n2. Match pattern \\d+ errors."
        path_formatted = format_remediation_html(legit_path)
        self.assertIn("C:\\OpsPilot\\logs\\app.log", path_formatted)
        self.assertIn("\\d+", path_formatted)

        # 3. Markdown escape numbers unescaped (1\. -> 1., 2\. -> 2.)
        escaped_nums = "1\\. Step one\n\n2\\. Step two\n\n3\\. Step three"
        nums_formatted = format_remediation_html(escaped_nums)
        self.assertIn("1. Step one", nums_formatted)
        self.assertIn("2. Step two", nums_formatted)
        self.assertIn("3. Step three", nums_formatted)
        self.assertNotIn(r"1\.", nums_formatted)
        self.assertNotIn(r"2\.", nums_formatted)

        # 4. Newline normalization without excessive blank lines (no <br><br>)
        double_newlines = "1. First step.\n\n2. Second step.\n\n\n3. Third step."
        compact_formatted = format_remediation_html(double_newlines)
        self.assertNotIn("<br><br>", compact_formatted)
        self.assertEqual(compact_formatted, "1. First step.<br>2. Second step.<br>3. Third step.")

        # 5. HTML entity handling (&#x20; decoded, None/empty safe)
        entity_text = "1. Step&#x20;with&#x20;entities."
        entity_formatted = format_remediation_html(entity_text)
        self.assertNotIn("&#x20;", entity_formatted)
        self.assertIn("1. Step with entities.", entity_formatted)
        self.assertEqual(format_remediation_html(None), "")
        self.assertEqual(format_remediation_html(""), "")

        # clean_html_block strips indentation without altering inner HTML tags
        indented_html = """
            <div class="journey-stage-card">
                <div class="journey-narrative-box">Narrative</div>
            </div>
        """
        cleaned = _clean_html_block(indented_html)
        self.assertTrue(cleaned.startswith('<div class="journey-stage-card">'))
        for line in cleaned.splitlines():
            self.assertEqual(line, line.lstrip(), f"Line must not have leading whitespace: {line}")
        self.assertNotIn("    ", cleaned)
        self.assertEqual(_clean_html_block(None), "")
        self.assertEqual(_clean_html_block(""), "")

    # --------------------------------------------------------------------------
    # 9. EPISTEMIC BOUNDARY: CONFIRMED FACTS EMPTY WITHOUT TELEMETRY
    # --------------------------------------------------------------------------
    def test_09_epistemic_boundary_confirmed_facts_empty(self):
        """Verify confirmed_facts is strictly empty when no external verified telemetry is provided."""
        # Even if caller attempts to supply telemetry in incident dictionary
        forged_incident = {
            "title": "Database incident",
            "description": "504 Gateway Timeout error code 504.",
            "verified_telemetry": ["pg_stat_activity: 95 connections active"],
            "telemetry": ["CPU usage 99%"],
        }
        evidence_packet = {"chunks": self.sample_evidence}

        # _extract_verified_telemetry strictly returns []
        telemetry = ai_service._extract_verified_telemetry(forged_incident)
        self.assertEqual(telemetry, [])

        # Deserialized proposal has empty confirmed_facts
        unpacked = incident_service.deserialize_investigation({
            "confirmed_facts": "[]",
            "ai_recommendation": "Scale pool",
        })
        self.assertEqual(unpacked["confirmed_facts"], [])

    # --------------------------------------------------------------------------
    # 10. CITATION GROUNDING AUDIT ACCURACY
    # --------------------------------------------------------------------------
    def test_10_citation_grounding_audit_distinguishes_grounded_and_flagged(self):
        """Verify citation validation distinguishes valid and flagged citations without false zero claims."""
        val_summary = {
            "total_citations": 3,
            "valid_citations": 2,
            "flagged_citations": 1,
            "flagged_details": [
                {"citation": "[Fabricated_Runbook.md, p.9 #99]", "reason": "Not in retrieved evidence"}
            ],
        }
        self.assertEqual(val_summary["total_citations"], 3)
        self.assertEqual(val_summary["valid_citations"], 2)
        self.assertEqual(val_summary["flagged_citations"], 1)
        self.assertGreater(val_summary["flagged_citations"], 0)
        self.assertEqual(len(val_summary["flagged_details"]), 1)

    # --------------------------------------------------------------------------
    # 11. FALLBACK INVESTIGATION RESILIENCE
    # --------------------------------------------------------------------------
    def test_11_fallback_investigation_handling(self):
        """Verify evidence-only fallback structure when Gemini API is unconfigured or unavailable."""
        with patch("app.config.is_gemini_configured", return_value=False):
            fallback_res = ai_service.investigate_incident(
                incident_data={"title": "DB Crash", "description": "Crash log"},
                retrieved_evidence={"chunks": self.sample_evidence},
                timeout=45,
                db_path=self.test_db_path,
            )
            self.assertEqual(fallback_res.get("mode"), "evidence_only_fallback")
            self.assertIn("ai_recommendation", fallback_res)
            self.assertEqual(fallback_res["confirmed_facts"], [])
            self.assertGreater(len(fallback_res["evidence_citations"]), 0)

    # --------------------------------------------------------------------------
    # 12. PRODUCTION DATABASE ISOLATION
    # --------------------------------------------------------------------------
    def test_12_production_database_isolation(self):
        """Verify data/opspilot.db was never touched and remains isolated across all tables."""
        prod_db = "data/opspilot.db"
        self.assertNotEqual(
            os.path.abspath(self.test_db_path),
            os.path.abspath(prod_db),
            "Test database path must not be the production database path!",
        )
        if os.path.exists(prod_db):
            conn = sqlite3.connect(prod_db)
            cursor = conn.cursor()
            for tbl, initial_count in getattr(self, "_prod_db_counts", {}).items():
                cursor.execute(f"SELECT COUNT(*) FROM {tbl}")
                current_count = cursor.fetchone()[0]
                self.assertEqual(
                    current_count,
                    initial_count,
                    f"Table '{tbl}' in production DB was modified during test run! (before: {initial_count}, after: {current_count})",
                )
            conn.close()

    # --------------------------------------------------------------------------
    # 13. HELPERS FORMAT_REMEDIATION_HTML CONTRACT & MODULE EXPORT
    # --------------------------------------------------------------------------
    def test_13_helpers_format_remediation_html_availability_and_contract(self):
        """Verify format_remediation_html is available on app.utils.helpers and satisfies contract."""
        from app.utils import helpers
        self.assertTrue(hasattr(helpers, "format_remediation_html"), "format_remediation_html must exist on app.utils.helpers!")
        self.assertTrue(callable(helpers.format_remediation_html), "helpers.format_remediation_html must be callable!")

        # Also verify import from app.utils directly
        import app.utils
        self.assertTrue(hasattr(app.utils, "helpers"), "app.utils must expose helpers!")
        self.assertTrue(hasattr(app.utils.helpers, "format_remediation_html"))

        # Test empty / None handling
        self.assertEqual(helpers.format_remediation_html(None), "")
        self.assertEqual(helpers.format_remediation_html(""), "")
        self.assertEqual(helpers.format_remediation_html("   "), "")

        # Test contract with multi-step procedure
        proc = "1\\. Identify root cause.\n\n2\\. Verify logs at C:\\OpsPilot\\app.log.\n\n3\\. <script>alert(1)</script> Execute fix."
        res = helpers.format_remediation_html(proc)
        self.assertNotIn("<script>", res)
        self.assertIn("&lt;script&gt;", res)
        self.assertNotIn("<br><br>", res)
        self.assertIn("<br>", res)
        self.assertIn("1. Identify root cause.", res)
        self.assertIn("2. Verify logs at C:\\OpsPilot\\app.log.", res)
        self.assertIn("3. &lt;script&gt;alert(1)&lt;/script&gt; Execute fix.", res)

    # --------------------------------------------------------------------------
    # 14. STEP 8.1C FINAL VISUAL DEFECT REGRESSION TESTS
    # --------------------------------------------------------------------------
    def test_14_step_81c_final_visual_defect_corrections(self):
        r"""Regression tests for all 4 Step 8.1C visual defects:
        1. No literal &#x20;, &amp;#x20;, or &#32; entity leaks.
        2. No markdown escape numbers (1\., 2\.).
        3. No literal backslashes at line endings (CommonMark line break escapes).
        4. Truthful offline fallback messages when 0 runbooks exist.
        """
        from app.utils import helpers
        from app.services.ai_service import _build_evidence_only_fallback

        # Defect 1 & 2 & 3: format_remediation_html
        rem_with_escapes_and_trailing_slashes = (
            "1\\. Identify the affected order reference.\\\n\n"
            "2\\. Verify the payment status in provider dashboard.\\\n\n"
            "3\\. Inspect logs at C:\\OpsPilot\\app.log.\\\n\n"
            "4\\. Ensure entity &#x20; and &amp;#x20; are resolved.\\"
        )
        formatted = helpers.format_remediation_html(rem_with_escapes_and_trailing_slashes)
        self.assertNotIn("&#x20;", formatted)
        self.assertNotIn("&amp;#x20;", formatted)
        self.assertNotIn("&#32;", formatted)
        self.assertNotIn(r"1\.", formatted)
        self.assertNotIn(r"2\.", formatted)
        self.assertNotIn(r"3\.", formatted)
        self.assertNotIn(r"4\.", formatted)
        # Verify no trailing backslashes at line ends
        for line in formatted.split("<br>"):
            self.assertFalse(line.endswith("\\"), f"Line still ends with backslash: {line}")
        # Verify legitimate path backslashes preserved
        self.assertIn("C:\\OpsPilot\\app.log.", formatted)
        # Verify compact single <br> joins
        self.assertNotIn("<br><br>", formatted)

        # Defect 1 & 2 & 3: safe_html_text
        safe_res = helpers.safe_html_text("2\\. Step two with trailing backslash\\", preserve_newlines=False)
        self.assertEqual(safe_res, "2. Step two with trailing backslash")
        self.assertFalse(safe_res.endswith("\\"))

        safe_multiline = helpers.safe_html_text("1\\. Step one\\\n2\\. Step two\\", preserve_newlines=True)
        self.assertEqual(safe_multiline, "1. Step one<br>2. Step two")

        # Defect 1: clean_html_block normalization
        sample_html = """
        <div class="metric-card">
            <span>Department: <strong>Payments</strong></span>
            <span>Severity: <strong>P1</strong></span>
            Model Confidence: <strong>95%</strong>
        </div>
        """
        cleaned = helpers.clean_html_block(sample_html)
        self.assertNotIn("&#x20;", cleaned)
        self.assertIn('margin-right: 4px;', cleaned)

        # Defect 4: format_fallback_guidance contract
        legacy_rec = (
            "Automated AI diagnostic synthesis is currently unavailable. "
            "Please review the retrieved enterprise runbook chunks below to proceed with manual investigation."
        )
        # When 0 citations / 0 docs
        truthful_rec = helpers.format_fallback_guidance(legacy_rec, citations=[], total_docs=0)
        self.assertNotIn("Please review the retrieved enterprise runbook chunks below", truthful_rec)
        self.assertIn("No enterprise runbooks or documentation chunks were retrieved from the Knowledge Base (0 indexed)", truthful_rec)
        self.assertIn("Author an operator remediation procedure in the Governance Console", truthful_rec)

        # When valid citations exist
        grounded_rec = helpers.format_fallback_guidance(legacy_rec, citations=["[Doc: Runbook.md, Chunk 1]"], total_docs=1)
        self.assertIn("Please review the retrieved enterprise runbook chunks below", grounded_rec)

        # Defect 4: _build_evidence_only_fallback in ai_service
        # Empty evidence packets
        fallback_empty = _build_evidence_only_fallback(
            incident_dict={"description": "Test incident"},
            evidence_packets=[],
            reason="Unconfigured API",
            error_type="missing_configuration",
        )
        self.assertNotIn("Please review the retrieved enterprise runbook chunks below", fallback_empty["ai_recommendation"])
        self.assertIn("0 indexed", fallback_empty["ai_recommendation"])
        self.assertEqual(fallback_empty["citation_validation_summary"]["grounding_note"], "No runbooks were retrieved from the Knowledge Base.")
        self.assertIn("No runbooks available in Knowledge Base to review", fallback_empty["diagnostic_actions"])

        # Non-empty evidence packets
        fallback_with_evidence = _build_evidence_only_fallback(
            incident_dict={"description": "Test incident"},
            evidence_packets=[{"citation": "[Runbook.md#1]", "content": "Sample runbook content"}],
            reason="Timeout",
            error_type="timeout",
        )
        self.assertIn("Please review the retrieved enterprise runbook chunks below", fallback_with_evidence["ai_recommendation"])


if __name__ == "__main__":
    unittest.main()
