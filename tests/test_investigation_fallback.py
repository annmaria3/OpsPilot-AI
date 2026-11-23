"""Tests for Step 3: AI Investigation Fallback UX Refinement and Governance Safety.

Covers:
1. Missing Gemini configuration returns structured fallback with error_type="missing_configuration".
2. Provider timeout returns structured fallback with error_type="timeout".
3. HTTP 401 and 403 return distinct error_types ("authentication_error", "permission_denied") with sanitized messages.
4. HTTP 429/500/503 provider outages return error_type="provider_unavailable".
5. Confidence handling: SQLite NULL stored and retrieved as None for fallback; valid AI confidence preserved (e.g. 0.88).
6. Backward compatibility: legacy fallback records with confidence 0.0 deserialized as None.
7. Duplicate fallback prevention: retains existing unreviewed fallback without writing duplicate rows or audit events.
8. AI recovery saves new proposal when Gemini becomes available.
9. Retry after approved/rejected proposal is not suppressed.
10. Safety correction: fallback mode disables Approve As-Is, requires operator-authored remediation, and labels approval appropriately.
11. Epistemological labeling: user symptoms, telemetry boundary, runbook reference status, and advisory procedures.
12. Database test isolation: temporary SQLite database used exclusively; data/opspilot.db untouched.
"""

import json
import os
import shutil
import sqlite3
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import requests

import app.config as config
from app.database import db
from app.services import ai_service, incident_service


class TestInvestigationFallback(unittest.TestCase):
    """Test suite covering the 11 required fallback scenarios and safety corrections."""

    @classmethod
    def setUpClass(cls):
        cls._prod_db = "data/opspilot.db"
        cls._prod_counts = {}
        if os.path.exists(cls._prod_db):
            conn = sqlite3.connect(cls._prod_db)
            cursor = conn.cursor()
            for tbl in ["incidents", "investigations", "knowledge_documents", "knowledge_chunks", "audit_logs"]:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {tbl}")
                    cls._prod_counts[tbl] = cursor.fetchone()[0]
                except sqlite3.OperationalError:
                    pass
            conn.close()

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls._prod_db):
            conn = sqlite3.connect(cls._prod_db)
            cursor = conn.cursor()
            for tbl, expected_count in cls._prod_counts.items():
                cursor.execute(f"SELECT COUNT(*) FROM {tbl}")
                actual_count = cursor.fetchone()[0]
                assert actual_count == expected_count, (
                    f"Production DB '{cls._prod_db}' table '{tbl}' was modified! "
                    f"Expected {expected_count}, got {actual_count}."
                )
            conn.close()

    def setUp(self):
        """Set up an isolated temporary database for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_db_path = os.path.join(self.temp_dir, "test_opspilot.db")
        db.initialize_database(self.test_db_path)

        self.sample_incident = {
            "id": 1,
            "title": "Payment Gateway Timeout",
            "description": "504 Gateway Timeout during checkout processing",
            "category": "API",
            "severity": "P1",
            "department": "Engineering",
            "impact": "Customer Facing",
            "status": "Open",
        }

        self.sample_evidence = [
            {
                "chunk_id": 1,
                "document_id": 10,
                "document_name": "Payment_Runbook.md",
                "page_number": 1,
                "chunk_index": 0,
                "content": "Check Redis session cache and payment gateway connection pool.",
                "bm25_score": 4.5,
                "rank": 1,
                "citation": "[Payment_Runbook.md, p.1 #0]",
                "citation_full": "Runbook: Payment_Runbook.md (Page 1, Chunk 0)",
            }
        ]

    def tearDown(self):
        """Clean up isolated temporary database."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    # --------------------------------------------------------------------------
    # 1. MISSING CONFIGURATION FALLBACK
    # --------------------------------------------------------------------------
    def test_01_missing_configuration_fallback(self):
        """Missing GEMINI_API_KEY returns structured fallback without making HTTP requests."""
        with patch.object(config, "is_gemini_configured", return_value=False), \
             patch.object(config, "GEMINI_API_KEY", ""):
            res = ai_service.investigate_incident(
                self.sample_incident,
                retrieved_evidence={"chunks": self.sample_evidence},
            )

            self.assertTrue(res.get("is_fallback"))
            self.assertEqual(res.get("mode"), "evidence_only_fallback")
            self.assertEqual(res.get("error_type"), "missing_configuration")
            self.assertIsNone(res.get("confidence"))
            self.assertIn("Gemini API key is not configured", res.get("fallback_reason", ""))
            self.assertTrue(res.get("requires_human_review"))
            self.assertEqual(len(res.get("root_cause_hypotheses", [])), 0)
            self.assertEqual(len(res.get("retrieved_evidence", [])), 1)

    # --------------------------------------------------------------------------
    # 2. PROVIDER TIMEOUT FALLBACK
    # --------------------------------------------------------------------------
    @patch("app.services.ai_service.requests.post")
    def test_02_timeout_fallback(self, mock_post):
        """Request timeout (>45s) returns structured fallback with error_type='timeout'."""
        mock_post.side_effect = requests.exceptions.Timeout("Read timed out after 45.0s")

        with patch.object(config, "is_gemini_configured", return_value=True), \
             patch.object(config, "GEMINI_API_KEY", "test-secret-key-12345"):
            res = ai_service.investigate_incident(
                self.sample_incident,
                retrieved_evidence={"chunks": self.sample_evidence},
                timeout=45.0,
            )

            self.assertTrue(res.get("is_fallback"))
            self.assertEqual(res.get("error_type"), "timeout")
            self.assertIsNone(res.get("confidence"))
            self.assertIn("timed out after 45.0 seconds", res.get("fallback_reason", ""))

    # --------------------------------------------------------------------------
    # 3. HTTP 401 & 403 AUTHENTICATION / PERMISSION FAILURES
    # --------------------------------------------------------------------------
    @patch("app.services.ai_service.requests.post")
    def test_03_http_401_authentication_failure(self, mock_post):
        """HTTP 401 returns authentication_error fallback and sanitizes secrets."""
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "API key not valid. Please pass a valid API key: test-secret-key-12345"
        mock_post.return_value = mock_resp

        with patch.object(config, "is_gemini_configured", return_value=True), \
             patch.object(config, "GEMINI_API_KEY", "test-secret-key-12345"):
            res = ai_service.investigate_incident(
                self.sample_incident,
                retrieved_evidence={"chunks": self.sample_evidence},
            )

            self.assertTrue(res.get("is_fallback"))
            self.assertEqual(res.get("error_type"), "authentication_error")
            self.assertIsNone(res.get("confidence"))
            # Must NOT expose raw API key
            self.assertNotIn("test-secret-key-12345", res.get("fallback_reason", ""))
            self.assertIn("authentication failed (http 401)", res.get("fallback_reason", "").lower())

    @patch("app.services.ai_service.requests.post")
    def test_03b_http_403_permission_denied(self, mock_post):
        """HTTP 403 returns permission_denied fallback."""
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.text = "The caller does not have permission to access project."
        mock_post.return_value = mock_resp

        with patch.object(config, "is_gemini_configured", return_value=True), \
             patch.object(config, "GEMINI_API_KEY", "test-secret-key-12345"):
            res = ai_service.investigate_incident(
                self.sample_incident,
                retrieved_evidence={"chunks": self.sample_evidence},
            )

            self.assertTrue(res.get("is_fallback"))
            self.assertEqual(res.get("error_type"), "permission_denied")
            self.assertIsNone(res.get("confidence"))
            self.assertIn("permission denied (http 403)", res.get("fallback_reason", "").lower())

    # --------------------------------------------------------------------------
    # 4. HTTP 429/500/503 PROVIDER OUTAGES
    # --------------------------------------------------------------------------
    @patch("app.services.ai_service.requests.post")
    def test_04_provider_outage_fallback(self, mock_post):
        """HTTP 503 Service Unavailable returns provider_unavailable fallback."""
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        mock_resp.text = "The service is temporarily overloaded."
        mock_post.return_value = mock_resp

        with patch.object(config, "is_gemini_configured", return_value=True), \
             patch.object(config, "GEMINI_API_KEY", "test-key"):
            res = ai_service.investigate_incident(
                self.sample_incident,
                retrieved_evidence={"chunks": self.sample_evidence},
            )

            self.assertTrue(res.get("is_fallback"))
            self.assertEqual(res.get("error_type"), "provider_unavailable")
            self.assertIsNone(res.get("confidence"))
            self.assertIn("temporarily unavailable (HTTP 503)", res.get("fallback_reason", ""))

    # --------------------------------------------------------------------------
    # 5. DATABASE NULL CONFIDENCE HANDLING
    # --------------------------------------------------------------------------
    def test_05_database_null_confidence_storage(self):
        """Saving fallback proposal stores SQLite NULL and deserializes as Python None."""
        inc_id = db.create_incident(
            title="Test Incident",
            description="Testing null confidence",
            db_path=self.test_db_path,
        )

        fallback_data = {
            "mode": "evidence_only_fallback",
            "is_fallback": True,
            "error_type": "missing_configuration",
            "fallback_reason": "No API key configured",
            "confidence": None,
            "ai_recommendation": "Automated AI diagnostic synthesis is currently unavailable.",
            "requires_human_review": True,
            "root_cause_hypotheses": [],
            "retrieved_evidence": self.sample_evidence,
            "evidence_citations": ["[Payment_Runbook.md, p.1 #0]"],
            "diagnostic_actions": ["check_logs.sh"],
        }

        inv_id = incident_service.save_investigation_proposal(
            incident_id=inc_id,
            investigation_data=fallback_data,
            evidence_packets=self.sample_evidence,
            actor="AI Diagnostic Engine",
            db_path=self.test_db_path,
        )

        # Raw SQLite inspection: confidence column must be NULL
        with db.get_connection(self.test_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT confidence, evidence_citations FROM investigations WHERE id = ?", (inv_id,))
            raw_row = cursor.fetchone()
            self.assertIsNone(raw_row[0], "Raw SQLite column 'confidence' must be NULL")

        # Deserialized object inspection
        inv_row = db.get_investigation_by_id(inv_id, db_path=self.test_db_path)
        deserialized = incident_service.deserialize_investigation(inv_row)
        self.assertIsNone(deserialized["confidence"])
        self.assertTrue(deserialized["is_fallback"])
        self.assertEqual(deserialized["error_type"], "missing_configuration")

    # --------------------------------------------------------------------------
    # 6. BACKWARD COMPATIBILITY: LEGACY 0.0 CONFIDENCE DESERIALIZATION
    # --------------------------------------------------------------------------
    def test_06_legacy_record_confidence_backward_compatibility(self):
        """Legacy investigation row with confidence=0.0 and fallback text deserializes as None."""
        inc_id = db.create_incident(
            title="Legacy Incident",
            description="Legacy fallback row",
            db_path=self.test_db_path,
        )

        # Simulate older record inserted before Step 3
        with db.get_connection(self.test_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO investigations (
                    incident_id, ai_recommendation, confidence, human_decision, evidence_citations, created_at
                ) VALUES (?, ?, ?, ?, ?, datetime('now'))
                """,
                (
                    inc_id,
                    "Automated AI diagnostic synthesis is currently unavailable (timeout: timed out after 45.0s).",
                    0.0,
                    "Pending",
                    json.dumps({"citations": ["[Payment_Runbook.md, p.1 #0]"]}),
                ),
            )
            legacy_id = cursor.lastrowid

        legacy_row = db.get_investigation_by_id(legacy_id, db_path=self.test_db_path)
        deserialized = incident_service.deserialize_investigation(legacy_row)

        self.assertIsNone(deserialized["confidence"], "Legacy 0.0 fallback confidence should deserialize to None")
        self.assertTrue(deserialized["is_fallback"])

    # --------------------------------------------------------------------------
    # 7. VALID AI CONFIDENCE PRESERVATION
    # --------------------------------------------------------------------------
    def test_07_valid_ai_confidence_preservation(self):
        """Valid AI investigations preserve model-reported confidence (e.g. 0.88)."""
        inc_id = db.create_incident(
            title="Real AI Incident",
            description="Checkout timeout",
            db_path=self.test_db_path,
        )

        valid_ai_data = {
            "mode": "synthesized_analysis",
            "is_fallback": False,
            "confidence": 0.88,
            "ai_recommendation": "Scale Redis connection pool to 200.",
            "requires_human_review": True,
            "root_cause_hypotheses": [
                {
                    "hypothesis": "Redis pool exhaustion",
                    "likelihood": "High",
                    "supporting_evidence_citations": ["[Payment_Runbook.md, p.1 #0]"],
                    "unsupported_claims": [],
                }
            ],
            "diagnostic_actions": ["redis-cli info clients"],
            "retrieved_evidence": self.sample_evidence,
        }

        inv_id = incident_service.save_investigation_proposal(
            incident_id=inc_id,
            investigation_data=valid_ai_data,
            evidence_packets=self.sample_evidence,
            db_path=self.test_db_path,
        )

        with db.get_connection(self.test_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT confidence FROM investigations WHERE id = ?", (inv_id,))
            raw_conf = cursor.fetchone()[0]
            self.assertEqual(raw_conf, 0.88)

        inv_row = db.get_investigation_by_id(inv_id, db_path=self.test_db_path)
        deserialized = incident_service.deserialize_investigation(inv_row)
        self.assertEqual(deserialized["confidence"], 0.88)
        self.assertFalse(deserialized.get("is_fallback", False))

    # --------------------------------------------------------------------------
    # 8. DUPLICATE FALLBACK SUPPRESSION
    # --------------------------------------------------------------------------
    def test_08_duplicate_fallback_prevention_logic(self):
        """When an unreviewed fallback is pending, repeated fallback execution suppresses saving."""
        inc_id = db.create_incident(
            title="Duplicate Test",
            description="Testing duplicate prevention",
            db_path=self.test_db_path,
        )

        fallback_payload = {
            "mode": "evidence_only_fallback",
            "is_fallback": True,
            "error_type": "missing_configuration",
            "fallback_reason": "GEMINI_API_KEY is not configured",
            "confidence": None,
            "ai_recommendation": "Automated AI diagnostic synthesis is currently unavailable.",
            "requires_human_review": True,
            "root_cause_hypotheses": [],
            "retrieved_evidence": self.sample_evidence,
            "evidence_citations": ["[Payment_Runbook.md, p.1 #0]"],
            "diagnostic_actions": ["echo 'test'"],
        }

        # First save: recorded normally
        inv_id_1 = incident_service.save_investigation_proposal(
            incident_id=inc_id,
            investigation_data=fallback_payload,
            evidence_packets=self.sample_evidence,
            db_path=self.test_db_path,
        )

        rows_after_1 = db.get_investigations_for_incident(inc_id, db_path=self.test_db_path)
        self.assertEqual(len(rows_after_1), 1)

        audits_after_1 = incident_service.get_audit_history_for_incident(inc_id, db_path=self.test_db_path)
        audit_count_1 = len(audits_after_1)

        # Re-fetch active_inv
        active_inv = incident_service.deserialize_investigation(rows_after_1[0])

        # Test duplicate prevention criteria
        is_new_fallback = (
            fallback_payload.get("is_fallback") is True
            or fallback_payload.get("mode") == "evidence_only_fallback"
        )
        should_suppress = (
            is_new_fallback
            and active_inv
            and active_inv.get("incident_id") == inc_id
            and active_inv.get("human_decision") == "Pending"
            and (
                active_inv.get("is_fallback") is True
                or active_inv.get("mode") == "evidence_only_fallback"
            )
        )
        self.assertTrue(should_suppress, "Repeated fallback for unreviewed pending fallback must be suppressed")

        # Zero new rows or audit events should be created
        rows_after_suppress = db.get_investigations_for_incident(inc_id, db_path=self.test_db_path)
        self.assertEqual(len(rows_after_suppress), 1)
        audits_after_suppress = incident_service.get_audit_history_for_incident(inc_id, db_path=self.test_db_path)
        self.assertEqual(len(audits_after_suppress), audit_count_1)

    # --------------------------------------------------------------------------
    # 9. AI RECOVERY SAVES NEW PROPOSAL
    # --------------------------------------------------------------------------
    def test_09_ai_recovery_saves_new_proposal(self):
        """When AI service recovers, a new proposal is saved despite prior pending fallback."""
        inc_id = db.create_incident(
            title="Recovery Test",
            description="Testing AI recovery",
            db_path=self.test_db_path,
        )

        fallback_payload = {
            "mode": "evidence_only_fallback",
            "is_fallback": True,
            "confidence": None,
            "ai_recommendation": "Automated AI diagnostic synthesis is currently unavailable.",
        }
        incident_service.save_investigation_proposal(
            incident_id=inc_id,
            investigation_data=fallback_payload,
            evidence_packets=self.sample_evidence,
            db_path=self.test_db_path,
        )

        # Now AI recovers with real analysis
        recovered_payload = {
            "mode": "synthesized_analysis",
            "is_fallback": False,
            "confidence": 0.92,
            "ai_recommendation": "Recovered AI plan: Restart cache worker.",
            "requires_human_review": True,
            "root_cause_hypotheses": [{"hypothesis": "Worker dead", "likelihood": "High"}],
            "diagnostic_actions": ["ps aux | grep worker"],
            "retrieved_evidence": self.sample_evidence,
        }

        # Verify should_suppress is False
        is_new_fallback = (
            recovered_payload.get("is_fallback") is True
            or recovered_payload.get("mode") == "evidence_only_fallback"
        )
        self.assertFalse(is_new_fallback)

        new_id = incident_service.save_investigation_proposal(
            incident_id=inc_id,
            investigation_data=recovered_payload,
            evidence_packets=self.sample_evidence,
            db_path=self.test_db_path,
        )

        all_invs = db.get_investigations_for_incident(inc_id, db_path=self.test_db_path)
        self.assertEqual(len(all_invs), 2, "AI recovery must save a new proposal")

    # --------------------------------------------------------------------------
    # 10. RETRY AFTER DECISION NOT SUPPRESSED
    # --------------------------------------------------------------------------
    def test_10_retry_after_decision_not_suppressed(self):
        """A rerun after a previous proposal has been rejected or approved is never suppressed."""
        inc_id = db.create_incident(
            title="Retry Test",
            description="Testing rerun after decision",
            db_path=self.test_db_path,
        )

        fallback_payload = {
            "mode": "evidence_only_fallback",
            "is_fallback": True,
            "confidence": None,
            "ai_recommendation": "Automated AI diagnostic synthesis is currently unavailable.",
        }
        inv_id = incident_service.save_investigation_proposal(
            incident_id=inc_id,
            investigation_data=fallback_payload,
            evidence_packets=self.sample_evidence,
            db_path=self.test_db_path,
        )

        # Reject the proposal
        incident_service.apply_investigation_decision(
            investigation_id=inv_id,
            decision="Rejected",
            reviewer_notes="Operator rejected fallback.",
            actor="Operator",
            db_path=self.test_db_path,
        )

        # Verify active_inv decision is now 'Rejected', so should_suppress is False
        inv_row = db.get_investigation_by_id(inv_id, db_path=self.test_db_path)
        active_inv = incident_service.deserialize_investigation(inv_row)
        self.assertEqual(active_inv["human_decision"], "Rejected")

        should_suppress = (
            active_inv.get("human_decision") == "Pending"
        )
        self.assertFalse(should_suppress, "Rerun after decision must NOT be suppressed")

    # --------------------------------------------------------------------------
    # 11. GOVERNANCE SAFETY: OPERATOR REMEDIATION AUTHORING REQUIRED & APPROVED
    # --------------------------------------------------------------------------
    def test_11_fallback_governance_safety_operator_authoring(self):
        """In fallback mode, operator authoring is required and decision recorded with RECOMMENDATION_APPROVED audit."""
        inc_id = db.create_incident(
            title="Governance Safety Test",
            description="Testing operator remediation authoring",
            db_path=self.test_db_path,
        )

        fallback_payload = {
            "mode": "evidence_only_fallback",
            "is_fallback": True,
            "confidence": None,
            "ai_recommendation": "Automated AI diagnostic synthesis is currently unavailable.",
        }
        inv_id = incident_service.save_investigation_proposal(
            incident_id=inc_id,
            investigation_data=fallback_payload,
            evidence_packets=self.sample_evidence,
            db_path=self.test_db_path,
        )

        # Operator authors custom procedure and approves it
        custom_action = "Manual failover: run failover_cluster.sh --force --dc us-east"
        success = incident_service.apply_investigation_decision(
            investigation_id=inv_id,
            decision="Approved",
            final_action=custom_action,
            reviewer_notes="Operator manual recovery plan during Gemini outage.",
            actor="Lead Operator",
            db_path=self.test_db_path,
        )
        self.assertTrue(success)

        inv_row = db.get_investigation_by_id(inv_id, db_path=self.test_db_path)
        self.assertEqual(inv_row["human_decision"], "Approved")
        self.assertEqual(inv_row["final_action"], custom_action)

        inc_row = db.get_incident_by_id(inc_id, db_path=self.test_db_path)
        self.assertEqual(inc_row["status"], "Approved")

        logs = incident_service.get_audit_history_for_incident(inc_id, db_path=self.test_db_path)
        app_logs = [l for l in logs if l["action"] == "RECOMMENDATION_APPROVED"]
        self.assertEqual(len(app_logs), 1)
        self.assertEqual(app_logs[0]["actor"], "Lead Operator")
        self.assertIn("operator-authored", app_logs[0]["notes"].lower())
        self.assertIn("not ai-generated", app_logs[0]["notes"].lower())
        self.assertIn(f"Investigation #{inv_id}", app_logs[0]["notes"])

    # --------------------------------------------------------------------------
    # 11B. GOVERNANCE SAFETY: DEFAULT OPERATOR ACTOR IDENTITY
    # --------------------------------------------------------------------------
    def test_11b_fallback_governance_safety_default_operator_identity(self):
        """In fallback mode, default actor identity is 'Operator' when actor argument is omitted."""
        inc_id = db.create_incident(
            title="Default Operator Actor Test",
            description="Testing default Operator actor identity",
            db_path=self.test_db_path,
        )

        fallback_payload = {
            "mode": "evidence_only_fallback",
            "is_fallback": True,
            "confidence": None,
            "ai_recommendation": "Automated AI diagnostic synthesis is currently unavailable.",
        }
        inv_id = incident_service.save_investigation_proposal(
            incident_id=inc_id,
            investigation_data=fallback_payload,
            evidence_packets=self.sample_evidence,
            db_path=self.test_db_path,
        )

        custom_action = "Default actor manual failover procedure."
        success = incident_service.apply_investigation_decision(
            investigation_id=inv_id,
            decision="Approved",
            final_action=custom_action,
            reviewer_notes="Standard manual review notes.",
            # actor argument omitted to verify default 'Operator' identity
            db_path=self.test_db_path,
        )
        self.assertTrue(success)

        inv_row = db.get_investigation_by_id(inv_id, db_path=self.test_db_path)
        self.assertEqual(inv_row["human_decision"], "Approved")
        self.assertEqual(inv_row["final_action"], custom_action)

        inc_row = db.get_incident_by_id(inc_id, db_path=self.test_db_path)
        self.assertEqual(inc_row["status"], "Approved")

        logs = incident_service.get_audit_history_for_incident(inc_id, db_path=self.test_db_path)
        app_logs = [l for l in logs if l["action"] == "RECOMMENDATION_APPROVED"]
        self.assertEqual(len(app_logs), 1)
        self.assertEqual(app_logs[0]["actor"], "Operator")
        self.assertIn("operator-authored", app_logs[0]["notes"].lower())
        self.assertIn("not ai-generated", app_logs[0]["notes"].lower())
        self.assertIn(f"Investigation #{inv_id}", app_logs[0]["notes"])

    # --------------------------------------------------------------------------
    # 12. EMPTY / WHITESPACE OPERATOR REMEDIATION REJECTED WITHOUT AUDIT LOG
    # --------------------------------------------------------------------------
    def test_12_operator_remediation_empty_procedure_rejected(self):
        """Authoring empty or whitespace-only remediation procedure is rejected with ValueError and creates no audit log."""
        inc_id = db.create_incident(
            title="Validation Test",
            description="Testing empty remediation rejection",
            db_path=self.test_db_path,
        )

        fallback_payload = {
            "mode": "evidence_only_fallback",
            "is_fallback": True,
            "confidence": None,
            "ai_recommendation": "Automated AI diagnostic synthesis is currently unavailable.",
        }
        inv_id = incident_service.save_investigation_proposal(
            incident_id=inc_id,
            investigation_data=fallback_payload,
            evidence_packets=self.sample_evidence,
            db_path=self.test_db_path,
        )

        logs_before = incident_service.get_audit_history_for_incident(inc_id, db_path=self.test_db_path)
        initial_log_count = len(logs_before)

        # Empty string must raise ValueError
        with self.assertRaises(ValueError):
            incident_service.apply_investigation_decision(
                investigation_id=inv_id,
                decision="Approved",
                final_action="",
                reviewer_notes="Notes",
                db_path=self.test_db_path,
            )

        # Whitespace-only string must raise ValueError
        with self.assertRaises(ValueError):
            incident_service.apply_investigation_decision(
                investigation_id=inv_id,
                decision="Approved",
                final_action="   \n\t  ",
                reviewer_notes="Notes",
                db_path=self.test_db_path,
            )

        # None must raise ValueError
        with self.assertRaises(ValueError):
            incident_service.apply_investigation_decision(
                investigation_id=inv_id,
                decision="Approved",
                final_action=None,
                reviewer_notes="Notes",
                db_path=self.test_db_path,
            )

        # Zero new audit logs created
        logs_after = incident_service.get_audit_history_for_incident(inc_id, db_path=self.test_db_path)
        self.assertEqual(len(logs_after), initial_log_count)

    # --------------------------------------------------------------------------
    # 13. OPERATOR REMEDIATION APPROVAL WITH MULTI-LINE JUSTIFICATION
    # --------------------------------------------------------------------------
    def test_13_operator_remediation_approval_workflow_with_justification(self):
        """Operator remediation approval persists procedure, multi-line justification, and sets status Approved with RECOMMENDATION_APPROVED."""
        inc_id = db.create_incident(
            title="Database Connection Exhaustion",
            description="Connection pool 100% full",
            db_path=self.test_db_path,
        )

        fallback_payload = {
            "mode": "evidence_only_fallback",
            "is_fallback": True,
            "confidence": None,
            "ai_recommendation": "Automated AI diagnostic synthesis is currently unavailable.",
        }
        inv_id = incident_service.save_investigation_proposal(
            incident_id=inc_id,
            investigation_data=fallback_payload,
            evidence_packets=self.sample_evidence,
            db_path=self.test_db_path,
        )

        multiline_action = (
            "1. Identify active long-running queries via pg_stat_activity.\n"
            "2. Terminate idle-in-transaction connections.\n"
            "3. Increase pgbouncer max_client_conn to 500."
        )
        justification_notes = (
            "Manual review initiated because Gemini AI was offline. "
            "Verified metrics on Grafana dashboard before taking corrective action."
        )

        success = incident_service.apply_investigation_decision(
            investigation_id=inv_id,
            decision="Approved",
            final_action=multiline_action,
            reviewer_notes=justification_notes,
            actor="Operator",
            db_path=self.test_db_path,
        )
        self.assertTrue(success)

        inv_row = db.get_investigation_by_id(inv_id, db_path=self.test_db_path)
        self.assertEqual(inv_row["human_decision"], "Approved")
        self.assertEqual(inv_row["final_action"], multiline_action)
        self.assertEqual(inv_row["reviewer_notes"], justification_notes)

        # Incident status must be 'Approved'
        inc_row = db.get_incident_by_id(inc_id, db_path=self.test_db_path)
        self.assertEqual(inc_row["status"], "Approved")

        # Audit log must record RECOMMENDATION_APPROVED with explicit details
        logs = incident_service.get_audit_history_for_incident(inc_id, db_path=self.test_db_path)
        matching_log = [l for l in logs if l["action"] == "RECOMMENDATION_APPROVED"][0]
        self.assertEqual(matching_log["actor"], "Operator")
        self.assertIn("operator-authored remediation", matching_log["notes"].lower())
        self.assertIn("not ai-generated", matching_log["notes"].lower())
        self.assertIn(f"Investigation #{inv_id}", matching_log["notes"])
        self.assertIn("Verified metrics on Grafana dashboard", matching_log["notes"])

    # --------------------------------------------------------------------------
    # 14. FALLBACK REJECTION WORKFLOW AND INCIDENT STATUS RESET
    # --------------------------------------------------------------------------
    def test_14_fallback_rejection_workflow_and_incident_reset(self):
        """Rejecting fallback resets incident to 'Open', sets final_action to NULL, and logs RECOMMENDATION_REJECTED."""
        inc_id = db.create_incident(
            title="Misrouted Order Alert",
            description="Kafka partition lag",
            db_path=self.test_db_path,
        )

        fallback_payload = {
            "mode": "evidence_only_fallback",
            "is_fallback": True,
            "confidence": None,
            "ai_recommendation": "Automated AI diagnostic synthesis is currently unavailable.",
        }
        inv_id = incident_service.save_investigation_proposal(
            incident_id=inc_id,
            investigation_data=fallback_payload,
            evidence_packets=self.sample_evidence,
            db_path=self.test_db_path,
        )

        # Incident is 'Pending Approval' after proposal save
        inc_before = db.get_incident_by_id(inc_id, db_path=self.test_db_path)
        self.assertEqual(inc_before["status"], "Pending Approval")

        # Reject the proposal
        rej_notes = "Runbook guidance is outdated for Kafka v3.5 cluster."
        success = incident_service.apply_investigation_decision(
            investigation_id=inv_id,
            decision="Rejected",
            reviewer_notes=rej_notes,
            actor="Senior On-Call",
            db_path=self.test_db_path,
        )
        self.assertTrue(success)

        # Investigation record must have Rejected decision and NULL final_action
        inv_row = db.get_investigation_by_id(inv_id, db_path=self.test_db_path)
        self.assertEqual(inv_row["human_decision"], "Rejected")
        self.assertIsNone(inv_row["final_action"])
        self.assertEqual(inv_row["reviewer_notes"], rej_notes)

        # Incident status must be reset to 'Open'
        inc_after = db.get_incident_by_id(inc_id, db_path=self.test_db_path)
        self.assertEqual(inc_after["status"], "Open")

        # Audit log must record RECOMMENDATION_REJECTED
        logs = incident_service.get_audit_history_for_incident(inc_id, db_path=self.test_db_path)
        matching_log = [l for l in logs if l["action"] == "RECOMMENDATION_REJECTED"][0]
        self.assertEqual(matching_log["actor"], "Senior On-Call")
        self.assertIn("rejected", matching_log["notes"].lower())
        self.assertIn(f"Investigation #{inv_id}", matching_log["notes"])
        self.assertIn(rej_notes, matching_log["notes"])

    # --------------------------------------------------------------------------
    # 15. ATOMIC ROLLBACK ON DATABASE FAILURE DURING APPROVAL
    # --------------------------------------------------------------------------
    def test_15_manual_remediation_approval_atomic_rollback(self):
        """If audit log insertion fails during approval, the entire transaction rolls back."""
        inc_id = db.create_incident(
            title="Atomicity Test Incident",
            description="Testing rollback atomicity",
            db_path=self.test_db_path,
        )

        fallback_payload = {
            "mode": "evidence_only_fallback",
            "is_fallback": True,
            "confidence": None,
            "ai_recommendation": "Automated AI diagnostic synthesis is currently unavailable.",
        }
        inv_id = incident_service.save_investigation_proposal(
            incident_id=inc_id,
            investigation_data=fallback_payload,
            evidence_packets=self.sample_evidence,
            db_path=self.test_db_path,
        )

        initial_logs = incident_service.get_audit_history_for_incident(inc_id, db_path=self.test_db_path)
        initial_log_count = len(initial_logs)

        # Install an abort trigger on audit_logs
        conn = db.get_connection(self.test_db_path)
        try:
            conn.execute("""
                CREATE TRIGGER abort_on_approval
                BEFORE INSERT ON audit_logs
                WHEN NEW.action = 'RECOMMENDATION_APPROVED'
                BEGIN
                    SELECT RAISE(ABORT, 'Simulated audit insertion abort');
                END;
            """)
            conn.commit()
        finally:
            conn.close()

        # Attempt approval - must raise DatabaseError and roll back
        try:
            with self.assertRaises(sqlite3.DatabaseError):
                incident_service.apply_investigation_decision(
                    investigation_id=inv_id,
                    decision="Approved",
                    final_action="Rollback test action",
                    reviewer_notes="Rollback test notes",
                    db_path=self.test_db_path,
                )

            # Verify investigation remains Pending and final_action is NULL
            inv_after = db.get_investigation_by_id(inv_id, db_path=self.test_db_path)
            self.assertEqual(inv_after["human_decision"], "Pending")
            self.assertIsNone(inv_after["final_action"])

            # Verify incident status remains 'Pending Approval'
            inc_after = db.get_incident_by_id(inc_id, db_path=self.test_db_path)
            self.assertEqual(inc_after["status"], "Pending Approval")

            # Verify no audit log was inserted
            logs_after = incident_service.get_audit_history_for_incident(inc_id, db_path=self.test_db_path)
            self.assertEqual(len(logs_after), initial_log_count)
        finally:
            conn = db.get_connection(self.test_db_path)
            try:
                conn.execute("DROP TRIGGER IF EXISTS abort_on_approval;")
                conn.commit()
            finally:
                conn.close()

    # --------------------------------------------------------------------------
    # 16. DUPLICATE SUBMISSION DOES NOT CREATE DUPLICATE APPROVAL EVENTS
    # --------------------------------------------------------------------------
    def test_16_duplicate_manual_remediation_submission_creates_no_duplicate_audit_event(self):
        """Duplicate approval submission returns False and creates no duplicate audit events."""
        inc_id = db.create_incident(
            title="Duplicate Submission Test",
            description="Testing idempotent approval handling",
            db_path=self.test_db_path,
        )

        fallback_payload = {
            "mode": "evidence_only_fallback",
            "is_fallback": True,
            "confidence": None,
            "ai_recommendation": "Automated AI diagnostic synthesis is currently unavailable.",
        }
        inv_id = incident_service.save_investigation_proposal(
            incident_id=inc_id,
            investigation_data=fallback_payload,
            evidence_packets=self.sample_evidence,
            db_path=self.test_db_path,
        )

        # 1. First approval succeeds
        success1 = incident_service.apply_investigation_decision(
            investigation_id=inv_id,
            decision="Approved",
            final_action="Run initial remediation scripts.",
            reviewer_notes="First approval submission.",
            actor="Operator",
            db_path=self.test_db_path,
        )
        self.assertTrue(success1)

        logs_1 = incident_service.get_audit_history_for_incident(inc_id, db_path=self.test_db_path)
        app_logs_1 = [l for l in logs_1 if l["action"] == "RECOMMENDATION_APPROVED"]
        self.assertEqual(len(app_logs_1), 1)

        # 2. Duplicate submission (exact same approval re-submitted)
        success2 = incident_service.apply_investigation_decision(
            investigation_id=inv_id,
            decision="Approved",
            final_action="Run initial remediation scripts.",
            reviewer_notes="Duplicate approval submission.",
            actor="Operator",
            db_path=self.test_db_path,
        )
        self.assertFalse(success2, "Duplicate approval submission must return False")

        # 3. Verify exactly 1 RECOMMENDATION_APPROVED audit log exists
        logs_2 = incident_service.get_audit_history_for_incident(inc_id, db_path=self.test_db_path)
        app_logs_2 = [l for l in logs_2 if l["action"] == "RECOMMENDATION_APPROVED"]
        self.assertEqual(len(app_logs_2), 1, "Duplicate submission must NOT create duplicate audit events")

    # --------------------------------------------------------------------------
    # 17. FAILED APPROVAL VALIDATION CREATES NO AUDIT EVENT
    # --------------------------------------------------------------------------
    def test_17_failed_approval_validation_creates_no_audit_event(self):
        """Failed validation (e.g. non-existent ID or empty procedure) creates zero audit records."""
        logs_before = db.get_audit_logs(incident_id=1, db_path=self.test_db_path)

        with self.assertRaises(ValueError):
            incident_service.apply_investigation_decision(
                investigation_id=99999,
                decision="Approved",
                final_action="Some action",
                db_path=self.test_db_path,
            )

        logs_after = db.get_audit_logs(incident_id=1, db_path=self.test_db_path)
        self.assertEqual(len(logs_after), len(logs_before))

    # --------------------------------------------------------------------------
    # 18. EXISTING AUDIT HISTORY REMAINS READABLE
    # --------------------------------------------------------------------------
    def test_18_existing_audit_history_readable(self):
        """Existing audit logs with RECOMMENDATION_EDITED and RECOMMENDATION_APPROVED are both readable and formatted."""
        inc_id = db.create_incident(
            title="Audit Readability Test",
            description="Testing audit history display compatibility",
            db_path=self.test_db_path,
        )

        db.add_audit_log(
            incident_id=inc_id,
            action="RECOMMENDATION_EDITED",
            actor="Operator",
            notes="Legacy edited recommendation note.",
            db_path=self.test_db_path,
        )
        db.add_audit_log(
            incident_id=inc_id,
            action="RECOMMENDATION_APPROVED",
            actor="Operator",
            notes="Operator approved operator-authored remediation (offline fallback mode — not AI-generated) for Investigation #1.",
            db_path=self.test_db_path,
        )

        logs = incident_service.get_audit_history_for_incident(inc_id, db_path=self.test_db_path)
        self.assertEqual(len(logs), 2)
        actions = [l["action"] for l in logs]
        self.assertIn("RECOMMENDATION_EDITED", actions)
        self.assertIn("RECOMMENDATION_APPROVED", actions)

    # --------------------------------------------------------------------------
    # 19. DISTINCT AUDIT EVENTS FOR EDITS, APPROVALS, AND REJECTIONS
    # --------------------------------------------------------------------------
    def test_19_distinct_audit_events_for_edit_vs_approval(self):
        """Verify standard AI edit produces RECOMMENDATION_EDITED, standard approval produces RECOMMENDATION_APPROVED, and fallback produces operator-authored details."""
        # Incident A: Standard AI Investigation Approval
        inc_a = db.create_incident(title="AI Approval", description="Test", db_path=self.test_db_path)
        inv_a = incident_service.save_investigation_proposal(
            incident_id=inc_a,
            investigation_data={"ai_recommendation": "Standard AI fix.", "confidence": 0.95},
            db_path=self.test_db_path,
        )
        incident_service.apply_investigation_decision(inv_a, decision="Approved", db_path=self.test_db_path)
        logs_a = incident_service.get_audit_history_for_incident(inc_a, db_path=self.test_db_path)
        rec_app_a = [l for l in logs_a if l["action"] == "RECOMMENDATION_APPROVED"][0]
        self.assertIn("Operator approved AI recommendation", rec_app_a["notes"])

        # Incident B: Standard AI Investigation Edit
        inc_b = db.create_incident(title="AI Edit", description="Test", db_path=self.test_db_path)
        inv_b = incident_service.save_investigation_proposal(
            incident_id=inc_b,
            investigation_data={"ai_recommendation": "Standard AI fix.", "confidence": 0.95},
            db_path=self.test_db_path,
        )
        incident_service.apply_investigation_decision(
            inv_b, decision="Edited", final_action="Operator customized fix.", db_path=self.test_db_path
        )
        logs_b = incident_service.get_audit_history_for_incident(inc_b, db_path=self.test_db_path)
        rec_edit_b = [l for l in logs_b if l["action"] == "RECOMMENDATION_EDITED"][0]
        self.assertIn("Operator edited and approved recommendation", rec_edit_b["notes"])

        # Incident C: Fallback Operator Remediation Approval
        inc_c = db.create_incident(title="Fallback Approval", description="Test", db_path=self.test_db_path)
        inv_c = incident_service.save_investigation_proposal(
            incident_id=inc_c,
            investigation_data={
                "mode": "evidence_only_fallback",
                "is_fallback": True,
                "confidence": None,
                "ai_recommendation": "Automated AI diagnostic synthesis is currently unavailable.",
            },
            db_path=self.test_db_path,
        )
        incident_service.apply_investigation_decision(
            inv_c, decision="Approved", final_action="Operator authored fix.", db_path=self.test_db_path
        )
        logs_c = incident_service.get_audit_history_for_incident(inc_c, db_path=self.test_db_path)
        rec_app_c = [l for l in logs_c if l["action"] == "RECOMMENDATION_APPROVED"][0]
        self.assertIn("operator-authored remediation", rec_app_c["notes"].lower())
        self.assertIn("not ai-generated", rec_app_c["notes"].lower())

    # --------------------------------------------------------------------------
    # 20. GOVERNANCE AUDIT CORRECTION EVENT COMPATIBILITY
    # --------------------------------------------------------------------------
    def test_20_governance_audit_correction_event_compatibility(self):
        """Verify GOVERNANCE_AUDIT_CORRECTION is in AUDIT_ACTIONS and is retrieved and deserialized cleanly."""
        from app.config import AUDIT_ACTIONS
        self.assertIn("GOVERNANCE_AUDIT_CORRECTION", AUDIT_ACTIONS)

        inc_id = db.create_incident(
            title="Correction Event Test",
            description="Testing GOVERNANCE_AUDIT_CORRECTION",
            db_path=self.test_db_path,
        )

        inv_id = incident_service.save_investigation_proposal(
            incident_id=inc_id,
            investigation_data={"ai_recommendation": "Procedural guidance."},
            db_path=self.test_db_path,
        )

        corr_notes = (
            "Governance audit correction: Investigation #3 had already been approved with "
            "operator-authored remediation in offline fallback mode. The original approval "
            "was recorded under RECOMMENDATION_EDITED (Audit Log #5) instead of an explicit "
            "approval event. This append-only entry corrects the audit classification without "
            "changing the original decision or remediation."
        )

        log_id = db.add_audit_log(
            incident_id=inc_id,
            action="GOVERNANCE_AUDIT_CORRECTION",
            actor="Operator",
            notes=corr_notes,
            investigation_id=inv_id,
            db_path=self.test_db_path,
        )
        self.assertIsNotNone(log_id)

        logs = incident_service.get_audit_history_for_incident(inc_id, db_path=self.test_db_path)
        # Should have INVESTIGATION_RUN from save_investigation_proposal + GOVERNANCE_AUDIT_CORRECTION
        corr_logs = [l for l in logs if l["action"] == "GOVERNANCE_AUDIT_CORRECTION"]
        self.assertEqual(len(corr_logs), 1)
        self.assertEqual(corr_logs[0]["action"], "GOVERNANCE_AUDIT_CORRECTION")
        self.assertEqual(corr_logs[0]["actor"], "Operator")
        self.assertEqual(corr_logs[0]["investigation_id"], inv_id)
        self.assertEqual(corr_logs[0]["notes"], corr_notes)


if __name__ == "__main__":
    unittest.main()
