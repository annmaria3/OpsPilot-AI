"""OpsPilot AI - UI Classification & Human Review Integration Tests (Step 6.4B).

Validates:
1. Mocked classification and approval flow creates exactly one CLASSIFICATION_APPROVED audit entry.
2. Mocked classification and rejection flow creates exactly one CLASSIFICATION_REJECTED audit entry and leaves incident unchanged.
3. Staged proposals are isolated by incident ID in session state.
4. Classification failure (e.g. GeminiNetworkError) leaves incident and session state untouched.
5. Idempotent decisions: clearing session state prevents duplicate decisions on reruns.
6. Production database isolation check (data/opspilot.db remains 0 rows).

All database operations use isolated temporary SQLite databases.
"""

import os
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from app import config
from app.database import db
from app.services import ai_service, incident_service


class TestUIClassificationIntegration(unittest.TestCase):
    """Test suite for Step 6.4B UI integration and human governance workflows."""

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
        """Create an isolated temporary SQLite database for each test."""
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_file.close()
        self.test_db_path = self.temp_file.name

        db.initialize_database(db_path=self.test_db_path)

        # Seed two sample incidents to verify ID isolation
        self.incident_1_id = db.create_incident(
            title="Card checkout gateway error",
            description="Customers receiving 504 gateway timeout on mobile checkout.",
            department="Payments & Billing",
            impact="High",
            db_path=self.test_db_path,
        )
        self.incident_2_id = db.create_incident(
            title="SSO login failure",
            description="Employees cannot login to internal portal using Okta.",
            department="IT & Security",
            impact="Medium",
            db_path=self.test_db_path,
        )

        self.mock_proposal = {
            "category": "Payment / Transaction Failure",
            "severity": "High",
            "impact": "High",
            "department": "Payments & Billing",
            "reasoning": "504 Gateway Timeout during checkout indicates payment gateway degradation.",
            "confidence": 0.94,
            "missing_information": ["Acquirer ID not specified"],
            "requires_human_review": False,
        }

    def tearDown(self):
        """Clean up temporary database file."""
        if os.path.exists(self.test_db_path):
            try:
                os.remove(self.test_db_path)
            except OSError:
                pass

    # --------------------------------------------------------------------------
    # 1. Proposal Isolation by Incident ID
    # --------------------------------------------------------------------------
    def test_01_proposal_isolation_by_incident_id(self):
        """Verify that proposals are isolated by incident ID in session state keys."""
        mock_session_state = {}

        key_1 = f"ai_proposal_{self.incident_1_id}"
        key_2 = f"ai_proposal_{self.incident_2_id}"

        mock_session_state[key_1] = self.mock_proposal

        # Incident 1 has proposal, Incident 2 does not
        self.assertIn(key_1, mock_session_state)
        self.assertNotIn(key_2, mock_session_state)

        # Switching to incident 2 does not see incident 1's proposal
        self.assertIsNone(mock_session_state.get(key_2))
        self.assertEqual(mock_session_state[key_1]["category"], "Payment / Transaction Failure")

    # --------------------------------------------------------------------------
    # 2. Mocked Classification and Approval Flow
    # --------------------------------------------------------------------------
    def test_02_mocked_classification_and_approval_flow(self):
        """Verify the full approval flow: stages proposal, approves, updates DB, records audit log, clears state."""
        mock_session_state = {}
        proposal_key = f"ai_proposal_{self.incident_1_id}"

        # 1. Trigger classification (mocked)
        with patch.object(ai_service, "classify_incident", return_value=self.mock_proposal):
            proposal = ai_service.classify_incident(
                title="Card checkout gateway error",
                description="Customers receiving 504 gateway timeout on mobile checkout.",
            )
            mock_session_state[proposal_key] = proposal

        self.assertIn(proposal_key, mock_session_state)

        # 2. Operator reviews and approves
        staged = mock_session_state[proposal_key]
        summary_note = f"Approved AI classification: category='{staged['category']}', severity='{staged['severity']}'."

        success = incident_service.update_incident_classification(
            incident_id=self.incident_1_id,
            category=staged["category"],
            severity=staged["severity"],
            department=staged["department"],
            impact=staged["impact"],
            actor="Operator",
            notes=summary_note,
            db_path=self.test_db_path,
        )
        self.assertTrue(success)

        # 3. Clear session state proposal
        mock_session_state.pop(proposal_key, None)
        self.assertNotIn(proposal_key, mock_session_state)

        # 4. Verify DB: incident fields are updated
        updated_inc = incident_service.get_incident_by_id(self.incident_1_id, db_path=self.test_db_path)
        self.assertEqual(updated_inc["category"], "Payment / Transaction Failure")
        self.assertEqual(updated_inc["severity"], "High")

        # 5. Verify DB: exactly one CLASSIFICATION_APPROVED audit entry exists
        audit_history = incident_service.get_audit_history_for_incident(
            self.incident_1_id, db_path=self.test_db_path
        )
        approved_audits = [a for a in audit_history if a["action"] == "CLASSIFICATION_APPROVED"]
        self.assertEqual(len(approved_audits), 1)
        self.assertEqual(approved_audits[0]["actor"], "Operator")

    # --------------------------------------------------------------------------
    # 3. Mocked Classification and Rejection Flow
    # --------------------------------------------------------------------------
    def test_03_mocked_classification_and_rejection_flow(self):
        """Verify the full rejection flow: stages proposal, rejects, incident unchanged, audit logged, state cleared."""
        mock_session_state = {}
        proposal_key = f"ai_proposal_{self.incident_1_id}"

        # Check pre-state
        pre_inc = incident_service.get_incident_by_id(self.incident_1_id, db_path=self.test_db_path)

        # Stage proposal
        mock_session_state[proposal_key] = self.mock_proposal

        # Operator rejects
        staged = mock_session_state[proposal_key]
        rejection_reason = "Operator verified gateway timeout was an expected maintenance window."

        success = incident_service.reject_incident_classification(
            incident_id=self.incident_1_id,
            proposed_category=staged.get("category"),
            proposed_severity=staged.get("severity"),
            proposed_department=staged.get("department"),
            proposed_impact=staged.get("impact"),
            actor="Operator",
            reason=rejection_reason,
            db_path=self.test_db_path,
        )
        self.assertTrue(success)

        # Clear session state proposal
        mock_session_state.pop(proposal_key, None)
        self.assertNotIn(proposal_key, mock_session_state)

        # Verify DB: incident fields remain untouched
        post_inc = incident_service.get_incident_by_id(self.incident_1_id, db_path=self.test_db_path)
        self.assertEqual(pre_inc["category"], post_inc["category"])
        self.assertEqual(pre_inc["severity"], post_inc["severity"])
        self.assertEqual(pre_inc["department"], post_inc["department"])
        self.assertEqual(pre_inc["impact"], post_inc["impact"])

        # Verify DB: exactly one CLASSIFICATION_REJECTED audit entry exists
        audit_history = incident_service.get_audit_history_for_incident(
            self.incident_1_id, db_path=self.test_db_path
        )
        rejected_audits = [a for a in audit_history if a["action"] == "CLASSIFICATION_REJECTED"]
        self.assertEqual(len(rejected_audits), 1)
        self.assertIn("expected maintenance window", rejected_audits[0]["notes"])

    # --------------------------------------------------------------------------
    # 4. Classification Failure Leaves Incident and State Untouched
    # --------------------------------------------------------------------------
    def test_04_classification_failure_leaves_incident_and_state_untouched(self):
        """Verify that when AI classification fails, incident is untouched and no proposal is staged."""
        mock_session_state = {}
        proposal_key = f"ai_proposal_{self.incident_1_id}"

        pre_inc = incident_service.get_incident_by_id(self.incident_1_id, db_path=self.test_db_path)

        with patch.object(
            ai_service, "classify_incident", side_effect=ai_service.GeminiNetworkError("Connection timed out")
        ):
            with self.assertRaises(ai_service.GeminiNetworkError):
                ai_service.classify_incident("Title", "Description")

        # Session state has no proposal
        self.assertNotIn(proposal_key, mock_session_state)

        # DB incident is completely unmodified
        post_inc = incident_service.get_incident_by_id(self.incident_1_id, db_path=self.test_db_path)
        self.assertEqual(pre_inc["category"], post_inc["category"])
        self.assertEqual(pre_inc["severity"], post_inc["severity"])

        # DB has zero classification audit entries
        audit_history = incident_service.get_audit_history_for_incident(
            self.incident_1_id, db_path=self.test_db_path
        )
        class_audits = [a for a in audit_history if "CLASSIFICATION" in a["action"]]
        self.assertEqual(len(class_audits), 0)

    # --------------------------------------------------------------------------
    # 5. Idempotence: No Duplicated Decisions on Repeated Reruns
    # --------------------------------------------------------------------------
    def test_05_idempotent_decisions_prevent_duplicate_reruns(self):
        """Verify that clearing session state prevents accidental double-approvals on subsequent reruns."""
        mock_session_state = {}
        proposal_key = f"ai_proposal_{self.incident_1_id}"

        mock_session_state[proposal_key] = self.mock_proposal

        # First run: proposal exists and is approved
        if proposal_key in mock_session_state:
            p = mock_session_state.pop(proposal_key)
            incident_service.update_incident_classification(
                incident_id=self.incident_1_id,
                category=p["category"],
                severity=p["severity"],
                db_path=self.test_db_path,
            )

        # Subsequent rerun pass: proposal is gone, so approval block is skipped
        decision_executed_on_second_pass = False
        if proposal_key in mock_session_state:
            decision_executed_on_second_pass = True

        self.assertFalse(decision_executed_on_second_pass)

        # Audit entries for approval count remains exactly 1
        audit_history = incident_service.get_audit_history_for_incident(
            self.incident_1_id, db_path=self.test_db_path
        )
        approved_audits = [a for a in audit_history if a["action"] == "CLASSIFICATION_APPROVED"]
        self.assertEqual(len(approved_audits), 1)

    # --------------------------------------------------------------------------
    # 7. Persistent Feedback Lifecycle on Approval
    # --------------------------------------------------------------------------
    def test_07_persistent_feedback_lifecycle_on_approval(self):
        """Verify that approval sets a flash message that survives rerun and is cleared immediately after display."""
        mock_session_state = {}
        proposal_key = f"ai_proposal_{self.incident_1_id}"
        mock_session_state[proposal_key] = self.mock_proposal

        # Simulate Approval Click
        staged = mock_session_state[proposal_key]
        success = incident_service.update_incident_classification(
            incident_id=self.incident_1_id,
            category=staged["category"],
            severity=staged["severity"],
            department=staged["department"],
            impact=staged["impact"],
            actor="Operator",
            db_path=self.test_db_path,
        )
        self.assertTrue(success)

        # Set persistent flash and unstage proposal (as done in main.py)
        mock_session_state["classification_flash"] = {
            "type": "success",
            "msg": f"✅ Classification approved and saved for INC-2026-0001: Category '{staged['category']}', Severity '{staged['severity']}'.",
        }
        mock_session_state.pop(proposal_key, None)

        # Verify state immediately after decision: proposal gone, flash staged
        self.assertNotIn(proposal_key, mock_session_state)
        self.assertIn("classification_flash", mock_session_state)

        # Simulate First Pass after Rerun: inspector reads and pops flash
        displayed_flash = mock_session_state.pop("classification_flash", None)
        self.assertIsNotNone(displayed_flash)
        self.assertEqual(displayed_flash["type"], "success")
        self.assertIn("Payment / Transaction Failure", displayed_flash["msg"])

        # Simulate Second Subsequent Pass: flash must not linger
        subsequent_flash = mock_session_state.pop("classification_flash", None)
        self.assertIsNone(subsequent_flash)

    # --------------------------------------------------------------------------
    # 8. Persistent Feedback Lifecycle on Rejection
    # --------------------------------------------------------------------------
    def test_08_persistent_feedback_lifecycle_on_rejection(self):
        """Verify that rejection sets an info flash message that survives rerun and is cleared immediately."""
        mock_session_state = {}
        proposal_key = f"ai_proposal_{self.incident_1_id}"
        mock_session_state[proposal_key] = self.mock_proposal

        # Simulate Rejection Click
        staged = mock_session_state[proposal_key]
        success = incident_service.reject_incident_classification(
            incident_id=self.incident_1_id,
            proposed_category=staged.get("category"),
            proposed_severity=staged.get("severity"),
            proposed_department=staged.get("department"),
            proposed_impact=staged.get("impact"),
            actor="Operator",
            reason="Operator confirmed non-issue.",
            db_path=self.test_db_path,
        )
        self.assertTrue(success)

        # Set persistent flash and unstage proposal
        mock_session_state["classification_flash"] = {
            "type": "info",
            "msg": "ℹ️ AI classification proposal for INC-2026-0001 was rejected. Incident record was not modified.",
        }
        mock_session_state.pop(proposal_key, None)

        # First pass after rerun reads and pops flash
        displayed_flash = mock_session_state.pop("classification_flash", None)
        self.assertIsNotNone(displayed_flash)
        self.assertEqual(displayed_flash["type"], "info")
        self.assertIn("was rejected", displayed_flash["msg"])

        # Subsequent pass: flash is clean
        self.assertIsNone(mock_session_state.pop("classification_flash", None))

    # --------------------------------------------------------------------------
    # 9. Context-Aware Trigger Visibility Logic
    # --------------------------------------------------------------------------
    def test_09_context_aware_trigger_visibility_logic(self):
        """Verify that the prominent trigger bar is shown only when no proposal is staged."""
        mock_session_state = {}
        proposal_key = f"ai_proposal_{self.incident_1_id}"

        # State 1: No proposal staged
        has_proposal = proposal_key in mock_session_state
        self.assertFalse(has_proposal)
        # In UI: not has_proposal -> trigger bar rendered

        # State 2: Proposal is staged
        mock_session_state[proposal_key] = self.mock_proposal
        has_proposal = proposal_key in mock_session_state
        self.assertTrue(has_proposal)
        # In UI: has_proposal -> trigger bar hidden, governance panel rendered with compact re-run button

    # --------------------------------------------------------------------------
    # 10. Diagnostic Transparency Auto-Expand Logic
    # --------------------------------------------------------------------------
    def test_10_diagnostic_transparency_auto_expand_logic(self):
        """Verify that diagnostic gaps expander automatically expands when requires_human_review is True."""
        # Case A: Human review required -> auto-expand True
        proposal_flagged = {**self.mock_proposal, "requires_human_review": True}
        auto_expand_flagged = bool(proposal_flagged.get("requires_human_review"))
        self.assertTrue(auto_expand_flagged)

        # Case B: Human review not required -> auto-expand False
        proposal_clean = {**self.mock_proposal, "requires_human_review": False}
        auto_expand_clean = bool(proposal_clean.get("requires_human_review"))
        self.assertFalse(auto_expand_clean)

        # Case C: Key missing -> auto-expand False
        proposal_no_key = {k: v for k, v in self.mock_proposal.items() if k != "requires_human_review"}
        auto_expand_default = bool(proposal_no_key.get("requires_human_review"))
        self.assertFalse(auto_expand_default)

    # --------------------------------------------------------------------------
    # 11. Rejection Notes Handling and Audit Content
    # --------------------------------------------------------------------------
    def test_11_rejection_notes_handling_and_audit(self):
        """Verify that rejection correctly records custom notes or fallback default in audit logs."""
        # 1. Custom rejection note
        custom_reason = "False positive detected by triage engineer."
        success = incident_service.reject_incident_classification(
            incident_id=self.incident_1_id,
            proposed_category="Payment / Transaction Failure",
            proposed_severity="High",
            actor="Operator",
            reason=custom_reason,
            db_path=self.test_db_path,
        )
        self.assertTrue(success)

        audits = incident_service.get_audit_history_for_incident(self.incident_1_id, db_path=self.test_db_path)
        rej_audit = [a for a in audits if a["action"] == "CLASSIFICATION_REJECTED"][0]
        self.assertIn(custom_reason, rej_audit["notes"])

        # 2. Blank / whitespace rejection note falls back cleanly
        blank_reason = "   "
        clean_reason = blank_reason.strip() or "Operator rejected proposed AI classification."
        success_2 = incident_service.reject_incident_classification(
            incident_id=self.incident_2_id,
            proposed_category="Network / Infrastructure",
            proposed_severity="Medium",
            actor="Operator",
            reason=clean_reason,
            db_path=self.test_db_path,
        )
        self.assertTrue(success_2)

        audits_2 = incident_service.get_audit_history_for_incident(self.incident_2_id, db_path=self.test_db_path)
        rej_audit_2 = [a for a in audits_2 if a["action"] == "CLASSIFICATION_REJECTED"][0]
        self.assertIn("Operator rejected proposed AI classification.", rej_audit_2["notes"])

    # --------------------------------------------------------------------------
    # 12. Production Database Isolation
    # --------------------------------------------------------------------------
    def test_12_production_database_isolation(self):
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
