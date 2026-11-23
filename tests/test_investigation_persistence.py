"""
Unit and Integration Tests for OpsPilot AI - Investigation Persistence & Atomic Audit (Step 7.4B).

Validates:
1. Atomic proposal persistence in investigations table with INVESTIGATION_RUN audit log.
2. Atomic rollback on database failure (neither investigation nor audit log persisted).
3. Atomic approval workflow updating final_action = ai_recommendation and logging RECOMMENDATION_APPROVED.
4. Atomic edit-and-approve workflow updating final_action to operator text while preserving ai_recommendation untouched.
5. Atomic rejection workflow setting final_action = NULL and logging RECOMMENDATION_REJECTED.
6. Rejection after prior approval clearing final_action to NULL (verifying no COALESCE leakage).
7. Decision-state validation, ID consistency checks, and rejection of invalid transitions.
8. Evidence provenance serialization and deserialization preserving all 9 chunk attributes.
9. Backward compatibility of deserialize_investigation with legacy plain-text and simple list data.
10. Database trigger enforcement of immutable audit logs.
11. Atomic decision rollback on audit insertion failure (retaining exact decision, final_action, notes, and incident status).
12. Complete production database isolation (data/opspilot.db untouched).

All tests run against temporary SQLite databases.
"""

import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

from app.database import db
from app.services import incident_service


class TestInvestigationPersistence(unittest.TestCase):
    """Test suite for investigation persistence, atomic auditing, and decision workflows."""

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
        """Create a fresh temporary database for each test with initialized schema."""
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_path = Path(self.temp_file.name)
        self.temp_file.close()

        # Initialize schema and triggers
        db.initialize_database(db_path=self.temp_path)

        # Create a sample baseline incident
        self.incident = incident_service.create_incident(
            title="Database Connection Pool Exhaustion",
            description="Users report 504 Gateway Timeout on checkout API.",
            category="Database Connectivity / Query Failure",
            severity="Critical",
            department="Engineering & DevOps",
            impact="Critical",
            actor="MonitoringSystem",
            db_path=self.temp_path,
        )
        self.incident_id = self.incident["id"]

        # Sample 9-attribute evidence chunks
        self.sample_evidence = [
            {
                "chunk_id": 42,
                "document_id": 3,
                "document_name": "Runbook_Postgres_Pool.md",
                "page_number": 1,
                "chunk_index": 2,
                "rank": 1,
                "citation": "[Runbook_Postgres_Pool.md, p.1 #2]",
                "citation_full": "Runbook: Runbook_Postgres_Pool.md (Page 1, Chunk 2)",
                "bm25_score": -3.854,
                "relevance_score": 3.854,
                "content": "When connections exceed 95%, pool exhaustion occurs. Recommended fix: scale max_connections.",
            },
            {
                "chunk_id": 43,
                "document_id": 3,
                "document_name": "Runbook_Postgres_Pool.md",
                "page_number": 2,
                "chunk_index": 3,
                "rank": 2,
                "citation": "[Runbook_Postgres_Pool.md, p.2 #3]",
                "citation_full": "Runbook: Runbook_Postgres_Pool.md (Page 2, Chunk 3)",
                "bm25_score": -2.140,
                "relevance_score": 2.140,
                "content": "Terminate idle-in-transaction worker threads via pg_terminate_backend.",
            },
        ]

        # Conforming investigation data dictionary
        self.investigation_data = {
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
                    "unsupported_claims": []
                }
            ],
            "evidence_citations": [
                "[Runbook_Postgres_Pool.md, p.1 #2]",
                "[Runbook_Postgres_Pool.md, p.2 #3]",
            ],
            "ai_recommendation": "1. Temporarily increase max_connections to 250.\n2. Restart idle worker threads.",
            "confidence": 0.88,
            "missing_information": [
                "Thread pool dump at time of failure"
            ],
            "diagnostic_actions": [
                "SELECT count(*), state FROM pg_stat_activity GROUP BY state;"
            ],
            "requires_human_review": True,
            "citation_validation_summary": {
                "total_citations": 2,
                "valid_citations": 2,
                "flagged_citations": 0,
                "flagged_details": [],
            },
        }

    def tearDown(self):
        """Clean up temporary database file."""
        if self.temp_path.exists():
            try:
                self.temp_path.unlink()
            except PermissionError:
                pass

    # --------------------------------------------------------------------------
    # 1. ATOMIC PROPOSAL PERSISTENCE AND AUDIT
    # --------------------------------------------------------------------------
    def test_01_atomic_proposal_creation_and_audit(self):
        """Verify proposal and INVESTIGATION_RUN audit log are atomically persisted in one transaction."""
        inv_id = incident_service.save_investigation_proposal(
            incident_id=self.incident_id,
            investigation_data=self.investigation_data,
            evidence_packets=self.sample_evidence,
            actor="OpsPilotAI",
            db_path=self.temp_path,
        )
        self.assertIsInstance(inv_id, int)
        self.assertGreater(inv_id, 0)

        # 1. Verify investigation record
        row = db.get_investigation_by_id(inv_id, db_path=self.temp_path)
        self.assertIsNotNone(row)
        self.assertEqual(row["incident_id"], self.incident_id)
        self.assertEqual(row["human_decision"], "Pending")
        self.assertIsNone(row["final_action"])
        self.assertIsNone(row["reviewer_notes"])
        self.assertEqual(row["confidence"], 0.88)
        self.assertEqual(row["ai_recommendation"], self.investigation_data["ai_recommendation"])

        # 2. Verify INVESTIGATION_RUN audit log
        logs = incident_service.get_audit_history_for_incident(self.incident_id, db_path=self.temp_path)
        inv_logs = [l for l in logs if l["action"] == "INVESTIGATION_RUN"]
        self.assertEqual(len(inv_logs), 1)
        self.assertEqual(inv_logs[0]["investigation_id"], inv_id)
        self.assertEqual(inv_logs[0]["actor"], "OpsPilotAI")
        self.assertIn("investigation conducted", inv_logs[0]["notes"].lower())

        # 3. Verify incident status transitioned to 'Pending Approval'
        inc = incident_service.get_incident_by_id(self.incident_id, db_path=self.temp_path)
        self.assertEqual(inc["status"], "Pending Approval")

    # --------------------------------------------------------------------------
    # 2. ATOMIC ROLLBACK ON PROPOSAL FAILURE
    # --------------------------------------------------------------------------
    def test_02_atomic_proposal_rollback_on_failure(self):
        """Verify that failure during proposal creation rolls back both investigation and audit log."""
        # Non-existent incident ID should fail
        with self.assertRaises(ValueError):
            incident_service.save_investigation_proposal(
                incident_id=99999,
                investigation_data=self.investigation_data,
                db_path=self.temp_path,
            )

        # Verify nothing was persisted in investigations or audit_logs for 99999
        conn = db.get_connection(self.temp_path)
        cursor = conn.cursor()
        cursor.execute("SELECT count(*) FROM investigations WHERE incident_id = 99999;")
        self.assertEqual(cursor.fetchone()[0], 0)
        cursor.execute("SELECT count(*) FROM audit_logs WHERE incident_id = 99999;")
        self.assertEqual(cursor.fetchone()[0], 0)
        conn.close()

    # --------------------------------------------------------------------------
    # 3. ATOMIC APPROVAL WORKFLOW
    # --------------------------------------------------------------------------
    def test_03_atomic_approval_sets_final_action_and_audit(self):
        """Verify operator approval atomically sets final_action = ai_recommendation and records audit log."""
        inv_id = incident_service.save_investigation_proposal(
            incident_id=self.incident_id,
            investigation_data=self.investigation_data,
            evidence_packets=self.sample_evidence,
            db_path=self.temp_path,
        )

        # Approve proposal
        success = incident_service.apply_investigation_decision(
            investigation_id=inv_id,
            decision="Approved",
            reviewer_notes="LGTM, proceed with connection increase.",
            actor="LeadSRE",
            db_path=self.temp_path,
        )
        self.assertTrue(success)

        # 1. Verify investigation row
        row = db.get_investigation_by_id(inv_id, db_path=self.temp_path)
        self.assertEqual(row["human_decision"], "Approved")
        self.assertEqual(row["final_action"], self.investigation_data["ai_recommendation"])
        self.assertEqual(row["reviewer_notes"], "LGTM, proceed with connection increase.")

        # 2. Verify RECOMMENDATION_APPROVED audit log
        logs = incident_service.get_audit_history_for_incident(self.incident_id, db_path=self.temp_path)
        app_logs = [l for l in logs if l["action"] == "RECOMMENDATION_APPROVED"]
        self.assertEqual(len(app_logs), 1)
        self.assertEqual(app_logs[0]["investigation_id"], inv_id)
        self.assertEqual(app_logs[0]["actor"], "LeadSRE")

        # 3. Verify incident status updated to 'Approved'
        inc = incident_service.get_incident_by_id(self.incident_id, db_path=self.temp_path)
        self.assertEqual(inc["status"], "Approved")

    # --------------------------------------------------------------------------
    # 4. ATOMIC EDIT-AND-APPROVE WORKFLOW
    # --------------------------------------------------------------------------
    def test_04_atomic_edit_and_approval_preserves_original_recommendation(self):
        """Verify editing recommendation stores modified text in final_action while preserving ai_recommendation untouched."""
        inv_id = incident_service.save_investigation_proposal(
            incident_id=self.incident_id,
            investigation_data=self.investigation_data,
            evidence_packets=self.sample_evidence,
            db_path=self.temp_path,
        )

        edited_text = "1. Scale max_connections to 300.\n2. Do NOT restart workers; kill idle sessions only."

        success = incident_service.apply_investigation_decision(
            investigation_id=inv_id,
            decision="Edited",
            final_action=edited_text,
            reviewer_notes="Modified max_connections threshold based on database instance capacity.",
            actor="SeniorDBA",
            db_path=self.temp_path,
        )
        self.assertTrue(success)

        # Verify investigation row
        row = db.get_investigation_by_id(inv_id, db_path=self.temp_path)
        self.assertEqual(row["human_decision"], "Approved")
        self.assertEqual(row["final_action"], edited_text)
        # CRITICAL: original ai_recommendation must remain untouched
        self.assertEqual(row["ai_recommendation"], self.investigation_data["ai_recommendation"])
        self.assertNotEqual(row["ai_recommendation"], row["final_action"])

        # Verify RECOMMENDATION_EDITED audit log
        logs = incident_service.get_audit_history_for_incident(self.incident_id, db_path=self.temp_path)
        edit_logs = [l for l in logs if l["action"] == "RECOMMENDATION_EDITED"]
        self.assertEqual(len(edit_logs), 1)
        self.assertEqual(edit_logs[0]["investigation_id"], inv_id)
        self.assertEqual(edit_logs[0]["actor"], "SeniorDBA")

        # Incident status updated to 'Approved'
        inc = incident_service.get_incident_by_id(self.incident_id, db_path=self.temp_path)
        self.assertEqual(inc["status"], "Approved")

    # --------------------------------------------------------------------------
    # 5. ATOMIC REJECTION WORKFLOW (EXPLICIT NULL ON FINAL_ACTION)
    # --------------------------------------------------------------------------
    def test_05_atomic_rejection_explicitly_clears_final_action_to_null(self):
        """Verify rejection sets human_decision = 'Rejected', final_action = NULL, and updates incident status to 'Open'."""
        inv_id = incident_service.save_investigation_proposal(
            incident_id=self.incident_id,
            investigation_data=self.investigation_data,
            evidence_packets=self.sample_evidence,
            db_path=self.temp_path,
        )

        success = incident_service.apply_investigation_decision(
            investigation_id=inv_id,
            decision="Rejected",
            reviewer_notes="Proposed runbook is for legacy v1 cluster; not applicable.",
            actor="LeadSRE",
            db_path=self.temp_path,
        )
        self.assertTrue(success)

        row = db.get_investigation_by_id(inv_id, db_path=self.temp_path)
        self.assertEqual(row["human_decision"], "Rejected")
        # STRICT REQUIREMENT: final_action must be explicitly NULL
        self.assertIsNone(row["final_action"])
        self.assertEqual(row["reviewer_notes"], "Proposed runbook is for legacy v1 cluster; not applicable.")

        # Verify RECOMMENDATION_REJECTED audit log
        logs = incident_service.get_audit_history_for_incident(self.incident_id, db_path=self.temp_path)
        rej_logs = [l for l in logs if l["action"] == "RECOMMENDATION_REJECTED"]
        self.assertEqual(len(rej_logs), 1)
        self.assertEqual(rej_logs[0]["investigation_id"], inv_id)
        self.assertEqual(rej_logs[0]["actor"], "LeadSRE")

        # Incident status reset to 'Open'
        inc = incident_service.get_incident_by_id(self.incident_id, db_path=self.temp_path)
        self.assertEqual(inc["status"], "Open")

    # --------------------------------------------------------------------------
    # 6. REJECTION AFTER PRIOR APPROVAL (PROVE NO COALESCE LEAKAGE)
    # --------------------------------------------------------------------------
    def test_06_rejection_after_prior_approval_clears_final_action(self):
        """Verify that rejecting a previously approved recommendation explicitly resets final_action to NULL."""
        inv_id = incident_service.save_investigation_proposal(
            incident_id=self.incident_id,
            investigation_data=self.investigation_data,
            db_path=self.temp_path,
        )

        # 1. First Approve
        incident_service.apply_investigation_decision(
            investigation_id=inv_id,
            decision="Approved",
            final_action="Temporarily approved action",
            db_path=self.temp_path,
        )
        approved_row = db.get_investigation_by_id(inv_id, db_path=self.temp_path)
        self.assertEqual(approved_row["final_action"], "Temporarily approved action")

        # 2. Then Reject
        incident_service.apply_investigation_decision(
            investigation_id=inv_id,
            decision="Rejected",
            reviewer_notes="Decision overturned upon further inspection.",
            db_path=self.temp_path,
        )
        rejected_row = db.get_investigation_by_id(inv_id, db_path=self.temp_path)

        # Must be genuinely NULL, NOT retaining previous value through COALESCE
        self.assertEqual(rejected_row["human_decision"], "Rejected")
        self.assertIsNone(rejected_row["final_action"])
        self.assertNotEqual(rejected_row["final_action"], "Temporarily approved action")

    # --------------------------------------------------------------------------
    # 7. INVALID DECISION TRANSITIONS AND ID MISMATCHES
    # --------------------------------------------------------------------------
    def test_07_invalid_decision_transitions_and_id_mismatch(self):
        """Verify validation errors and rollbacks on invalid decisions or incident/investigation mismatches."""
        inv_id = incident_service.save_investigation_proposal(
            incident_id=self.incident_id,
            investigation_data=self.investigation_data,
            db_path=self.temp_path,
        )

        # 1. Unsupported decision string
        with self.assertRaises(ValueError) as ctx:
            incident_service.apply_investigation_decision(
                investigation_id=inv_id,
                decision="MaybeLater",
                db_path=self.temp_path,
            )
        self.assertIn("Must be one of", str(ctx.exception))

        # 2. Non-existent investigation ID
        with self.assertRaises(ValueError) as ctx:
            incident_service.apply_investigation_decision(
                investigation_id=99999,
                decision="Approved",
                db_path=self.temp_path,
            )
        self.assertIn("not found", str(ctx.exception))

        # 3. ID mismatch at database layer: investigation belongs to incident A, but caller passes incident B
        other_incident = incident_service.create_incident(
            title="Other Incident",
            description="Separate failure domain.",
            db_path=self.temp_path,
        )
        with self.assertRaises(ValueError) as ctx:
            db.apply_investigation_decision_with_audit(
                investigation_id=inv_id,
                incident_id=other_incident["id"],  # Mismatched incident ID!
                human_decision="Approved",
                action="RECOMMENDATION_APPROVED",
                db_path=self.temp_path,
            )
        self.assertIn("belongs to incident", str(ctx.exception))

        # 4. Incompatible action/decision pair
        with self.assertRaises(ValueError):
            db.apply_investigation_decision_with_audit(
                investigation_id=inv_id,
                incident_id=self.incident_id,
                human_decision="Approved",
                action="RECOMMENDATION_REJECTED",  # Incompatible!
                db_path=self.temp_path,
            )

    # --------------------------------------------------------------------------
    # 8. EVIDENCE PROVENANCE PRESERVED IN SERIALIZATION ROUND TRIP
    # --------------------------------------------------------------------------
    def test_08_evidence_provenance_preserved_in_serialization_round_trip(self):
        """Verify all 9 chunk provenance attributes survive JSON serialization and deserialization."""
        inv_id = incident_service.save_investigation_proposal(
            incident_id=self.incident_id,
            investigation_data=self.investigation_data,
            evidence_packets=self.sample_evidence,
            db_path=self.temp_path,
        )

        unpacked = incident_service.get_investigation_by_id(inv_id, db_path=self.temp_path)
        self.assertIsNotNone(unpacked)

        # Check provenance chunks
        self.assertIn("provenance", unpacked)
        prov = unpacked["provenance"]
        self.assertEqual(len(prov), 2)

        first_chunk = prov[0]
        # Verify 9 core attributes
        required_attrs = [
            "chunk_id", "document_id", "document_name", "page_number",
            "chunk_index", "rank", "citation", "citation_full", "bm25_score"
        ]
        for attr in required_attrs:
            self.assertIn(attr, first_chunk, f"Missing provenance attribute '{attr}'")

        self.assertEqual(first_chunk["chunk_id"], 42)
        self.assertEqual(first_chunk["document_id"], 3)
        self.assertEqual(first_chunk["document_name"], "Runbook_Postgres_Pool.md")
        self.assertEqual(first_chunk["page_number"], 1)
        self.assertEqual(first_chunk["chunk_index"], 2)
        self.assertEqual(first_chunk["rank"], 1)
        self.assertEqual(first_chunk["citation"], "[Runbook_Postgres_Pool.md, p.1 #2]")
        self.assertEqual(first_chunk["bm25_score"], -3.854)

        # Check assumptions and citations
        self.assertEqual(len(unpacked["evidence_citations"]), 2)
        self.assertEqual(len(unpacked["assumptions"]), 1)
        self.assertIn("checkout workers connect directly", unpacked["assumptions"][0])

    # --------------------------------------------------------------------------
    # 9. BACKWARD COMPATIBLE DESERIALIZATION
    # --------------------------------------------------------------------------
    def test_09_backward_compatible_deserialization(self):
        """Verify deserialize_investigation handles legacy plain-text, simple lists, and empty records."""
        # Case A: Legacy plain text citations
        legacy_row_a = {
            "id": 1,
            "incident_id": 1,
            "confirmed_facts": "Fact 1\nFact 2",
            "root_cause_hypotheses": None,
            "evidence_citations": "[Runbook_DB.md, p.1 #2]",
            "ai_recommendation": "Fix it.",
        }
        res_a = incident_service.deserialize_investigation(legacy_row_a)
        self.assertEqual(res_a["confirmed_facts"], ["Fact 1", "Fact 2"])
        self.assertEqual(res_a["evidence_citations"], ["[Runbook_DB.md, p.1 #2]"])
        self.assertEqual(res_a["provenance"], [])
        self.assertEqual(res_a["assumptions"], [])

        # Case B: Legacy simple JSON list
        legacy_row_b = {
            "id": 2,
            "incident_id": 1,
            "confirmed_facts": '["Confirmed fact 1"]',
            "root_cause_hypotheses": '[{"hypothesis": "h1"}]',
            "evidence_citations": '["[Runbook_A.md, p.1 #1]", "[Runbook_B.md, p.2 #1]"]',
            "ai_recommendation": "Fix it.",
        }
        res_b = incident_service.deserialize_investigation(legacy_row_b)
        self.assertEqual(res_b["confirmed_facts"], ["Confirmed fact 1"])
        self.assertEqual(len(res_b["root_cause_hypotheses"]), 1)
        self.assertEqual(len(res_b["evidence_citations"]), 2)
        self.assertEqual(res_b["provenance"], [])

        # Case C: Empty/None row
        res_c = incident_service.deserialize_investigation(None)
        self.assertEqual(res_c, {})

    # --------------------------------------------------------------------------
    # 10. IMMUTABLE AUDIT LOG TRIGGER ENFORCEMENT
    # --------------------------------------------------------------------------
    def test_10_immutable_audit_logs_triggers(self):
        """Verify SQLite database triggers prevent modification or deletion of investigation audit logs."""
        inv_id = incident_service.save_investigation_proposal(
            incident_id=self.incident_id,
            investigation_data=self.investigation_data,
            db_path=self.temp_path,
        )

        conn = db.get_connection(self.temp_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM audit_logs WHERE investigation_id = ?;", (inv_id,))
        log_id = cursor.fetchone()["id"]

        # 1. Attempt UPDATE on audit log -> must be aborted by trigger
        with self.assertRaises(sqlite3.IntegrityError):
            cursor.execute("UPDATE audit_logs SET actor = 'MaliciousHacker' WHERE id = ?;", (log_id,))

        # 2. Attempt DELETE on audit log -> must be aborted by trigger
        with self.assertRaises(sqlite3.IntegrityError):
            cursor.execute("DELETE FROM audit_logs WHERE id = ?;", (log_id,))

        conn.close()

    # --------------------------------------------------------------------------
    # 11. ATOMIC DECISION ROLLBACK ON AUDIT FAILURE
    # --------------------------------------------------------------------------
    def test_11_atomic_decision_rollback_on_audit_failure(self):
        """Verify atomic rollback when audit insertion fails during decision application.

        Ensures that if audit logging fails:
        1. Exception is propagated to caller.
        2. Investigation retains original human_decision, final_action, and reviewer_notes.
        3. Parent incident retains original status.
        4. No new audit log is committed.
        """
        # 1. Create proposal
        inv_id = incident_service.save_investigation_proposal(
            incident_id=self.incident_id,
            investigation_data=self.investigation_data,
            evidence_packets=self.sample_evidence,
            db_path=self.temp_path,
        )

        # 2. Establish a known prior decision state and final_action
        success = incident_service.apply_investigation_decision(
            investigation_id=inv_id,
            decision="Approved",
            final_action="Initial Confirmed Baseline Action",
            reviewer_notes="First pass approval",
            actor="OperatorA",
            db_path=self.temp_path,
        )
        self.assertTrue(success)

        # Verify baseline state before failed transaction attempt
        before_inv = db.get_investigation_by_id(inv_id, db_path=self.temp_path)
        self.assertEqual(before_inv["human_decision"], "Approved")
        self.assertEqual(before_inv["final_action"], "Initial Confirmed Baseline Action")
        self.assertEqual(before_inv["reviewer_notes"], "First pass approval")

        before_inc = incident_service.get_incident_by_id(self.incident_id, db_path=self.temp_path)
        self.assertEqual(before_inc["status"], "Approved")

        before_logs = incident_service.get_audit_history_for_incident(self.incident_id, db_path=self.temp_path)
        baseline_log_count = len(before_logs)

        # 3. Create temporary SQLite trigger that deliberately aborts audit_logs insertion on decision action
        conn = db.get_connection(self.temp_path)
        try:
            conn.execute("""
                CREATE TRIGGER simulate_audit_failure_on_decision
                BEFORE INSERT ON audit_logs
                WHEN NEW.action IN ('RECOMMENDATION_REJECTED', 'RECOMMENDATION_APPROVED', 'RECOMMENDATION_EDITED')
                BEGIN
                    SELECT RAISE(ABORT, 'Simulated audit insertion failure during decision update');
                END;
            """)
            conn.commit()
        finally:
            conn.close()

        # 4 & 5. Attempt decision update (e.g. rejection) and verify exception propagation
        try:
            with self.assertRaises(sqlite3.DatabaseError) as ctx:
                db.apply_investigation_decision_with_audit(
                    investigation_id=inv_id,
                    incident_id=self.incident_id,
                    human_decision="Rejected",
                    reviewer_notes="Attempted second-pass rejection notes",
                    actor="OperatorB",
                    action="RECOMMENDATION_REJECTED",
                    db_path=self.temp_path,
                )
            self.assertIn("Simulated audit insertion failure during decision update", str(ctx.exception))

            # 6. Verify investigation retains its exact original decision, final_action, and notes
            after_inv = db.get_investigation_by_id(inv_id, db_path=self.temp_path)
            self.assertEqual(after_inv["human_decision"], "Approved")
            self.assertEqual(after_inv["final_action"], "Initial Confirmed Baseline Action")
            self.assertEqual(after_inv["reviewer_notes"], "First pass approval")
            self.assertNotEqual(after_inv["human_decision"], "Rejected")
            self.assertIsNotNone(after_inv["final_action"])

            # 7. Verify parent incident retains its original status ('Approved', NOT changed to 'Open')
            after_inc = incident_service.get_incident_by_id(self.incident_id, db_path=self.temp_path)
            self.assertEqual(after_inc["status"], "Approved")
            self.assertNotEqual(after_inc["status"], "Open")

            # 8. Verify no new audit record was committed
            after_logs = incident_service.get_audit_history_for_incident(self.incident_id, db_path=self.temp_path)
            self.assertEqual(len(after_logs), baseline_log_count)

        finally:
            # 9. Clean up temporary trigger and close connection
            conn = db.get_connection(self.temp_path)
            try:
                conn.execute("DROP TRIGGER IF EXISTS simulate_audit_failure_on_decision;")
                conn.commit()
            finally:
                conn.close()

    # --------------------------------------------------------------------------
    # 12. PRODUCTION DATABASE ISOLATION
    # --------------------------------------------------------------------------
    def test_12_production_database_isolation(self):
        """Verify data/opspilot.db was never touched and remains isolated across all tables."""
        prod_db = "data/opspilot.db"
        self.assertNotEqual(
            os.path.abspath(str(self.temp_path)),
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


if __name__ == "__main__":
    unittest.main()

