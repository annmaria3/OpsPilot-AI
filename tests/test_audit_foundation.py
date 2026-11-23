"""OpsPilot AI - Human Decision Audit Foundation Tests (Step 6.4A.2).

Validates:
1. Successful approval updates classification and creates exactly one CLASSIFICATION_APPROVED audit record.
2. Missing incident returns False or raises service-layer validation error.
3. Simulated audit insertion failure rolls back the classification update (transaction atomicity).
4. Successful rejection creates exactly one CLASSIFICATION_REJECTED audit record.
5. Rejection leaves all incident classification fields unchanged.
6. Approval never records AI_CLASSIFIED.
7. Invalid taxonomy values are rejected without database changes.
8. Production database isolation check (0 records modified or created in data/opspilot.db).

Uses isolated temporary SQLite databases for all tests.
"""

import os
import sqlite3
import tempfile
import unittest

from app import config
from app.database import db
from app.services import incident_service


class TestAuditFoundation(unittest.TestCase):
    """Test suite for human approval, rejection, and audit transaction atomicity."""

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
        """Create a temporary isolated SQLite database for each test."""
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_file.close()
        self.test_db_path = self.temp_file.name

        # Initialize schema, triggers, and foreign keys in isolated test DB
        db.initialize_database(db_path=self.test_db_path)

        # Seed one sample incident in the test database
        self.incident_id = db.create_incident(
            title="Payment gateway timeout",
            description="Customers report 504 gateway timeout on checkout page.",
            department="Payments & Billing",
            impact="High",
            db_path=self.test_db_path,
        )

    def tearDown(self):
        """Clean up temporary test database file."""
        if os.path.exists(self.test_db_path):
            try:
                os.remove(self.test_db_path)
            except OSError:
                pass

    # --------------------------------------------------------------------------
    # 1. Successful Approval Updates Classification & Records Audit
    # --------------------------------------------------------------------------
    def test_01_successful_approval_updates_classification_and_audit(self):
        success = incident_service.update_incident_classification(
            incident_id=self.incident_id,
            category="Payment / Transaction Failure",
            severity="High",
            department="Payments & Billing",
            impact="Critical",
            actor="Operator",
            notes="Operator verified symptoms match payment gateway outage.",
            db_path=self.test_db_path,
        )
        self.assertTrue(success)

        # Verify incident record was updated
        incident = incident_service.get_incident_by_id(self.incident_id, db_path=self.test_db_path)
        self.assertEqual(incident["category"], "Payment / Transaction Failure")
        self.assertEqual(incident["severity"], "High")
        self.assertEqual(incident["department"], "Payments & Billing")
        self.assertEqual(incident["impact"], "Critical")

        # Verify audit record was created with CLASSIFICATION_APPROVED
        audit_logs = incident_service.get_audit_history_for_incident(
            self.incident_id, db_path=self.test_db_path
        )
        approval_logs = [log for log in audit_logs if log["action"] == "CLASSIFICATION_APPROVED"]
        self.assertEqual(len(approval_logs), 1)
        app_log = approval_logs[0]
        self.assertEqual(app_log["actor"], "Operator")
        self.assertIn("verified symptoms", app_log["notes"])

    # --------------------------------------------------------------------------
    # 2. Missing Incident Handling
    # --------------------------------------------------------------------------
    def test_02_missing_incident_handling(self):
        # Service layer raises ValueError for non-existent incident
        with self.assertRaises(ValueError) as ctx:
            incident_service.update_incident_classification(
                incident_id=999999,
                category="Payment / Transaction Failure",
                severity="High",
                db_path=self.test_db_path,
            )
        self.assertIn("not found", str(ctx.exception).lower())

        with self.assertRaises(ValueError) as ctx_rej:
            incident_service.reject_incident_classification(
                incident_id=999999,
                proposed_category="Payment / Transaction Failure",
                db_path=self.test_db_path,
            )
        self.assertIn("not found", str(ctx_rej.exception).lower())

        # Low-level DB function returns False if incident does not exist
        res = db.apply_incident_classification_with_audit(
            incident_id=999999,
            category="Payment / Transaction Failure",
            severity="High",
            db_path=self.test_db_path,
        )
        self.assertFalse(res)

    # --------------------------------------------------------------------------
    # 3. Transaction Atomicity: Audit Failure Rolls Back Classification
    # --------------------------------------------------------------------------
    def test_03_audit_failure_rolls_back_classification_update(self):
        """Verify that if the audit log insertion fails, the incident classification is rolled back."""
        # Check pre-state
        pre_incident = incident_service.get_incident_by_id(self.incident_id, db_path=self.test_db_path)
        self.assertIsNone(pre_incident["category"])
        self.assertIsNone(pre_incident["severity"])

        # Install a temporary trigger in the test DB that simulates an audit insertion failure
        conn = db.get_connection(self.test_db_path)
        conn.execute("""
            CREATE TRIGGER trigger_simulate_audit_failure
            BEFORE INSERT ON audit_logs
            WHEN NEW.action = 'CLASSIFICATION_APPROVED'
            BEGIN
                SELECT RAISE(ABORT, 'Simulated disk write failure during audit insertion');
            END;
        """)
        conn.commit()
        conn.close()

        # Attempt classification update - should raise sqlite3.IntegrityError from trigger
        with self.assertRaises(sqlite3.IntegrityError) as ctx:
            incident_service.update_incident_classification(
                incident_id=self.incident_id,
                category="Authentication / Access Issue",
                severity="Critical",
                db_path=self.test_db_path,
            )
        self.assertIn("Simulated disk write failure", str(ctx.exception))

        # Check post-state: classification update on incidents must be ROLLED BACK
        post_incident = incident_service.get_incident_by_id(self.incident_id, db_path=self.test_db_path)
        self.assertIsNone(post_incident["category"], "Category must be rolled back to None")
        self.assertIsNone(post_incident["severity"], "Severity must be rolled back to None")

        # Verify no orphan CLASSIFICATION_APPROVED audit record was left
        audit_logs = incident_service.get_audit_history_for_incident(
            self.incident_id, db_path=self.test_db_path
        )
        approval_logs = [log for log in audit_logs if log["action"] == "CLASSIFICATION_APPROVED"]
        self.assertEqual(len(approval_logs), 0)

        # Drop the simulation trigger
        conn = db.get_connection(self.test_db_path)
        conn.execute("DROP TRIGGER trigger_simulate_audit_failure;")
        conn.commit()
        conn.close()

    # --------------------------------------------------------------------------
    # 4. Successful Rejection Records CLASSIFICATION_REJECTED
    # --------------------------------------------------------------------------
    def test_04_successful_rejection_records_audit(self):
        success = incident_service.reject_incident_classification(
            incident_id=self.incident_id,
            proposed_category="Infrastructure / Server Downtime",
            proposed_severity="Low",
            proposed_department="General Operations",
            proposed_impact="Low",
            actor="LeadAnalyst",
            reason="Symptoms indicate payment gateway failure, not general infrastructure downtime.",
            db_path=self.test_db_path,
        )
        self.assertTrue(success)

        audit_logs = incident_service.get_audit_history_for_incident(
            self.incident_id, db_path=self.test_db_path
        )
        rejection_logs = [log for log in audit_logs if log["action"] == "CLASSIFICATION_REJECTED"]
        self.assertEqual(len(rejection_logs), 1)

        rej_log = rejection_logs[0]
        self.assertEqual(rej_log["actor"], "LeadAnalyst")
        self.assertIn("Infrastructure / Server Downtime", rej_log["notes"])
        self.assertIn("payment gateway failure", rej_log["notes"])

    # --------------------------------------------------------------------------
    # 5. Rejection Leaves Incident Fields Unchanged
    # --------------------------------------------------------------------------
    def test_05_rejection_leaves_incident_fields_unchanged(self):
        before = incident_service.get_incident_by_id(self.incident_id, db_path=self.test_db_path)

        incident_service.reject_incident_classification(
            incident_id=self.incident_id,
            proposed_category="Application Performance & Latency",
            proposed_severity="Medium",
            actor="Operator",
            reason="Incorrect category.",
            db_path=self.test_db_path,
        )

        after = incident_service.get_incident_by_id(self.incident_id, db_path=self.test_db_path)

        self.assertEqual(before["title"], after["title"])
        self.assertEqual(before["description"], after["description"])
        self.assertEqual(before["department"], after["department"])
        self.assertEqual(before["impact"], after["impact"])
        self.assertEqual(before["category"], after["category"])
        self.assertEqual(before["severity"], after["severity"])
        self.assertEqual(before["status"], after["status"])

    # --------------------------------------------------------------------------
    # 6. Approval Never Records AI_CLASSIFIED
    # --------------------------------------------------------------------------
    def test_06_approval_never_records_ai_classified(self):
        incident_service.update_incident_classification(
            incident_id=self.incident_id,
            category="Database Connectivity / Query Failure",
            severity="Medium",
            actor="Operator",
            db_path=self.test_db_path,
        )

        audit_logs = incident_service.get_audit_history_for_incident(
            self.incident_id, db_path=self.test_db_path
        )
        actions = [log["action"] for log in audit_logs]

        self.assertNotIn("AI_CLASSIFIED", actions, "Human approval must never record AI_CLASSIFIED.")
        self.assertIn("CLASSIFICATION_APPROVED", actions)

    # --------------------------------------------------------------------------
    # 7. Invalid Taxonomy Values Rejected Without DB Changes
    # --------------------------------------------------------------------------
    def test_07_invalid_taxonomy_rejected_without_db_changes(self):
        # Invalid Category
        with self.assertRaises(ValueError) as ctx_cat:
            incident_service.update_incident_classification(
                incident_id=self.incident_id,
                category="Nonexistent Fake Category",
                db_path=self.test_db_path,
            )
        self.assertIn("invalid category", str(ctx_cat.exception).lower())

        # Invalid Severity
        with self.assertRaises(ValueError) as ctx_sev:
            incident_service.update_incident_classification(
                incident_id=self.incident_id,
                severity="SuperCritical",
                db_path=self.test_db_path,
            )
        self.assertIn("invalid severity", str(ctx_sev.exception).lower())

        # Invalid Department
        with self.assertRaises(ValueError) as ctx_dept:
            incident_service.update_incident_classification(
                incident_id=self.incident_id,
                department="Marketing Department",
                db_path=self.test_db_path,
            )
        self.assertIn("invalid department", str(ctx_dept.exception).lower())

        # Invalid Impact
        with self.assertRaises(ValueError) as ctx_imp:
            incident_service.update_incident_classification(
                incident_id=self.incident_id,
                impact="Disastrous",
                db_path=self.test_db_path,
            )
        self.assertIn("invalid impact", str(ctx_imp.exception).lower())

        # Empty update
        with self.assertRaises(ValueError):
            incident_service.update_incident_classification(
                incident_id=self.incident_id,
                db_path=self.test_db_path,
            )

        # Check database: no audit logs added
        audit_logs = incident_service.get_audit_history_for_incident(
            self.incident_id, db_path=self.test_db_path
        )
        self.assertEqual(len(audit_logs), 0)

    # --------------------------------------------------------------------------
    # 8. Production Database Isolation Check
    # --------------------------------------------------------------------------
    def test_08_production_database_isolation(self):
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
