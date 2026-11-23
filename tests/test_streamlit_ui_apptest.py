"""OpsPilot AI - Streamlit AppTest Automated UI Integration Test Suite (Step 7.4D).

Exercises real Streamlit execution, widget event loop, session-state management,
and page navigation via `streamlit.testing.v1.AppTest`.

Covers:
1. Application startup and page rendering across all pages.
2. Navigation between all 6 pages.
3. Incident selection and empty-state rendering.
4. Investigation execution with mocked AI and retrieval results.
5. Operator approval interaction (Approve As-Is).
6. Operator edit-and-approve interaction (Save & Approve Customized Action).
7. Operator rejection interaction (Confirm Rejection, final_action = NULL).
8. Evidence-only fallback state rendering.
9. Cross-navigation from Incident Center to AI Investigation Workspace.
10. Production database isolation check.

All tests operate on isolated temporary SQLite databases.
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
from app.services import incident_service


class TestStreamlitUIAppTest(unittest.TestCase):
    """Automated UI test suite using Streamlit AppTest."""

    @classmethod
    def setUpClass(cls):
        """Locate the main Streamlit application entrypoint."""
        cls.script_path = str(Path(__file__).resolve().parent.parent / "app" / "main.py")
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

        # Standard conforming mock investigation data
        self.mock_investigation = {
            "reported_symptoms": ["Users report 504 Gateway Timeout on checkout."],
            "confirmed_facts": [],
            "assumptions": ["Assuming database connection pool saturation."],
            "root_cause_hypotheses": [
                {
                    "hypothesis": "PostgreSQL connection pool exhausted by unclosed connections",
                    "likelihood": "High",
                    "supporting_evidence_citations": ["[Runbook_Postgres.md, p.1 #2]"],
                    "unsupported_claims": ["Connection leak rate unmeasured"],
                }
            ],
            "evidence_citations": ["[Runbook_Postgres.md, p.1 #2]"],
            "ai_recommendation": "1. Temporarily scale max_connections to 250.\n2. Terminate idle worker connections.",
            "confidence": 0.88,
            "missing_information": ["SELECT * FROM pg_stat_activity;"],
            "diagnostic_actions": ["SELECT count(*) FROM pg_stat_activity WHERE state = 'idle in transaction';"],
            "requires_human_review": True,
            "citation_validation_summary": {
                "total_citations": 1,
                "valid_citations": 1,
                "flagged_citations": 0,
                "flagged_details": [],
            },
            "retrieved_evidence": [
                {
                    "chunk_id": 201,
                    "document_id": 10,
                    "document_name": "Runbook_Postgres.md",
                    "page_number": 1,
                    "chunk_index": 2,
                    "rank": 1,
                    "citation": "[Runbook_Postgres.md, p.1 #2]",
                    "citation_full": "Runbook: Runbook_Postgres.md (Page 1, Chunk 2)",
                    "bm25_score": -2.45,
                    "relevance_score": 2.45,
                    "content": "Scale max_connections to 250 when active connections exceed 90%.",
                }
            ],
        }

    def tearDown(self):
        """Clean up the isolated database file."""
        if os.path.exists(self.test_db_path):
            try:
                os.remove(self.test_db_path)
            except OSError:
                pass

    # --------------------------------------------------------------------------
    # 1. APPLICATION STARTUP & PAGE RENDERING
    # --------------------------------------------------------------------------
    def test_01_app_startup_and_initial_render(self):
        """Verify the application boots and renders the Incident Journey dashboard without unhandled exceptions."""
        with patch.object(app.config, "DATABASE_PATH", self.test_db_path), \
             patch.object(db, "DATABASE_PATH", self.test_db_path):
            at = AppTest.from_file(self.script_path)
            at.run(timeout=30)

            self.assertEqual(len(at.exception), 0, f"Startup produced exceptions: {at.exception}")
            self.assertGreater(len(at.markdown), 5)
            self.assertEqual(len(at.sidebar.radio), 1)
            self.assertEqual(at.sidebar.radio[0].value, "Incident Journey")

    # --------------------------------------------------------------------------
    # 2. NAVIGATION ACROSS ALL 7 PAGES
    # --------------------------------------------------------------------------
    def test_02_navigation_between_pages(self):
        """Verify navigation across all 7 workspace pages completes without exceptions."""
        with patch.object(app.config, "DATABASE_PATH", self.test_db_path), \
             patch.object(db, "DATABASE_PATH", self.test_db_path):
            at = AppTest.from_file(self.script_path)
            at.run(timeout=30)

            pages = [
                "Incident Journey",
                "Incident Center",
                "AI Investigation",
                "Knowledge Base",
                "Simulation Lab",
                "Evaluation",
                "Overview",
            ]
            for page_name in pages:
                at.sidebar.radio[0].set_value(page_name).run(timeout=30)
                self.assertEqual(len(at.exception), 0, f"Navigation to '{page_name}' failed with exceptions: {at.exception}")
                self.assertEqual(at.sidebar.radio[0].value, page_name)

    # --------------------------------------------------------------------------
    # 3. INCIDENT SELECTION & EMPTY STATE
    # --------------------------------------------------------------------------
    def test_03_incident_selection_and_empty_state(self):
        """Verify empty-state rendering when no incidents exist, and correct selector when seeded."""
        with patch.object(app.config, "DATABASE_PATH", self.test_db_path), \
             patch.object(db, "DATABASE_PATH", self.test_db_path):
            # A. Empty state
            at_empty = AppTest.from_file(self.script_path)
            at_empty.run(timeout=30)
            at_empty.sidebar.radio[0].set_value("AI Investigation").run(timeout=30)
            self.assertEqual(len(at_empty.exception), 0)
            # Info box rendered for no incidents
            info_texts = [i.value for i in at_empty.info]
            self.assertTrue(any("No incidents have been logged yet" in t for t in info_texts))

            # B. Seeded incident state
            inc_id = db.create_incident(
                title="API Gateway High Latency",
                description="Latency spike to 3500ms on /checkout endpoint.",
                category="API / Gateway Latency",
                severity="High",
                department="Engineering & DevOps",
                impact="High",
                db_path=self.test_db_path,
            )
            at_seeded = AppTest.from_file(self.script_path)
            at_seeded.run(timeout=30)
            at_seeded.sidebar.radio[0].set_value("AI Investigation").run(timeout=30)
            self.assertEqual(len(at_seeded.exception), 0)

            # Selector contains seeded incident
            sb = [s for s in at_seeded.selectbox if s.key == "investigation_incident_selector"][0]
            self.assertTrue(any(f"INC-{inc_id:04d}" in opt for opt in sb.options))

    # --------------------------------------------------------------------------
    # 4. INVESTIGATION EXECUTION WITH MOCKED AI
    # --------------------------------------------------------------------------
    def test_04_investigation_execution_with_mocked_ai(self):
        """Verify clicking 'Run AI Investigation' creates proposal and renders findings and evidence."""
        inc_id = db.create_incident(
            title="Database Connection Pool Saturation",
            description="504 errors on checkout.",
            db_path=self.test_db_path,
        )

        with patch.object(app.config, "DATABASE_PATH", self.test_db_path), \
             patch.object(db, "DATABASE_PATH", self.test_db_path), \
             patch("app.services.ai_service.investigate_incident", return_value=self.mock_investigation):
            at = AppTest.from_file(self.script_path)
            at.run(timeout=30)
            at.sidebar.radio[0].set_value("AI Investigation").run(timeout=30)

            run_btn = [b for b in at.button if b.key == f"btn_run_inv_{inc_id}"][0]
            run_btn.click().run(timeout=30)
            self.assertEqual(len(at.exception), 0)

            # Check investigation row persisted in database
            invs = db.get_investigations_for_incident(inc_id, db_path=self.test_db_path)
            self.assertEqual(len(invs), 1)
            self.assertEqual(invs[0]["human_decision"], "Pending")

            # Governance console rendered with Approve As-Is button
            app_buttons = [b for b in at.button if b.key.startswith("btn_app_asis_")]
            self.assertEqual(len(app_buttons), 1)

    # --------------------------------------------------------------------------
    # 5. OPERATOR APPROVAL INTERACTION
    # --------------------------------------------------------------------------
    def test_05_operator_approval_interaction(self):
        """Verify operator clicking 'Approve As-Is' commits approval, logs audit, and updates UI."""
        inc_id = db.create_incident(
            title="Database Failure",
            description="504 errors",
            db_path=self.test_db_path,
        )

        with patch.object(app.config, "DATABASE_PATH", self.test_db_path), \
             patch.object(db, "DATABASE_PATH", self.test_db_path), \
             patch("app.services.ai_service.investigate_incident", return_value=self.mock_investigation):
            at = AppTest.from_file(self.script_path)
            at.run(timeout=30)
            at.sidebar.radio[0].set_value("AI Investigation").run(timeout=30)

            # 1. Run investigation
            run_btn = [b for b in at.button if b.key == f"btn_run_inv_{inc_id}"][0]
            run_btn.click().run(timeout=30)

            # 2. Click Approve As-Is
            app_btn = [b for b in at.button if b.key.startswith("btn_app_asis_")][0]
            app_btn.click().run(timeout=30)
            self.assertEqual(len(at.exception), 0)

            # 3. Verify SQLite persistence
            inv_row = db.get_investigations_for_incident(inc_id, db_path=self.test_db_path)[0]
            self.assertEqual(inv_row["human_decision"], "Approved")
            self.assertEqual(inv_row["final_action"], self.mock_investigation["ai_recommendation"])

            inc_row = db.get_incident_by_id(inc_id, db_path=self.test_db_path)
            self.assertEqual(inc_row["status"], "Approved")

            # 4. Verify audit event
            logs = incident_service.get_audit_history_for_incident(inc_id, db_path=self.test_db_path)
            self.assertTrue(any(l["action"] == "RECOMMENDATION_APPROVED" for l in logs))

    # --------------------------------------------------------------------------
    # 6. OPERATOR EDIT-AND-APPROVE INTERACTION
    # --------------------------------------------------------------------------
    def test_06_operator_edit_and_approve_interaction(self):
        """Verify operator can customize recommendation and commit custom final action via AppTest."""
        inc_id = db.create_incident(
            title="Cache cluster eviction",
            description="High latency on cache keys",
            db_path=self.test_db_path,
        )

        with patch.object(app.config, "DATABASE_PATH", self.test_db_path), \
             patch.object(db, "DATABASE_PATH", self.test_db_path), \
             patch("app.services.ai_service.investigate_incident", return_value=self.mock_investigation):
            at = AppTest.from_file(self.script_path)
            at.run(timeout=30)
            at.sidebar.radio[0].set_value("AI Investigation").run(timeout=30)

            # Run investigation
            run_btn = [b for b in at.button if b.key == f"btn_run_inv_{inc_id}"][0]
            run_btn.click().run(timeout=30)

            # Check edit checkbox
            chk_edit = [c for c in at.checkbox if c.key.startswith("chk_edit_")][0]
            chk_edit.check().run(timeout=30)

            # Input custom text
            custom_text = "Custom action plan verified by Lead SRE."
            txt_area = [t for t in at.text_area if t.key.startswith("custom_action_input_")][0]
            txt_area.input(custom_text).run(timeout=30)

            # Click Save & Approve
            save_btn = [b for b in at.button if b.key.startswith("btn_save_custom_")][0]
            save_btn.click().run(timeout=30)
            self.assertEqual(len(at.exception), 0)

            # Verify SQLite record
            inv_row = db.get_investigations_for_incident(inc_id, db_path=self.test_db_path)[0]
            self.assertEqual(inv_row["human_decision"], "Approved")
            self.assertEqual(inv_row["final_action"], custom_text)
            self.assertEqual(inv_row["ai_recommendation"], self.mock_investigation["ai_recommendation"])

            # Verify audit event
            logs = incident_service.get_audit_history_for_incident(inc_id, db_path=self.test_db_path)
            self.assertTrue(any(l["action"] == "RECOMMENDATION_EDITED" for l in logs))

    # --------------------------------------------------------------------------
    # 7. OPERATOR REJECTION INTERACTION
    # --------------------------------------------------------------------------
    def test_07_operator_rejection_interaction(self):
        """Verify operator can reject recommendation, setting final_action to NULL and status to Open."""
        inc_id = db.create_incident(
            title="Network partition",
            description="Packet drop between AZs",
            db_path=self.test_db_path,
        )

        with patch.object(app.config, "DATABASE_PATH", self.test_db_path), \
             patch.object(db, "DATABASE_PATH", self.test_db_path), \
             patch("app.services.ai_service.investigate_incident", return_value=self.mock_investigation):
            at = AppTest.from_file(self.script_path)
            at.run(timeout=30)
            at.sidebar.radio[0].set_value("AI Investigation").run(timeout=30)

            # Run investigation
            run_btn = [b for b in at.button if b.key == f"btn_run_inv_{inc_id}"][0]
            run_btn.click().run(timeout=30)

            # Set notes
            notes_input = [t for t in at.text_input if t.key.startswith("inv_notes_input_")][0]
            notes_input.input("Runbook does not apply to this AZ topology.").run(timeout=30)

            # Check reject checkbox
            chk_rej = [c for c in at.checkbox if c.key.startswith("chk_rej_")][0]
            chk_rej.check().run(timeout=30)

            # Confirm rejection
            rej_btn = [b for b in at.button if b.key.startswith("btn_confirm_rej_")][0]
            rej_btn.click().run(timeout=30)
            self.assertEqual(len(at.exception), 0)

            # Verify SQLite record
            inv_row = db.get_investigations_for_incident(inc_id, db_path=self.test_db_path)[0]
            self.assertEqual(inv_row["human_decision"], "Rejected")
            self.assertIsNone(inv_row["final_action"])

            inc_row = db.get_incident_by_id(inc_id, db_path=self.test_db_path)
            self.assertEqual(inc_row["status"], "Open")

            # Verify audit event
            logs = incident_service.get_audit_history_for_incident(inc_id, db_path=self.test_db_path)
            self.assertTrue(any(l["action"] == "RECOMMENDATION_REJECTED" for l in logs))

    # --------------------------------------------------------------------------
    # 8. EVIDENCE-ONLY FALLBACK STATE RENDERING & GOVERNANCE SAFETY
    # --------------------------------------------------------------------------
    def test_08_evidence_only_fallback_rendering(self):
        """Verify fallback investigation renders notice banner, N/A confidence, disables Approve As-Is, and requires operator remediation."""
        inc_id = db.create_incident(
            title="Kubernetes Ingress Pod Crash",
            description="CrashLoopBackOff on ingress controller",
            db_path=self.test_db_path,
        )

        fallback_payload = dict(self.mock_investigation)
        fallback_payload["mode"] = "evidence_only_fallback"
        fallback_payload["is_fallback"] = True
        fallback_payload["confidence"] = None
        fallback_payload["error_type"] = "missing_configuration"
        fallback_payload["fallback_reason"] = "Gemini API key is not configured in .env."
        fallback_payload["ai_recommendation"] = "Automated AI diagnostic synthesis is currently unavailable (missing_configuration: Gemini API key is not configured in .env).\n\nProcedural runbook excerpts have been retrieved below for manual review."

        with patch.object(app.config, "DATABASE_PATH", self.test_db_path), \
             patch.object(db, "DATABASE_PATH", self.test_db_path), \
             patch("app.services.ai_service.investigate_incident", return_value=fallback_payload):
            at = AppTest.from_file(self.script_path)
            at.run(timeout=30)
            at.sidebar.radio[0].set_value("AI Investigation").run(timeout=30)

            run_btn = [b for b in at.button if b.key == f"btn_run_inv_{inc_id}"][0]
            run_btn.click().run(timeout=30)
            self.assertEqual(len(at.exception), 0)

            # 1. Fallback recommendation and NULL confidence persisted in SQLite
            invs = db.get_investigations_for_incident(inc_id, db_path=self.test_db_path)
            self.assertEqual(len(invs), 1)
            self.assertIsNone(invs[0]["confidence"])
            self.assertIn("Automated AI diagnostic synthesis is currently unavailable", invs[0]["ai_recommendation"])

            # 2. Check UI elements: Approve As-Is must be disabled
            app_buttons = [b for b in at.button if b.key.startswith("btn_app_asis_")]
            self.assertEqual(len(app_buttons), 1)
            self.assertTrue(app_buttons[0].disabled)

            # 3. Check operator authoring textarea exists
            txt_areas = [t for t in at.text_area if t.key.startswith("custom_action_input_")]
            self.assertEqual(len(txt_areas), 1)

            # 4. Save & Approve Operator Remediation
            manual_procedure = "Restart ingress pods: kubectl rollout restart deployment/nginx-ingress -n ingress"
            txt_areas[0].input(manual_procedure).run(timeout=30)

            save_btn = [b for b in at.button if b.key.startswith("btn_save_custom_")][0]
            save_btn.click().run(timeout=30)
            self.assertEqual(len(at.exception), 0)

            # 5. Verify approved state and audit event
            updated_inv = db.get_investigations_for_incident(inc_id, db_path=self.test_db_path)[0]
            self.assertEqual(updated_inv["human_decision"], "Approved")
            self.assertEqual(updated_inv["final_action"], manual_procedure)

            logs = incident_service.get_audit_history_for_incident(inc_id, db_path=self.test_db_path)
            self.assertTrue(any(l["action"] == "RECOMMENDATION_APPROVED" and "operator-authored" in l["notes"].lower() for l in logs))

            # 6. Verify in-container success banner and approved label in rendered markdown
            self.assertTrue(any("Remediation decision recorded successfully" in s.value for s in at.success))
            self.assertTrue(any("Operator-authored remediation — not AI-generated" in m.value for m in at.markdown))

    # --------------------------------------------------------------------------
    # 9. CROSS-NAVIGATION FROM INCIDENT CENTER
    # --------------------------------------------------------------------------
    def test_09_cross_navigation_from_incident_center(self):
        """Verify deep-linking button in Incident Center navigates to AI Investigation Workspace."""
        inc_id = db.create_incident(
            title="Search Index Sync Failure",
            description="Elasticsearch index lagging by 45 minutes",
            db_path=self.test_db_path,
        )

        with patch.object(app.config, "DATABASE_PATH", self.test_db_path), \
             patch.object(db, "DATABASE_PATH", self.test_db_path):
            at = AppTest.from_file(self.script_path)
            at.run(timeout=30)
            at.sidebar.radio[0].set_value("Incident Center").run(timeout=30)

            # Click launch investigation deep-link
            nav_btn = [b for b in at.button if b.key == f"btn_start_ai_inv_{inc_id}"][0]
            nav_btn.click().run(timeout=30)
            self.assertEqual(len(at.exception), 0)

            # Verify navigation transitioned to AI Investigation
            self.assertEqual(at.sidebar.radio[0].value, "AI Investigation")
            # Verify incident selector is focused on this incident
            sb = [s for s in at.selectbox if s.key == "investigation_incident_selector"][0]
            self.assertIn(f"INC-{inc_id:04d}", sb.value)

    # --------------------------------------------------------------------------
    # 10. PRODUCTION DATABASE ISOLATION
    # --------------------------------------------------------------------------
    def test_10_production_database_isolation(self):
        """Verify production database (data/opspilot.db) was not modified by any test."""
        prod_db = "data/opspilot.db"
        self.assertNotEqual(
            os.path.abspath(self.test_db_path),
            os.path.abspath(prod_db),
            "Test database path must not be the production database path!",
        )
        if os.path.exists(prod_db):
            conn = sqlite3.connect(prod_db)
            cursor = conn.cursor()
            for table, initial_count in getattr(self, "_prod_db_counts", {}).items():
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                current_count = cursor.fetchone()[0]
                self.assertEqual(
                    current_count,
                    initial_count,
                    f"Table '{table}' in data/opspilot.db was modified during test run! (before: {initial_count}, after: {current_count})",
                )
            conn.close()

    # --------------------------------------------------------------------------
    # 11. DEEP-LINK NAVIGATION SYNCHRONIZATION & ISOLATION
    # --------------------------------------------------------------------------
    def test_11_deep_link_navigation_synchronization_and_isolation(self):
        """Verify deep-link navigation synchronizes radio widget state and does not leak across subsequent navigation."""
        inc_1 = db.create_incident(
            title="Payment gateway timeout",
            description="Stripe 504 errors",
            db_path=self.test_db_path,
        )
        inc_2 = db.create_incident(
            title="Redis cache memory saturation",
            description="OOM on redis cluster",
            db_path=self.test_db_path,
        )

        with patch.object(app.config, "DATABASE_PATH", self.test_db_path), \
             patch.object(db, "DATABASE_PATH", self.test_db_path):
            at = AppTest.from_file(self.script_path)
            at.run(timeout=30)

            # 1. Start application on Incident Center
            at.sidebar.radio[0].set_value("Incident Center").run(timeout=30)
            self.assertEqual(at.sidebar.radio[0].value, "Incident Center")

            # Select inc_1 in inspector dropdown
            insp_sb = [s for s in at.selectbox if s.options and any(f"INC-{inc_1:04d}" in o for o in s.options)][0]
            opt_inc1 = [o for o in insp_sb.options if f"INC-{inc_1:04d}" in o][0]
            insp_sb.select(opt_inc1).run(timeout=30)

            # 2. Stage deep-link target by clicking Launch AI Investigation button
            deep_btn = [b for b in at.button if b.key == f"btn_start_ai_inv_{inc_1}"][0]

            # 3. Execute Streamlit rerun via click
            deep_btn.click().run(timeout=30)
            self.assertEqual(len(at.exception), 0)

            # 4. Verify the actual main_nav_radio value becomes AI Investigation
            self.assertEqual(at.sidebar.radio[0].value, "AI Investigation")
            self.assertEqual(at.session_state.get("main_nav_radio"), "AI Investigation")

            # 5. Verify the AI Investigation page renders
            rendered_markdown = " ".join(m.value for m in at.markdown)
            self.assertIn("AI Investigation Workspace", rendered_markdown)

            # 6. Verify the intended incident ID is retained
            sel = [s for s in at.selectbox if s.key == "investigation_incident_selector"][0]
            self.assertIn(f"INC-{inc_1:04d}", sel.value)

            # 7. Verify the incident selection does not leak across subsequent navigation
            # A. Staged keys have been consumed
            self.assertIsNone(at.session_state.get("nav_page"))
            self.assertIsNone(at.session_state.get("selected_investigation_incident_id"))

            # B. Navigate to Overview
            at.sidebar.radio[0].set_value("Overview").run(timeout=30)
            self.assertEqual(at.sidebar.radio[0].value, "Overview")

            # C. Return to AI Investigation directly from sidebar without a deep-link
            at.sidebar.radio[0].set_value("AI Investigation").run(timeout=30)
            self.assertEqual(at.sidebar.radio[0].value, "AI Investigation")
            # Session state has not received any phantom deep-link key
            self.assertIsNone(at.session_state.get("nav_page"))
            self.assertIsNone(at.session_state.get("selected_investigation_incident_id"))

    # --------------------------------------------------------------------------
    # 12. INVESTIGATION HISTORY CARD RENDERING (DEDENT FIX ACROSS BOTH LOCATIONS)
    # --------------------------------------------------------------------------
    def test_12_investigation_history_card_rendering_dedent(self):
        """Verify investigation history card HTML is not indented with 4+ spaces across both Incident Center and AI Investigation tabs."""
        inc_id = db.create_incident(
            title="Database Latency Spike",
            description="DB connection timeout",
            db_path=self.test_db_path,
        )
        # Create iteration 1 (prior proposal)
        inv_1 = incident_service.save_investigation_proposal(
            incident_id=inc_id,
            investigation_data=self.mock_investigation,
            actor="AI Engine 1",
            db_path=self.test_db_path,
        )
        # Create iteration 2 (active proposal, making iteration 1 appear in tab_hist)
        inv_2 = incident_service.save_investigation_proposal(
            incident_id=inc_id,
            investigation_data=self.mock_investigation,
            actor="AI Engine 2",
            db_path=self.test_db_path,
        )

        with patch.object(app.config, "DATABASE_PATH", self.test_db_path), \
             patch.object(db, "DATABASE_PATH", self.test_db_path):
            at = AppTest.from_file(self.script_path, default_timeout=30)
            at.run(timeout=30)

            # Location A: Incident Center -> tab_inv (Investigation History)
            at.sidebar.radio[0].set_value("Incident Center").run(timeout=30)
            self.assertEqual(len(at.exception), 0, f"Incident Center threw exceptions: {at.exception}")

            inv_card_found = False
            for m in at.markdown:
                val = m.value
                if f"Investigation #{inv_2}" in val and "AI Recommendation:" in val:
                    inv_card_found = True
                    # Root tag must not have leading indentation spaces (which triggers CommonMark <pre><code> block)
                    for line in val.split("\n"):
                        if line.strip().startswith('<div class="metric-card"'):
                            self.assertEqual(line, line.lstrip(), "tab_inv metric card root tag must not have leading indentation spaces!")
                    self.assertIn("AI Recommendation:", val)
                    self.assertIn("PENDING APPROVAL", val)
            self.assertTrue(inv_card_found, f"Investigation #{inv_2} history card was not found in Incident Center tab_inv!")

            # Location B: AI Investigation -> tab_hist (Previous Investigation Proposals)
            at.sidebar.radio[0].set_value("AI Investigation").run(timeout=30)
            self.assertEqual(len(at.exception), 0, f"AI Investigation threw exceptions: {at.exception}")

            hist_card_found = False
            for m in at.markdown:
                val = m.value
                if f"Investigation #{inv_1}" in val and "Decision:" in val:
                    hist_card_found = True
                    # Root tag must not have leading indentation spaces
                    for line in val.split("\n"):
                        if line.strip().startswith('<div class="metric-card"'):
                            self.assertEqual(line, line.lstrip(), "tab_hist metric card root tag must not have leading indentation spaces!")
                    self.assertIn(f"Investigation #{inv_1}", val)
                    self.assertIn("Decision:", val)
            self.assertTrue(hist_card_found, f"Prior Investigation #{inv_1} card was not found in AI Investigation tab_hist!")

    # --------------------------------------------------------------------------
    # 13. INCIDENT JOURNEY: EMPTY STATE AND SHELL RENDERING (STEP 8.1)
    # --------------------------------------------------------------------------
    def test_13_incident_journey_empty_state_and_navigation(self):
        """Verify Incident Journey renders empty state when database has 0 incidents."""
        with patch.object(app.config, "DATABASE_PATH", self.test_db_path), \
             patch.object(db, "DATABASE_PATH", self.test_db_path):
            at = AppTest.from_file(self.script_path)
            at.run(timeout=30)

            # Boots directly into Incident Journey as primary landing page
            self.assertEqual(at.sidebar.radio[0].value, "Incident Journey")
            self.assertEqual(len(at.exception), 0, f"Incident Journey startup produced exceptions: {at.exception}")

            # Verify empty state notice
            info_texts = [i.value for i in at.info]
            self.assertTrue(any("No incidents have been logged yet" in t for t in info_texts),
                            "Incident Journey must show informative empty state when no incidents exist.")

    # --------------------------------------------------------------------------
    # 14. INCIDENT JOURNEY: STAGE STATE DERIVATION & GOVERNANCE CHECKS (STEP 8.1)
    # --------------------------------------------------------------------------
    def test_14_incident_journey_stage_state_derivation_and_governance(self):
        """Verify stage statuses derive strictly from persisted records and enforce governance rules.

        Mandatory Corrections Covered:
        1. Never mark all seven journey stages completed merely because an investigation was approved.
        2. Derive stage statuses from actual persisted records and available evidence.
        3. Distinguish recommendation approval from actual remediation execution and verified incident resolution.
        4. Do not invent retrieval results, evidence, classification outcomes or AI analysis.
        5. Display missing Knowledge Base documents and unavailable Gemini analysis honestly.
        """
        # A. Seed incident in 'Approved' status (recommendation approved, resolution pending)
        inc_id = db.create_incident(
            title="Database Connection Saturation",
            description="504 errors on checkout service.",
            category="API / Gateway Latency",
            severity="High",
            department="Engineering & DevOps",
            impact="High",
            status="Approved",
            db_path=self.test_db_path,
        )

        # Seed investigation in fallback mode (Gemini offline / timeout)
        fallback_inv = {
            "is_fallback": True,
            "error_type": "timeout",
            "fallback_reason": "Gemini API timeout after 45s",
            "reported_symptoms": ["504 errors on checkout service."],
            "confirmed_facts": [],
            "assumptions": [],
            "root_cause_hypotheses": [],
            "evidence_citations": [],
            "ai_recommendation": "Review runbooks manually.",
            "confidence": None,
            "missing_information": ["Telemetry offline"],
            "diagnostic_actions": ["Inspect active connections"],
            "requires_human_review": True,
            "citation_validation_summary": {"total_citations": 0, "valid_citations": 0, "flagged_citations": 0},
            "retrieved_evidence": [],
        }
        inv_id = incident_service.save_investigation_proposal(
            incident_id=inc_id,
            investigation_data=fallback_inv,
            actor="AI Fallback",
            db_path=self.test_db_path,
        )
        # Apply human decision = Approved with operator-authored action
        incident_service.apply_investigation_decision(
            investigation_id=inv_id,
            decision="Approved",
            final_action="Scale max_connections to 250 and restart pooler.",
            reviewer_notes="Operator approved manual procedure during offline fallback.",
            actor="Operator",
            db_path=self.test_db_path,
        )

        with patch.object(app.config, "DATABASE_PATH", self.test_db_path), \
             patch.object(db, "DATABASE_PATH", self.test_db_path):
            at = AppTest.from_file(self.script_path)
            at.run(timeout=30)

            self.assertEqual(at.sidebar.radio[0].value, "Incident Journey")
            self.assertEqual(len(at.exception), 0, f"Incident Journey threw exceptions: {at.exception}")

            markdown_texts = [m.value for m in at.markdown]
            full_markup = "\n".join(markdown_texts)

            # 1. Verify 7-stage visual navigation rendered via tabs and stage inspection cards
            self.assertIn("journey-stage-card", full_markup)
            self.assertIn("Incident Received", full_markup)
            self.assertIn("Impact &amp; Classification", full_markup)
            self.assertIn("Knowledge Retrieval", full_markup)
            self.assertIn("AI Root Cause Investigation", full_markup)
            self.assertIn("Remediation Plan", full_markup)
            self.assertIn("Human Review &amp; Decision", full_markup)
            self.assertIn("Outcome &amp; Audit Trail", full_markup)

            # Verify tabs contain the 7 stage short titles
            tab_labels = [tab.label for tab in at.tabs]
            self.assertEqual(len(tab_labels), 7)
            self.assertIn("1. Received", tab_labels[0])
            self.assertIn("2. Impact", tab_labels[1])
            self.assertIn("3. Retrieval", tab_labels[2])
            self.assertIn("4. AI Analysis", tab_labels[3])
            self.assertIn("5. Remediation", tab_labels[4])
            self.assertIn("6. Governance", tab_labels[5])
            self.assertIn("7. Outcome", tab_labels[6])

            # 2. Stage 1: Completed
            self.assertIn("Tracking ID: INC-0001", full_markup)
            self.assertIn("Completed", full_markup)

            # 3. Stage 2: Classified
            self.assertIn("Severity: High", full_markup)
            self.assertIn("Classified", full_markup)

            # 4. Stage 3: Honestly reports 0 runbooks indexed (no invented docs)
            self.assertIn("0 Docs Indexed", full_markup)
            self.assertIn("0 documents are indexed", full_markup)

            # 5. Stage 4: Honestly reports Model Offline fallback (never disguised as successful AI)
            self.assertIn("Model Offline", full_markup)
            self.assertIn("Operator Manual Mode", full_markup)

            # 6. Stage 6: Human review decision approved
            self.assertIn("Approved by Operator", full_markup)

            # 7. Stage 7: MANDATORY CORRECTION - NOT marked completed merely because investigation was approved!
            # Shows 'Awaiting Execution' because incident status is 'Approved', not 'Resolved'
            self.assertIn("Awaiting Execution", full_markup)
            self.assertIn("Remediation approved — Resolution pending", full_markup)

            # B. Now mark incident as officially Resolved and verify Stage 7 completes
            db.update_incident_status(inc_id, "Resolved", db_path=self.test_db_path)

            at_resolved = AppTest.from_file(self.script_path)
            at_resolved.run(timeout=30)
            self.assertEqual(len(at_resolved.exception), 0)

            resolved_markup = "\n".join([m.value for m in at_resolved.markdown])
            self.assertIn("Incident officially Resolved", resolved_markup)
            self.assertIn("Remediation verified", resolved_markup)

    # --------------------------------------------------------------------------
    # 15. INCIDENT JOURNEY: HTML RENDERING NOT ESCAPED AS CODE BLOCKS
    # --------------------------------------------------------------------------
    def test_15_incident_journey_html_rendering_not_escaped_as_code_blocks(self):
        """Verify HTML blocks in Incident Journey (stepper, narrative box, evidence) render as genuine DOM elements without <pre><code> escaping."""
        import markdown_it
        md = markdown_it.MarkdownIt()

        inc_id = db.create_incident(
            title="Payment Processing Latency",
            description="504 gateway timeout on payments endpoint",
            db_path=self.test_db_path,
        )

        with patch.object(app.config, "DATABASE_PATH", self.test_db_path), \
             patch.object(db, "DATABASE_PATH", self.test_db_path):
            at = AppTest.from_file(self.script_path)
            at.run(timeout=30)

            self.assertEqual(at.sidebar.radio[0].value, "Incident Journey")
            self.assertEqual(len(at.exception), 0)

            # Inspect all rendered markdown outputs on Incident Journey
            journey_stage_card_found = False

            for m in at.markdown:
                val = m.value
                # Skip the global <style> stylesheet block
                if val.strip().startswith("<style"):
                    continue

                # Defect 1: No literal &#x20; entities in markdown
                self.assertNotIn("&#x20;", val, f"Literal &#x20; found in Incident Journey: {val}")

                # Defect 2: No escaped dots like 2\. in markdown
                self.assertNotIn(r"2\.", val, f"Escaped 2\\. found in Incident Journey: {val}")

                # Defect 4: Duplicated static stepper must NOT be present
                self.assertNotIn("journey-stepper-container", val, f"Duplicated static stepper found in Incident Journey!")

                # 1. No line in HTML blocks should have 4+ leading spaces (CommonMark code block trigger)
                for line in val.splitlines():
                    stripped = line.strip()
                    if stripped.startswith("<div") or stripped.startswith("<span") or stripped.startswith("<strong"):
                        self.assertEqual(line, line.lstrip(), f"HTML line must have 0 leading spaces: {line}")

                # 2. Render through CommonMark parser to verify browser DOM representation
                rendered_dom = md.render(val)

                if val.strip().startswith('<div class="journey-stage-card">'):
                    journey_stage_card_found = True
                    self.assertNotIn("<pre><code>", rendered_dom, f"Stage card was converted to code block:\n{rendered_dom}")
                    self.assertIn('<div class="journey-narrative-box">', rendered_dom)
                    self.assertNotIn('&lt;div class="journey-narrative-box"&gt;', rendered_dom)
                    self.assertIn('PERSISTED RECORD &amp; EVIDENCE', rendered_dom)

            self.assertTrue(journey_stage_card_found, "journey-stage-card was not rendered in Incident Journey!")

    def test_16_incident_center_html_and_remediation_rendering_clean(self):
        """Step 8.1B & 8.1C: Verify Incident Center detail cards and investigation history render cleanly.

        Asserts that:
        1. No literal '&#x20;' entities appear in any markdown element.
        2. No literal HTML tags leak as preformatted code blocks (<pre><code>) or inline code.
        3. No markdown escape characters (e.g. '2\\.') appear in approved remediation.
        4. No excessive blank lines (<br><br>) in approved remediation.
        5. Investigation history cards (#1, #2, #3) render without broken HTML or leaked tags.
        6. Safe HTML helpers unescape markdown backslashes while fully preserving XSS protection and legitimate paths.
        """
        import markdown_it
        from app.utils.helpers import safe_html_text, clean_html_block, format_remediation_html
        md = markdown_it.MarkdownIt()

        # Unit verification of format_remediation_html and safe_html_text
        self.assertEqual(safe_html_text("2\\. Verify status"), "2. Verify status")
        self.assertEqual(safe_html_text("Value&#x20;with&#x20;entities"), "Value with entities")
        self.assertIn("&lt;script&gt;", safe_html_text("<script>alert('xss')</script>"))
        self.assertNotIn("<script>", safe_html_text("<script>alert('xss')</script>"))
        self.assertEqual(clean_html_block("  <div>&#x20;Hello&#x20;</div>  "), "<div> Hello </div>")

        # Step 8.1C remediation formatting assertions
        rem_input = "1\\. Step one.\n\n2\\. Step two.\n\n\n3\\. Step three with C:\\logs\\incident.log"
        formatted_rem = format_remediation_html(rem_input)
        self.assertNotIn("<br><br>", formatted_rem, "Remediation must not contain excessive blank lines (<br><br>)")
        self.assertNotIn(r"1\.", formatted_rem)
        self.assertNotIn(r"2\.", formatted_rem)
        self.assertIn("1. Step one.<br>2. Step two.<br>3. Step three with C:\\logs\\incident.log", formatted_rem)

        # Create multi-investigation test database
        inc_id = db.create_incident(
            title="payment deducted but order not confirmed",
            description="payment deducted but order not confirmed",
            department="Not sure",
            impact="Not sure",
            db_path=self.test_db_path
        )

        # Investigation 1 (pending, no final_action)
        incident_service.save_investigation_proposal(
            incident_id=inc_id,
            investigation_data={
                "ai_recommendation": "Automated AI diagnostic synthesis is currently unavailable.",
                "confidence": None,
                "is_fallback": True,
                "root_cause_hypotheses": [],
                "evidence_citations": [],
            },
            actor="AI Fallback",
            db_path=self.test_db_path
        )
        # Investigation 2 (pending, no final_action)
        incident_service.save_investigation_proposal(
            incident_id=inc_id,
            investigation_data={
                "ai_recommendation": "Automated AI diagnostic synthesis is currently unavailable.",
                "confidence": None,
                "is_fallback": True,
                "root_cause_hypotheses": [],
                "evidence_citations": [],
            },
            actor="AI Fallback",
            db_path=self.test_db_path
        )
        # Investigation 3 (approved with operator-authored numbered remediation)
        inv3_id = incident_service.save_investigation_proposal(
            incident_id=inc_id,
            investigation_data={
                "ai_recommendation": "Automated AI diagnostic synthesis is currently unavailable.",
                "confidence": None,
                "is_fallback": True,
                "root_cause_hypotheses": [],
                "evidence_citations": [],
            },
            actor="AI Fallback",
            db_path=self.test_db_path
        )
        incident_service.apply_investigation_decision(
            investigation_id=inv3_id,
            decision="Approved",
            final_action="1. Identify the affected order.\n\n2. Verify the payment status.\n\n3. Inspect the corresponding order record.",
            reviewer_notes="Approved operator-authored remediation.",
            actor="Operator",
            db_path=self.test_db_path
        )

        with patch.object(app.config, "DATABASE_PATH", self.test_db_path), \
             patch.object(db, "DATABASE_PATH", self.test_db_path):
            at = AppTest.from_file(self.script_path)
            at.run(timeout=30)

            # Navigate to Incident Center
            at.sidebar.radio[0].set_value("Incident Center")
            at.run(timeout=30)

            self.assertEqual(len(at.exception), 0)

            overview_card_found = False
            inv_history_found = False

            for idx, m in enumerate(at.markdown):
                val = m.value
                if val.strip().startswith("<style"):
                    continue

                # Defect 1: No literal &#x20; entities
                self.assertNotIn("&#x20;", val, f"Literal &#x20; found in markdown[{idx}]: {val}")

                # Defect 3: No escaped dots like 2\. in markdown
                self.assertNotIn(r"2\.", val, f"Escaped 2\\. found in markdown[{idx}]: {val}")

                # Step 8.1C: No excessive blank lines (<br><br>) in markdown elements
                self.assertNotIn("<br><br>", val, f"Excessive blank lines (<br><br>) found in markdown[{idx}]: {val}")

                # Render through CommonMark parser
                rendered_dom = md.render(val)

                # Defect 1 (rendered): No literal &#x20; in rendered HTML
                self.assertNotIn("&#x20;", rendered_dom, f"Literal &#x20; in rendered DOM[{idx}]: {rendered_dom}")

                # Defect 2: No literal HTML tags leaked into <pre><code> blocks
                if "<pre><code>" in rendered_dom:
                    self.assertNotIn("<div", rendered_dom, f"Leaked <div> in code block markdown[{idx}]:\n{rendered_dom}")
                    self.assertNotIn("&lt;div", rendered_dom, f"Leaked &lt;div in code block markdown[{idx}]:\n{rendered_dom}")
                    self.assertNotIn("Investigation", rendered_dom, f"Leaked Investigation in code block markdown[{idx}]:\n{rendered_dom}")

                # Defect 4: Clean investigation history cards
                if "Investigation #" in val:
                    inv_history_found = True
                    self.assertIn("metric-card", val)
                    self.assertNotIn("<pre><code>", rendered_dom, f"Investigation card turned into code block:\n{rendered_dom}")
                    self.assertNotIn("&lt;div style=", rendered_dom, f"Leaked literal div in investigation card:\n{rendered_dom}")

                if 'class="metric-card"' in val and "INC-" in val:
                    overview_card_found = True
                    self.assertNotIn("<pre><code>", rendered_dom)

            self.assertTrue(overview_card_found, "Overview card not found in Incident Center!")
            self.assertTrue(inv_history_found, "Investigation history cards not found in Incident Center!")


if __name__ == "__main__":
    unittest.main()





