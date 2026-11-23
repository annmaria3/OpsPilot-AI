"""OpsPilot AI - SQLite Database Layer.

Provides connection management, schema initialization with non-destructive
migrations, foreign-key enforcement, and CRUD operations for:
- Incidents
- Investigations (preserving AI recommendations vs human decisions)
- Audit Logs (immutable, append-only, protected by ON DELETE RESTRICT and SQLite triggers)
- Knowledge Documents & Chunks
"""

import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from app.config import DATABASE_PATH
from app.utils.helpers import clean_text


def get_connection(db_path: Optional[Union[Path, str]] = None) -> sqlite3.Connection:
    """Create and configure a connection to the SQLite database.

    Configures:
    1. sqlite3.Row row factory for clean name-based column access.
    2. Foreign key enforcement (PRAGMA foreign_keys = ON).
    """
    target_path = Path(db_path) if db_path is not None else DATABASE_PATH
    conn = sqlite3.connect(str(target_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def initialize_database(db_path: Optional[Union[Path, str]] = None) -> None:
    """Initialize the 5 relational tables if they do not exist.

    Implements safe, non-destructive migrations:
    1. Adds missing columns to 'incidents' without touching existing rows.
    2. Migrates 'audit_logs' foreign keys (incident_id and investigation_id)
       to ON DELETE RESTRICT with explicit transaction handling.
    3. Installs database-level triggers to enforce audit log immutability (ABORT
       on UPDATE or DELETE).
    """
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()

        # ----------------------------------------------------------------------
        # 1. INCIDENTS TABLE
        # ----------------------------------------------------------------------
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                department TEXT,
                impact TEXT,
                category TEXT,
                severity TEXT,
                status TEXT DEFAULT 'Open',
                created_at TEXT NOT NULL
            );
        """)

        # Safe migration: add columns if an older version of the table exists
        cursor.execute("PRAGMA table_info(incidents);")
        existing_cols = {row["name"] for row in cursor.fetchall()}

        if "category" not in existing_cols:
            cursor.execute("ALTER TABLE incidents ADD COLUMN category TEXT;")
        if "severity" not in existing_cols:
            cursor.execute("ALTER TABLE incidents ADD COLUMN severity TEXT;")

        # ----------------------------------------------------------------------
        # 2. INVESTIGATIONS TABLE
        # ----------------------------------------------------------------------
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS investigations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                incident_id INTEGER NOT NULL,
                confirmed_facts TEXT,
                root_cause_hypotheses TEXT,
                evidence_citations TEXT,
                ai_recommendation TEXT NOT NULL,
                human_decision TEXT DEFAULT 'Pending',
                final_action TEXT,
                reviewer_notes TEXT,
                confidence REAL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (incident_id) REFERENCES incidents (id) ON DELETE CASCADE
            );
        """)

        # ----------------------------------------------------------------------
        # 3. AUDIT LOGS TABLE
        # ----------------------------------------------------------------------
        # Both incident_id and investigation_id use ON DELETE RESTRICT to guarantee
        # that neither incidents nor investigations can be deleted while referenced
        # in the immutable audit trail.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                incident_id INTEGER NOT NULL,
                investigation_id INTEGER,
                action TEXT NOT NULL,
                actor TEXT NOT NULL DEFAULT 'System',
                notes TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (incident_id) REFERENCES incidents (id) ON DELETE RESTRICT,
                FOREIGN KEY (investigation_id) REFERENCES investigations (id) ON DELETE RESTRICT
            );
        """)

        # ----------------------------------------------------------------------
        # AUDIT LOGS MIGRATION (Idempotent, Transaction-Safe)
        # ----------------------------------------------------------------------
        # Detect if an existing audit_logs table was created with old foreign-key rules
        # (e.g. incident_id with CASCADE, or investigation_id with SET NULL).
        cursor.execute("PRAGMA foreign_key_list(audit_logs);")
        audit_fks = cursor.fetchall()
        needs_audit_migration = any(
            (row["from"] == "incident_id" and row["on_delete"].upper() != "RESTRICT")
            or (row["from"] == "investigation_id" and row["on_delete"].upper() != "RESTRICT")
            for row in audit_fks
        )

        if needs_audit_migration:
            # Commit any pending read/DDL work so no transaction is active
            conn.commit()
            # In SQLite, PRAGMA foreign_keys can only be toggled outside a transaction
            conn.execute("PRAGMA foreign_keys = OFF;")
            try:
                conn.execute("BEGIN TRANSACTION;")
                cursor.execute("""
                    CREATE TABLE audit_logs_migrated (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        incident_id INTEGER NOT NULL,
                        investigation_id INTEGER,
                        action TEXT NOT NULL,
                        actor TEXT NOT NULL DEFAULT 'System',
                        notes TEXT,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (incident_id) REFERENCES incidents (id) ON DELETE RESTRICT,
                        FOREIGN KEY (investigation_id) REFERENCES investigations (id) ON DELETE RESTRICT
                    );
                """)
                cursor.execute("""
                    INSERT INTO audit_logs_migrated (
                        id, incident_id, investigation_id, action, actor, notes, created_at
                    )
                    SELECT id, incident_id, investigation_id, action, actor, notes, created_at
                    FROM audit_logs;
                """)
                cursor.execute("DROP TABLE audit_logs;")
                cursor.execute("ALTER TABLE audit_logs_migrated RENAME TO audit_logs;")
                conn.execute("COMMIT;")
            except Exception:
                conn.execute("ROLLBACK;")
                raise
            finally:
                conn.execute("PRAGMA foreign_keys = ON;")

        # ----------------------------------------------------------------------
        # IMMUTABILITY TRIGGERS FOR AUDIT LOGS
        # ----------------------------------------------------------------------
        # Block any direct SQL UPDATE on audit_logs
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS prevent_audit_log_update
            BEFORE UPDATE ON audit_logs
            BEGIN
                SELECT RAISE(ABORT, 'Audit logs are immutable');
            END;
        """)

        # Block any direct SQL DELETE on audit_logs
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS prevent_audit_log_delete
            BEFORE DELETE ON audit_logs
            BEGIN
                SELECT RAISE(ABORT, 'Audit logs are immutable');
            END;
        """)

        # ----------------------------------------------------------------------
        # 4. KNOWLEDGE DOCUMENTS TABLE
        # ----------------------------------------------------------------------
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_name TEXT NOT NULL UNIQUE,
                file_path TEXT NOT NULL,
                file_size_bytes INTEGER,
                content_hash TEXT,
                document_type TEXT,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'Indexed',
                uploaded_at TEXT NOT NULL
            );
        """)

        # Safe non-destructive migration: add missing columns if older schema exists
        cursor.execute("PRAGMA table_info(knowledge_documents);")
        existing_doc_cols = {row["name"] for row in cursor.fetchall()}

        if "file_size_bytes" not in existing_doc_cols:
            cursor.execute("ALTER TABLE knowledge_documents ADD COLUMN file_size_bytes INTEGER;")
        if "content_hash" not in existing_doc_cols:
            cursor.execute("ALTER TABLE knowledge_documents ADD COLUMN content_hash TEXT;")
        if "document_type" not in existing_doc_cols:
            cursor.execute("ALTER TABLE knowledge_documents ADD COLUMN document_type TEXT;")
        if "chunk_count" not in existing_doc_cols:
            cursor.execute("ALTER TABLE knowledge_documents ADD COLUMN chunk_count INTEGER NOT NULL DEFAULT 0;")
        if "status" not in existing_doc_cols:
            cursor.execute("ALTER TABLE knowledge_documents ADD COLUMN status TEXT NOT NULL DEFAULT 'Indexed';")

        # ----------------------------------------------------------------------
        # 5. KNOWLEDGE CHUNKS TABLE
        # ----------------------------------------------------------------------
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                page_number INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (document_id) REFERENCES knowledge_documents (id) ON DELETE CASCADE
            );
        """)

        # Performance index on document_id foreign key
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON knowledge_chunks (document_id);
        """)

        # ----------------------------------------------------------------------
        # 6. SQLITE FTS5 SEARCH INDEX & SYNCHRONIZATION TRIGGERS
        # ----------------------------------------------------------------------
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_chunks_fts USING fts5(
                chunk_id UNINDEXED,
                document_id UNINDEXED,
                document_name UNINDEXED,
                page_number UNINDEXED,
                content,
                tokenize='porter unicode61'
            );
        """)

        # Trigger 1: Inserting a chunk inserts into FTS
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS sync_knowledge_chunk_insert
            AFTER INSERT ON knowledge_chunks
            BEGIN
                INSERT INTO knowledge_chunks_fts (
                    chunk_id, document_id, document_name, page_number, content
                )
                SELECT
                    NEW.id,
                    NEW.document_id,
                    d.document_name,
                    NEW.page_number,
                    NEW.content
                FROM knowledge_documents d
                WHERE d.id = NEW.document_id;
            END;
        """)

        # Trigger 2: Updating a chunk updates FTS
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS sync_knowledge_chunk_update
            AFTER UPDATE ON knowledge_chunks
            BEGIN
                DELETE FROM knowledge_chunks_fts WHERE chunk_id = OLD.id;
                INSERT INTO knowledge_chunks_fts (
                    chunk_id, document_id, document_name, page_number, content
                )
                SELECT
                    NEW.id,
                    NEW.document_id,
                    d.document_name,
                    NEW.page_number,
                    NEW.content
                FROM knowledge_documents d
                WHERE d.id = NEW.document_id;
            END;
        """)

        # Trigger 3: Deleting a chunk removes its FTS entry
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS sync_knowledge_chunk_delete
            AFTER DELETE ON knowledge_chunks
            BEGIN
                DELETE FROM knowledge_chunks_fts WHERE chunk_id = OLD.id;
            END;
        """)

        # Trigger 4: Deleting a document cleans all associated FTS entries
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS sync_knowledge_doc_delete_fts
            AFTER DELETE ON knowledge_documents
            BEGIN
                DELETE FROM knowledge_chunks_fts WHERE document_id = OLD.id;
            END;
        """)

        # Synchronize pre-existing chunks not yet present in FTS (idempotent, avoids duplicates)
        cursor.execute("""
            INSERT INTO knowledge_chunks_fts (
                chunk_id, document_id, document_name, page_number, content
            )
            SELECT c.id, c.document_id, d.document_name, c.page_number, c.content
            FROM knowledge_chunks c
            JOIN knowledge_documents d ON c.document_id = d.id
            WHERE c.id NOT IN (SELECT chunk_id FROM knowledge_chunks_fts);
        """)

        conn.commit()
    finally:
        conn.close()


# ==============================================================================
# INCIDENT OPERATIONS
# ==============================================================================

def create_incident(
    title: str,
    description: str,
    department: Optional[str] = None,
    impact: Optional[str] = None,
    category: Optional[str] = None,
    severity: Optional[str] = None,
    status: str = "Open",
    db_path: Optional[Union[Path, str]] = None,
) -> int:
    """Save a new incident into SQLite and return its auto-generated ID."""
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute("""
            INSERT INTO incidents (
                title, description, department, impact, category, severity, status, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            title.strip(),
            description.strip(),
            department,
            impact,
            category,
            severity,
            status,
            now_str,
        ))

        incident_id = cursor.lastrowid
        conn.commit()
        return incident_id
    finally:
        conn.close()


def get_incident_by_id(
    incident_id: int,
    db_path: Optional[Union[Path, str]] = None
) -> Optional[sqlite3.Row]:
    """Retrieve a single incident by its primary key ID."""
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, title, description, department, impact, category, severity, status, created_at
            FROM incidents
            WHERE id = ?
        """, (incident_id,))
        return cursor.fetchone()
    finally:
        conn.close()


def get_all_incidents(
    db_path: Optional[Union[Path, str]] = None
) -> List[sqlite3.Row]:
    """Retrieve all incidents ordered newest first."""
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, title, description, department, impact, category, severity, status, created_at
            FROM incidents
            ORDER BY id DESC
        """)
        return cursor.fetchall()
    finally:
        conn.close()


def update_incident_status(
    incident_id: int,
    status: str,
    db_path: Optional[Union[Path, str]] = None
) -> bool:
    """Update the status of an existing incident."""
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE incidents
            SET status = ?
            WHERE id = ?
        """, (status, incident_id))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def update_incident_classification(
    incident_id: int,
    category: str,
    severity: str,
    department: Optional[str] = None,
    impact: Optional[str] = None,
    db_path: Optional[Union[Path, str]] = None
) -> bool:
    """Update classification fields determined by AI or user."""
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE incidents
            SET category = COALESCE(?, category),
                severity = COALESCE(?, severity),
                department = COALESCE(?, department),
                impact = COALESCE(?, impact)
            WHERE id = ?
        """, (category, severity, department, impact, incident_id))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def apply_incident_classification_with_audit(
    incident_id: int,
    category: Optional[str] = None,
    severity: Optional[str] = None,
    department: Optional[str] = None,
    impact: Optional[str] = None,
    actor: str = "Operator",
    notes: Optional[str] = None,
    db_path: Optional[Union[Path, str]] = None,
) -> bool:
    """Atomically update incident classification and record a CLASSIFICATION_APPROVED audit log.

    Executes both operations in a single SQLite transaction using one connection.
    Commits only when both succeed; rolls back if either operation fails.
    Returns False if the incident does not exist.
    """
    conn = get_connection(db_path)
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE incidents
                SET category = COALESCE(?, category),
                    severity = COALESCE(?, severity),
                    department = COALESCE(?, department),
                    impact = COALESCE(?, impact)
                WHERE id = ?
            """, (category, severity, department, impact, incident_id))

            if cursor.rowcount == 0:
                return False

            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute("""
                INSERT INTO audit_logs (
                    incident_id, investigation_id, action, actor, notes, created_at
                )
                VALUES (?, NULL, ?, ?, ?, ?)
            """, (
                incident_id,
                "CLASSIFICATION_APPROVED",
                actor,
                notes,
                now_str,
            ))

        return True
    finally:
        conn.close()


def get_incident_stats(
    db_path: Optional[Union[Path, str]] = None
) -> Dict[str, int]:
    """Calculate dynamic counts for dashboard metrics."""
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()

        # Total incidents
        cursor.execute("SELECT COUNT(*) FROM incidents;")
        total = cursor.fetchone()[0]

        # Open incidents (awaiting resolution)
        cursor.execute("SELECT COUNT(*) FROM incidents WHERE status NOT IN ('Resolved', 'Closed');")
        open_count = cursor.fetchone()[0]

        # Critical incidents
        cursor.execute("SELECT COUNT(*) FROM incidents WHERE severity = 'Critical' OR impact = 'Critical';")
        critical = cursor.fetchone()[0]

        # Resolved incidents
        cursor.execute("SELECT COUNT(*) FROM incidents WHERE status IN ('Resolved', 'Closed');")
        resolved = cursor.fetchone()[0]

        return {
            "total": total,
            "open": open_count,
            "critical": critical,
            "resolved": resolved,
        }
    finally:
        conn.close()


# ==============================================================================
# INVESTIGATION OPERATIONS
# ==============================================================================

def save_investigation(
    incident_id: int,
    ai_recommendation: str,
    confirmed_facts: Optional[str] = None,
    root_cause_hypotheses: Optional[str] = None,
    evidence_citations: Optional[str] = None,
    confidence: Optional[float] = None,
    human_decision: str = "Pending",
    final_action: Optional[str] = None,
    reviewer_notes: Optional[str] = None,
    db_path: Optional[Union[Path, str]] = None,
) -> int:
    """Record an AI investigation run while preserving original recommendation."""
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute("""
            INSERT INTO investigations (
                incident_id, confirmed_facts, root_cause_hypotheses,
                evidence_citations, ai_recommendation, human_decision,
                final_action, reviewer_notes, confidence, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            incident_id,
            confirmed_facts,
            root_cause_hypotheses,
            evidence_citations,
            ai_recommendation,
            human_decision,
            final_action,
            reviewer_notes,
            confidence,
            now_str,
        ))

        investigation_id = cursor.lastrowid
        conn.commit()
        return investigation_id
    finally:
        conn.close()


def get_investigations_for_incident(
    incident_id: int,
    db_path: Optional[Union[Path, str]] = None
) -> List[sqlite3.Row]:
    """Retrieve all investigations for an incident, newest first."""
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, incident_id, confirmed_facts, root_cause_hypotheses,
                   evidence_citations, ai_recommendation, human_decision,
                   final_action, reviewer_notes, confidence, created_at
            FROM investigations
            WHERE incident_id = ?
            ORDER BY id DESC
        """, (incident_id,))
        return cursor.fetchall()
    finally:
        conn.close()


def update_investigation_decision(
    investigation_id: int,
    human_decision: str,
    final_action: Optional[str] = None,
    reviewer_notes: Optional[str] = None,
    db_path: Optional[Union[Path, str]] = None,
) -> bool:
    """Update human review decision and approved action for an investigation."""
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE investigations
            SET human_decision = ?,
                final_action = COALESCE(?, final_action),
                reviewer_notes = COALESCE(?, reviewer_notes)
            WHERE id = ?
        """, (human_decision, final_action, reviewer_notes, investigation_id))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def save_investigation_with_audit(
    incident_id: int,
    ai_recommendation: str,
    confirmed_facts: Optional[str] = None,
    root_cause_hypotheses: Optional[str] = None,
    evidence_citations: Optional[str] = None,
    confidence: Optional[float] = None,
    human_decision: str = "Pending",
    actor: str = "System",
    db_path: Optional[Union[Path, str]] = None,
) -> int:
    """Atomically record an AI investigation proposal and an INVESTIGATION_RUN audit log.

    Executes both operations in a single SQLite transaction using one connection.
    If the incident does not exist or any operation fails, the entire transaction rolls back.
    If the incident status is 'Open', transitions it to 'Pending Approval'.

    Returns:
        Generated investigation_id (int).

    Raises:
        ValueError: If incident_id does not exist, or ai_recommendation is invalid.
    """
    clean_rec = ai_recommendation.strip() if isinstance(ai_recommendation, str) else str(ai_recommendation).strip()
    if not clean_rec:
        raise ValueError("ai_recommendation is required and must be a non-empty string.")

    if confidence is not None:
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError(f"confidence must be a numeric float, got {type(confidence).__name__}")
        if not (0.0 <= float(confidence) <= 1.0):
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {confidence}")

    safe_actor = clean_text(actor) or "System"

    conn = get_connection(db_path)
    try:
        with conn:
            cursor = conn.cursor()

            # 1. Verify incident exists
            cursor.execute("SELECT id, status FROM incidents WHERE id = ?;", (incident_id,))
            inc_row = cursor.fetchone()
            if inc_row is None:
                raise ValueError(f"Incident with ID {incident_id} not found.")

            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 2. Insert investigation proposal
            cursor.execute("""
                INSERT INTO investigations (
                    incident_id, confirmed_facts, root_cause_hypotheses,
                    evidence_citations, ai_recommendation, human_decision,
                    final_action, reviewer_notes, confidence, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?);
            """, (
                incident_id,
                confirmed_facts,
                root_cause_hypotheses,
                evidence_citations,
                clean_rec,
                human_decision,
                float(confidence) if confidence is not None else None,
                now_str,
            ))

            investigation_id = cursor.lastrowid

            # 3. Insert INVESTIGATION_RUN audit log
            cursor.execute("""
                INSERT INTO audit_logs (
                    incident_id, investigation_id, action, actor, notes, created_at
                )
                VALUES (?, ?, 'INVESTIGATION_RUN', ?, 'AI investigation conducted and proposal recorded.', ?);
            """, (
                incident_id,
                investigation_id,
                safe_actor,
                now_str,
            ))

            # 4. Update incident status from 'Open' to 'Pending Approval'
            cursor.execute("""
                UPDATE incidents
                SET status = 'Pending Approval'
                WHERE id = ? AND status = 'Open';
            """, (incident_id,))

        return investigation_id
    finally:
        conn.close()


def get_investigation_by_id(
    investigation_id: int,
    db_path: Optional[Union[Path, str]] = None,
) -> Optional[sqlite3.Row]:
    """Retrieve an investigation record by primary key."""
    if not isinstance(investigation_id, int) or investigation_id <= 0:
        return None

    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, incident_id, confirmed_facts, root_cause_hypotheses,
                   evidence_citations, ai_recommendation, human_decision,
                   final_action, reviewer_notes, confidence, created_at
            FROM investigations
            WHERE id = ?;
        """, (investigation_id,))
        return cursor.fetchone()
    finally:
        conn.close()


def apply_investigation_decision_with_audit(
    investigation_id: int,
    incident_id: int,
    human_decision: str,
    final_action: Optional[str] = None,
    reviewer_notes: Optional[str] = None,
    actor: str = "Operator",
    action: str = "RECOMMENDATION_APPROVED",
    audit_notes: Optional[str] = None,
    db_path: Optional[Union[Path, str]] = None,
) -> bool:
    """Atomically update an investigation decision and insert an audit log in one transaction.

    Validates:
    - Permitted audit action in {'RECOMMENDATION_APPROVED', 'RECOMMENDATION_EDITED', 'RECOMMENDATION_REJECTED'}.
    - Permitted human_decision in {'Approved', 'Rejected'}.
    - Matching action/decision combinations:
      - 'Approved' with 'RECOMMENDATION_APPROVED' or 'RECOMMENDATION_EDITED'
      - 'Rejected' with 'RECOMMENDATION_REJECTED'
    - Investigation exists and belongs to incident_id (incident/investigation consistency).
    - If already decided with the exact same human_decision, duplicate submission is suppressed and returns False.
    - If human_decision == 'Rejected', final_action is explicitly cleared to NULL (no COALESCE retention).
    - If human_decision == 'Approved' and action == 'RECOMMENDATION_APPROVED', final_action defaults to ai_recommendation.
    - If human_decision == 'Approved' and action == 'RECOMMENDATION_EDITED', final_action is set to the edited text,
      preserving the original ai_recommendation untouched.
    - Updates incident status to 'Approved' on approval, or resets to 'Open' on rejection.
    - Commits both the investigation update, audit log, and incident status atomically in a single transaction.

    Returns:
        True on success, False if duplicate submission.

    Raises:
        ValueError: If action/decision validation fails or investigation does not match incident.
    """
    valid_actions = {"RECOMMENDATION_APPROVED", "RECOMMENDATION_EDITED", "RECOMMENDATION_REJECTED"}
    if action not in valid_actions:
        raise ValueError(f"Invalid audit action '{action}'. Must be one of {sorted(valid_actions)}.")

    clean_decision = clean_text(human_decision).title()
    if clean_decision not in {"Approved", "Rejected"}:
        raise ValueError(f"Invalid human decision '{human_decision}'. Must be 'Approved' or 'Rejected'.")

    # Validate decision / action pair compatibility
    if clean_decision == "Approved" and action == "RECOMMENDATION_REJECTED":
        raise ValueError("Cannot log 'RECOMMENDATION_REJECTED' for an 'Approved' decision.")
    if clean_decision == "Rejected" and action in {"RECOMMENDATION_APPROVED", "RECOMMENDATION_EDITED"}:
        raise ValueError(f"Cannot log '{action}' for a 'Rejected' decision.")

    safe_actor = clean_text(actor) or "Operator"
    clean_notes = clean_text(reviewer_notes) or None

    conn = get_connection(db_path)
    try:
        with conn:
            cursor = conn.cursor()

            # 1. Fetch investigation and verify incident consistency
            cursor.execute("""
                SELECT id, incident_id, ai_recommendation, human_decision, final_action
                FROM investigations
                WHERE id = ?;
            """, (investigation_id,))
            inv_row = cursor.fetchone()
            if inv_row is None:
                raise ValueError(f"Investigation with ID {investigation_id} not found.")

            if inv_row["incident_id"] != incident_id:
                raise ValueError(
                    f"Investigation #{investigation_id} belongs to incident #{inv_row['incident_id']}, "
                    f"not incident #{incident_id}."
                )

            # Prevent duplicate decision submission and duplicate audit log creation
            if inv_row["human_decision"] == clean_decision and clean_decision in {"Approved", "Rejected"}:
                return False

            # 2. Verify incident exists
            cursor.execute("SELECT id, status FROM incidents WHERE id = ?;", (incident_id,))
            inc_row = cursor.fetchone()
            if inc_row is None:
                raise ValueError(f"Incident with ID {incident_id} not found.")

            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 3. Update investigation row based on decision
            if clean_decision == "Approved":
                if action == "RECOMMENDATION_EDITED":
                    clean_final = final_action.strip() if isinstance(final_action, str) else ""
                    if not clean_final:
                        raise ValueError("final_action must be provided when editing recommendation.")
                    resolved_action = clean_final
                else:
                    resolved_action = final_action.strip() if (isinstance(final_action, str) and final_action.strip()) else inv_row["ai_recommendation"]

                cursor.execute("""
                    UPDATE investigations
                    SET human_decision = 'Approved',
                        final_action = ?,
                        reviewer_notes = COALESCE(?, reviewer_notes)
                    WHERE id = ?;
                """, (resolved_action, clean_notes, investigation_id))

                new_inc_status = "Approved"
                default_audit_note = "Operator approved AI recommendation." if action == "RECOMMENDATION_APPROVED" else "Operator edited and approved recommendation."

            else:  # Rejected
                # Strictly set final_action = NULL (never retain previous value through COALESCE)
                cursor.execute("""
                    UPDATE investigations
                    SET human_decision = 'Rejected',
                        final_action = NULL,
                        reviewer_notes = ?
                    WHERE id = ?;
                """, (clean_notes, investigation_id))

                new_inc_status = "Open"
                default_audit_note = "Operator rejected AI recommendation."

            # 4. Insert audit log
            if audit_notes and audit_notes.strip():
                resolved_audit_note = audit_notes.strip()
            else:
                resolved_audit_note = clean_notes if clean_notes else default_audit_note

            cursor.execute("""
                INSERT INTO audit_logs (
                    incident_id, investigation_id, action, actor, notes, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?);
            """, (
                incident_id,
                investigation_id,
                action,
                safe_actor,
                resolved_audit_note,
                now_str,
            ))

            # 5. Update incident status
            cursor.execute("""
                UPDATE incidents
                SET status = ?
                WHERE id = ?;
            """, (new_inc_status, incident_id))

        return True
    finally:
        conn.close()


# ==============================================================================
# AUDIT LOG OPERATIONS (APPEND-ONLY, IMMUTABLE)
# ==============================================================================

def add_audit_log(
    incident_id: int,
    action: str,
    actor: str = "System",
    notes: Optional[str] = None,
    investigation_id: Optional[int] = None,
    db_path: Optional[Union[Path, str]] = None,
) -> int:
    """Log an auditable decision, state transition, or AI event.

    Audit records are strictly append-only and enforced by database triggers.
    """
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute("""
            INSERT INTO audit_logs (
                incident_id, investigation_id, action, actor, notes, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            incident_id,
            investigation_id,
            action,
            actor,
            notes,
            now_str,
        ))

        log_id = cursor.lastrowid
        conn.commit()
        return log_id
    finally:
        conn.close()


def get_audit_logs(
    incident_id: int,
    db_path: Optional[Union[Path, str]] = None
) -> List[sqlite3.Row]:
    """Retrieve full audit history for an incident in chronological order."""
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, incident_id, investigation_id, action, actor, notes, created_at
            FROM audit_logs
            WHERE incident_id = ?
            ORDER BY id ASC
        """, (incident_id,))
        return cursor.fetchall()
    finally:
        conn.close()


# ==============================================================================
# KNOWLEDGE BASE OPERATIONS
# ==============================================================================

def create_knowledge_document(
    document_name: str,
    file_path: str,
    file_size_bytes: Optional[int] = None,
    content_hash: Optional[str] = None,
    document_type: str = "Runbook",
    chunk_count: int = 0,
    status: str = "Indexed",
    db_path: Optional[Union[Path, str]] = None,
) -> int:
    """Register a new knowledge document into SQLite.

    Raises:
        ValueError: If document_name or file_path is empty.
        sqlite3.IntegrityError: If document_name already exists (UNIQUE constraint).
    """
    clean_name = str(document_name).strip() if document_name is not None else ""
    clean_path = str(file_path).strip() if file_path is not None else ""
    if not clean_name:
        raise ValueError("Document name cannot be empty.")
    if not clean_path:
        raise ValueError("File path cannot be empty.")

    clean_type = str(document_type).strip() if document_type else "Runbook"
    clean_status = str(status).strip() if status else "Indexed"
    size_val = int(file_size_bytes) if file_size_bytes is not None else None
    hash_val = str(content_hash).strip() if content_hash else None
    count_val = max(0, int(chunk_count))

    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute("""
            INSERT INTO knowledge_documents (
                document_name, file_path, file_size_bytes, content_hash,
                document_type, chunk_count, status, uploaded_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            clean_name,
            clean_path,
            size_val,
            hash_val,
            clean_type,
            count_val,
            clean_status,
            now_str,
        ))

        doc_id = cursor.lastrowid
        conn.commit()
        return doc_id
    finally:
        conn.close()


def get_all_knowledge_documents(
    db_path: Optional[Union[Path, str]] = None,
) -> List[sqlite3.Row]:
    """Retrieve all knowledge documents, ordered newest first."""
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, document_name, file_path, file_size_bytes, content_hash,
                   document_type, chunk_count, status, uploaded_at
            FROM knowledge_documents
            ORDER BY id DESC;
        """)
        return cursor.fetchall()
    finally:
        conn.close()


def get_knowledge_document_by_id(
    document_id: int,
    db_path: Optional[Union[Path, str]] = None,
) -> Optional[sqlite3.Row]:
    """Retrieve a single knowledge document by its primary key ID."""
    if not isinstance(document_id, int) or document_id <= 0:
        return None

    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, document_name, file_path, file_size_bytes, content_hash,
                   document_type, chunk_count, status, uploaded_at
            FROM knowledge_documents
            WHERE id = ?;
        """, (document_id,))
        return cursor.fetchone()
    finally:
        conn.close()


def get_knowledge_document_by_hash(
    content_hash: str,
    db_path: Optional[Union[Path, str]] = None,
) -> Optional[sqlite3.Row]:
    """Retrieve an existing document matching a SHA-256 content hash."""
    if not content_hash or not str(content_hash).strip():
        return None

    clean_hash = str(content_hash).strip()
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, document_name, file_path, file_size_bytes, content_hash,
                   document_type, chunk_count, status, uploaded_at
            FROM knowledge_documents
            WHERE content_hash = ?
            ORDER BY id ASC
            LIMIT 1;
        """, (clean_hash,))
        return cursor.fetchone()
    finally:
        conn.close()


def delete_knowledge_document(
    document_id: int,
    db_path: Optional[Union[Path, str]] = None,
) -> bool:
    """Delete a document and cascade deletion to all its chunks and FTS records."""
    if not isinstance(document_id, int) or document_id <= 0:
        return False

    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM knowledge_documents WHERE id = ?;", (document_id,))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def insert_knowledge_chunks(
    document_id: int,
    chunks: List[Dict[str, Any]],
    db_path: Optional[Union[Path, str]] = None,
) -> int:
    """Insert multiple knowledge chunks for a document in a single atomic transaction.

    Automatically triggers FTS index population via database triggers and updates
    the document's total chunk_count.

    Raises:
        ValueError: If document_id is invalid, document does not exist, or chunk data is malformed.
    """
    if not isinstance(document_id, int) or document_id <= 0:
        raise ValueError(f"Invalid document ID: {document_id}. Must be a positive integer.")

    if not chunks:
        return 0

    validated_rows = []
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for idx, c in enumerate(chunks):
        if not isinstance(c, dict):
            raise ValueError(f"Chunk at index {idx} must be a dictionary.")
        content = str(c.get("content", "")).strip()
        if not content:
            raise ValueError(f"Chunk at index {idx} contains empty content.")
        try:
            page_num = int(c.get("page_number", 1))
            if page_num < 1:
                page_num = 1
        except (TypeError, ValueError):
            page_num = 1
        try:
            chunk_idx = int(c.get("chunk_index", idx))
            if chunk_idx < 0:
                chunk_idx = idx
        except (TypeError, ValueError):
            chunk_idx = idx

        validated_rows.append((document_id, page_num, chunk_idx, content, now_str))

    conn = get_connection(db_path)
    try:
        with conn:
            cursor = conn.cursor()

            # Confirm parent document exists
            cursor.execute("SELECT id FROM knowledge_documents WHERE id = ?;", (document_id,))
            if cursor.fetchone() is None:
                raise ValueError(f"Knowledge document with ID {document_id} not found.")

            # Batch insert chunks
            cursor.executemany("""
                INSERT INTO knowledge_chunks (
                    document_id, page_number, chunk_index, content, created_at
                )
                VALUES (?, ?, ?, ?, ?);
            """, validated_rows)

            inserted_count = len(validated_rows)

            # Update document chunk count
            cursor.execute("""
                UPDATE knowledge_documents
                SET chunk_count = (SELECT COUNT(*) FROM knowledge_chunks WHERE document_id = ?)
                WHERE id = ?;
            """, (document_id, document_id))

        return inserted_count
    finally:
        conn.close()


def get_chunks_for_document(
    document_id: int,
    db_path: Optional[Union[Path, str]] = None,
) -> List[sqlite3.Row]:
    """Retrieve all chunks belonging to a document, ordered by page and chunk index."""
    if not isinstance(document_id, int) or document_id <= 0:
        return []

    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, document_id, page_number, chunk_index, content, created_at
            FROM knowledge_chunks
            WHERE document_id = ?
            ORDER BY page_number ASC, chunk_index ASC;
        """, (document_id,))
        return cursor.fetchall()
    finally:
        conn.close()


def search_knowledge_chunks_fts(
    query: str,
    limit: int = 5,
    db_path: Optional[Union[Path, str]] = None,
) -> List[Dict[str, Any]]:
    """Perform BM25 full-text search across indexed knowledge chunks.

    Returns the top matching chunks ordered by BM25 relevance (most relevant first),
    including document metadata, page number, chunk index, raw bm25_score, and relevance_score.

    Score semantics:
    - SQLite FTS5 bm25() returns negative floating-point values where more negative
      scores indicate stronger matches (higher relevance).
    - The returned 'relevance_score' is the positive magnitude: round(abs(raw_bm25_score), 4).
    - This positive value is an unbounded ranking metric, NOT a normalized percentage or
      probability (do not format with '%' or interpret as a 0-1 scale).

    Document metadata freshness:
    - Joins 'knowledge_documents' to return the current, live document name, preventing
      stale source names if a document was renamed. Falls back to FTS document_name via COALESCE.

    Safely sanitizes input query to prevent SQLite FTS5 syntax errors. Returns an empty
    list if query contains no valid alphanumeric search terms or if search errors occur.
    """
    if not query or not isinstance(query, str):
        return []

    words = re.findall(r'\b\w+\b', query)
    if not words:
        return []

    bounded_words = words[:20]
    fts_query_str = " OR ".join(f'"{w}"*' for w in bounded_words)

    safe_limit = max(1, min(int(limit), 100)) if isinstance(limit, (int, float)) else 5

    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                f.chunk_id,
                f.document_id,
                COALESCE(d.document_name, f.document_name) AS document_name,
                f.page_number,
                c.chunk_index,
                f.content,
                bm25(knowledge_chunks_fts) AS bm25_score
            FROM knowledge_chunks_fts f
            LEFT JOIN knowledge_chunks c ON c.id = f.chunk_id
            LEFT JOIN knowledge_documents d ON d.id = f.document_id
            WHERE knowledge_chunks_fts MATCH ?
            ORDER BY bm25_score ASC
            LIMIT ?;
        """, (fts_query_str, safe_limit))

        rows = cursor.fetchall()
        results = []
        for r in rows:
            raw_bm25 = float(r["bm25_score"]) if r["bm25_score"] is not None else 0.0
            results.append({
                "chunk_id": r["chunk_id"],
                "document_id": r["document_id"],
                "document_name": r["document_name"] or "Unknown Document",
                "page_number": r["page_number"] or 1,
                "chunk_index": r["chunk_index"] if r["chunk_index"] is not None else 0,
                "content": r["content"] or "",
                "bm25_score": raw_bm25,
                "relevance_score": round(abs(raw_bm25), 4),
            })
        return results
    except sqlite3.Error:
        return []
    finally:
        conn.close()


def create_knowledge_document_with_chunks(
    document_name: str,
    file_path: str,
    file_size_bytes: int,
    content_hash: str,
    document_type: str,
    chunks: List[Dict[str, Any]],
    status: str = "Indexed",
    db_path: Optional[Union[Path, str]] = None,
) -> Dict[str, Any]:
    """Atomically create a knowledge document record and persist all its chunks.

    Guarantees two-phase consistency within SQLite:
    - If document insert or any chunk insert fails, the entire transaction rolls back.
    - The document record is never committed without all of its chunks.
    - Status is set to 'Indexed' only when both document metadata and chunks succeed.

    Raises:
        ValueError: If document_name, file_path, or content_hash is empty, or chunks is empty/malformed.
        sqlite3.IntegrityError: If document_name already exists (UNIQUE constraint).
        sqlite3.Error: On database failure.
    """
    clean_name = str(document_name).strip() if document_name is not None else ""
    clean_path = str(file_path).strip() if file_path is not None else ""
    clean_hash = str(content_hash).strip() if content_hash is not None else ""

    if not clean_name:
        raise ValueError("Document name cannot be empty.")
    if not clean_path:
        raise ValueError("File path cannot be empty.")
    if not clean_hash:
        raise ValueError("Content hash cannot be empty.")
    if not chunks or not isinstance(chunks, list):
        raise ValueError("Cannot persist knowledge document with empty or invalid chunks list.")

    clean_type = str(document_type).strip() if document_type else "Runbook"
    clean_status = str(status).strip() if status else "Indexed"
    size_val = max(0, int(file_size_bytes)) if file_size_bytes is not None else 0
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Validate all chunks in memory prior to starting database transaction
    validated_chunk_rows = []
    for idx, c in enumerate(chunks):
        if not isinstance(c, dict):
            raise ValueError(f"Chunk at index {idx} must be a dictionary.")
        content = str(c.get("content", "")).strip()
        if not content:
            raise ValueError(f"Chunk at index {idx} contains empty content.")
        try:
            page_num = int(c.get("page_number", 1))
            if page_num < 1:
                page_num = 1
        except (TypeError, ValueError):
            page_num = 1
        try:
            chunk_idx = int(c.get("chunk_index", idx))
            if chunk_idx < 0:
                chunk_idx = idx
        except (TypeError, ValueError):
            chunk_idx = idx

        validated_chunk_rows.append((page_num, chunk_idx, content, now_str))

    chunk_count = len(validated_chunk_rows)

    conn = get_connection(db_path)
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO knowledge_documents (
                    document_name, file_path, file_size_bytes, content_hash,
                    document_type, chunk_count, status, uploaded_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                clean_name,
                clean_path,
                size_val,
                clean_hash,
                clean_type,
                chunk_count,
                clean_status,
                now_str,
            ))
            doc_id = cursor.lastrowid

            # Prepare rows with generated doc_id
            chunk_insert_tuples = [
                (doc_id, p_num, c_idx, cnt, ts)
                for (p_num, c_idx, cnt, ts) in validated_chunk_rows
            ]

            cursor.executemany("""
                INSERT INTO knowledge_chunks (
                    document_id, page_number, chunk_index, content, created_at
                )
                VALUES (?, ?, ?, ?, ?);
            """, chunk_insert_tuples)

        return {
            "document_id": doc_id,
            "document_name": clean_name,
            "file_path": clean_path,
            "file_size_bytes": size_val,
            "content_hash": clean_hash,
            "document_type": clean_type,
            "chunk_count": chunk_count,
            "status": clean_status,
            "uploaded_at": now_str,
        }
    finally:
        conn.close()


