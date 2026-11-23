"""OpsPilot AI - Incident Business Logic Service Layer.

Acts as the intermediary between the presentation layer (Streamlit UI) and
the persistence layer (app/database/db.py). Enforces input validation,
audit logging on lifecycle updates, domain taxonomy checks, and data
formatting without duplicating raw SQL logic.
"""

import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Union

import app.database.db as db
from app.config import (
    AUDIT_ACTIONS,
    DEPARTMENTS,
    INCIDENT_CATEGORIES,
    INCIDENT_STATUSES,
    SEVERITY_LEVELS,
)
from app.utils.helpers import (
    clean_text,
    format_incident_id,
    row_to_dict,
    rows_to_dict_list,
)

# Permitted department options: standard configured departments plus fallback options
PERMITTED_DEPARTMENTS = set(DEPARTMENTS) | {"Other", "Not sure"}

# Permitted impact levels: impact reflects business blast radius
# Aligns with severity levels (Low, Medium, High, Critical) plus 'Not sure'
PERMITTED_IMPACT_LEVELS = set(SEVERITY_LEVELS) | {"Not sure"}


def _normalize_actor(actor: Any, default: str = "System") -> str:
    """Safely normalize actor identifier for audit trails.

    Handles None, empty strings, non-string types, and whitespace.
    """
    cleaned = clean_text(actor)
    return cleaned if cleaned else default


# ==============================================================================
# 1. INCIDENT CREATION
# ==============================================================================

def create_incident(
    title: str,
    description: str,
    department: Optional[str] = None,
    impact: Optional[str] = None,
    category: Optional[str] = None,
    severity: Optional[str] = None,
    status: str = "Open",
    actor: Any = "Employee",
    db_path: Optional[Union[Path, str]] = None,
) -> Dict[str, Any]:
    """Validate and register a new incident in the system.

    Validates:
    - Title must be between 3 and 250 characters.
    - Description must be at least 5 characters.
    - Department must be in PERMITTED_DEPARTMENTS if provided.
    - Impact must be in PERMITTED_IMPACT_LEVELS if provided.
    - Category must be in INCIDENT_CATEGORIES if provided.
    - Severity must be in SEVERITY_LEVELS if provided.
    - Status must be in INCIDENT_STATUSES.

    Automatically records an 'INCIDENT_CREATED' audit log entry.
    Returns the created incident as a dictionary including its formatted_id.
    """
    clean_title = clean_text(title)
    clean_desc = clean_text(description)

    # 1. Required field validations
    if not clean_title or len(clean_title) < 3:
        raise ValueError("Incident title is required and must be at least 3 characters long.")
    if len(clean_title) > 250:
        raise ValueError("Incident title cannot exceed 250 characters.")

    if not clean_desc or len(clean_desc) < 5:
        raise ValueError("Incident description is required and must be at least 5 characters long.")

    # 2. Department validation
    clean_dept = clean_text(department) or None
    if clean_dept and clean_dept not in PERMITTED_DEPARTMENTS:
        raise ValueError(
            f"Invalid department: '{clean_dept}'. Must be one of {sorted(PERMITTED_DEPARTMENTS)}."
        )

    # 3. Impact validation
    clean_impact = clean_text(impact) or None
    if clean_impact and clean_impact not in PERMITTED_IMPACT_LEVELS:
        raise ValueError(
            f"Invalid impact level: '{clean_impact}'. Must be one of {sorted(PERMITTED_IMPACT_LEVELS)}."
        )

    # 4. Category validation
    clean_category = clean_text(category) or None
    if clean_category and clean_category not in INCIDENT_CATEGORIES:
        raise ValueError(
            f"Invalid category: '{clean_category}'. Must be one of {INCIDENT_CATEGORIES}."
        )

    # 5. Severity validation
    clean_severity = clean_text(severity) or None
    if clean_severity and clean_severity not in SEVERITY_LEVELS:
        raise ValueError(
            f"Invalid severity: '{clean_severity}'. Must be one of {SEVERITY_LEVELS}."
        )

    # 6. Status validation
    clean_status = clean_text(status) or "Open"
    if clean_status not in INCIDENT_STATUSES:
        raise ValueError(
            f"Invalid initial status: '{clean_status}'. Must be one of {INCIDENT_STATUSES}."
        )

    # Persist incident via database layer
    incident_id = db.create_incident(
        title=clean_title,
        description=clean_desc,
        department=clean_dept,
        impact=clean_impact,
        category=clean_category,
        severity=clean_severity,
        status=clean_status,
        db_path=db_path,
    )

    # Log initial audit event using standard AUDIT_ACTIONS taxonomy
    safe_actor = _normalize_actor(actor, default="Employee")
    db.add_audit_log(
        incident_id=incident_id,
        action="INCIDENT_CREATED",
        actor=safe_actor,
        notes=f"Incident registered with initial status '{clean_status}'.",
        db_path=db_path,
    )

    # Return full created record
    created = get_incident_by_id(incident_id, db_path=db_path)
    if created is None:
        raise RuntimeError(f"Failed to retrieve incident #{incident_id} immediately after creation.")
    return created


# ==============================================================================
# 2. INCIDENT RETRIEVAL
# ==============================================================================

def get_incident_by_id(
    incident_id: int,
    db_path: Optional[Union[Path, str]] = None
) -> Optional[Dict[str, Any]]:
    """Retrieve a single incident by its integer ID.

    Returns a dictionary with added 'formatted_id' (e.g. INC-0001),
    or None if the incident does not exist.
    """
    if not isinstance(incident_id, int) or incident_id <= 0:
        raise ValueError(f"Invalid incident ID: {incident_id}. Must be a positive integer.")

    row = db.get_incident_by_id(incident_id, db_path=db_path)
    if row is None:
        return None

    data = row_to_dict(row)
    if data is not None:
        data["formatted_id"] = format_incident_id(data["id"])
    return data


def get_all_incidents(
    status_filter: Optional[str] = None,
    department_filter: Optional[str] = None,
    severity_filter: Optional[str] = None,
    db_path: Optional[Union[Path, str]] = None,
) -> List[Dict[str, Any]]:
    """Retrieve all incidents with optional in-memory filtering.

    Ordered newest first by ID descending.
    """
    rows = db.get_all_incidents(db_path=db_path)
    incidents = rows_to_dict_list(rows)

    # Enrich with formatted_id
    for item in incidents:
        item["formatted_id"] = format_incident_id(item["id"])

    # Clean and apply optional filters consistently
    clean_stat_filter = clean_text(status_filter) or None
    clean_dept_filter = clean_text(department_filter) or None
    clean_sev_filter = clean_text(severity_filter) or None

    if clean_stat_filter:
        incidents = [i for i in incidents if i.get("status") == clean_stat_filter]
    if clean_dept_filter:
        incidents = [i for i in incidents if i.get("department") == clean_dept_filter]
    if clean_sev_filter:
        incidents = [i for i in incidents if i.get("severity") == clean_sev_filter]

    return incidents


# ==============================================================================
# 3. STATUS & CLASSIFICATION UPDATES
# ==============================================================================

def update_incident_status(
    incident_id: int,
    status: str,
    actor: Any = "Operator",
    notes: Optional[str] = None,
    db_path: Optional[Union[Path, str]] = None,
) -> bool:
    """Update an incident's lifecycle status and record a STATUS_UPDATED audit log event."""
    clean_status = clean_text(status)
    if clean_status not in INCIDENT_STATUSES:
        raise ValueError(f"Invalid status: '{clean_status}'. Must be one of {INCIDENT_STATUSES}.")

    current = get_incident_by_id(incident_id, db_path=db_path)
    if current is None:
        raise ValueError(f"Incident with ID {incident_id} not found.")

    old_status = current.get("status", "Unknown")
    success = db.update_incident_status(incident_id, clean_status, db_path=db_path)

    if success:
        safe_actor = _normalize_actor(actor, default="Operator")
        audit_note = clean_text(notes) or f"Status changed from '{old_status}' to '{clean_status}'."
        db.add_audit_log(
            incident_id=incident_id,
            action="STATUS_UPDATED",
            actor=safe_actor,
            notes=audit_note,
            db_path=db_path,
        )

    return success


def update_incident_classification(
    incident_id: int,
    category: Optional[str] = None,
    severity: Optional[str] = None,
    department: Optional[str] = None,
    impact: Optional[str] = None,
    actor: Any = "Operator",
    notes: Optional[str] = None,
    db_path: Optional[Union[Path, str]] = None,
) -> bool:
    """Update incident classification parameters and log a CLASSIFICATION_APPROVED audit event.

    Executes classification update and audit record insertion atomically in a single transaction.

    Enforces:
    - Incident must exist.
    - At least one classification field must be provided (non-empty).
    - Category must be in INCIDENT_CATEGORIES if provided.
    - Severity must be in SEVERITY_LEVELS if provided.
    - Department must be in PERMITTED_DEPARTMENTS if provided.
    - Impact must be in PERMITTED_IMPACT_LEVELS if provided.
    - Audit action strictly records 'CLASSIFICATION_APPROVED'.
    """
    if not isinstance(incident_id, int) or incident_id <= 0:
        raise ValueError(f"Invalid incident ID: {incident_id}. Must be a positive integer.")

    current = get_incident_by_id(incident_id, db_path=db_path)
    if current is None:
        raise ValueError(f"Incident with ID {incident_id} not found.")

    clean_cat = clean_text(category) or None
    clean_sev = clean_text(severity) or None
    clean_dept = clean_text(department) or None
    clean_imp = clean_text(impact) or None

    # Prevent empty updates
    if clean_cat is None and clean_sev is None and clean_dept is None and clean_imp is None:
        raise ValueError(
            "At least one classification parameter (category, severity, department, or impact) must be provided."
        )

    # Taxonomy validations
    if clean_cat and clean_cat not in INCIDENT_CATEGORIES:
        raise ValueError(f"Invalid category: '{clean_cat}'. Must be one of {INCIDENT_CATEGORIES}.")

    if clean_sev and clean_sev not in SEVERITY_LEVELS:
        raise ValueError(f"Invalid severity: '{clean_sev}'. Must be one of {SEVERITY_LEVELS}.")

    if clean_dept and clean_dept not in PERMITTED_DEPARTMENTS:
        raise ValueError(
            f"Invalid department: '{clean_dept}'. Must be one of {sorted(PERMITTED_DEPARTMENTS)}."
        )

    if clean_imp and clean_imp not in PERMITTED_IMPACT_LEVELS:
        raise ValueError(
            f"Invalid impact level: '{clean_imp}'. Must be one of {sorted(PERMITTED_IMPACT_LEVELS)}."
        )

    safe_actor = _normalize_actor(actor, default="Operator")
    summary_parts = []
    if clean_cat:
        summary_parts.append(f"category='{clean_cat}'")
    if clean_sev:
        summary_parts.append(f"severity='{clean_sev}'")
    if clean_dept:
        summary_parts.append(f"department='{clean_dept}'")
    if clean_imp:
        summary_parts.append(f"impact='{clean_imp}'")

    log_note = clean_text(notes) or f"Classification approved: {', '.join(summary_parts)}."

    success = db.apply_incident_classification_with_audit(
        incident_id=incident_id,
        category=clean_cat,
        severity=clean_sev,
        department=clean_dept,
        impact=clean_imp,
        actor=safe_actor,
        notes=log_note,
        db_path=db_path,
    )

    return success


def reject_incident_classification(
    incident_id: int,
    proposed_category: Optional[str] = None,
    proposed_severity: Optional[str] = None,
    proposed_department: Optional[str] = None,
    proposed_impact: Optional[str] = None,
    actor: Any = "Operator",
    reason: Optional[str] = None,
    db_path: Optional[Union[Path, str]] = None,
) -> bool:
    """Record a CLASSIFICATION_REJECTED audit event without modifying incident data.

    Preserves the incident's current state while maintaining an immutable audit
    trail of the human governance decision.

    Validates:
    - Incident must exist.
    - Records 'CLASSIFICATION_REJECTED' with safe actor and structured notes.
    - Zero updates are made to the incidents table.
    """
    if not isinstance(incident_id, int) or incident_id <= 0:
        raise ValueError(f"Invalid incident ID: {incident_id}. Must be a positive integer.")

    current = get_incident_by_id(incident_id, db_path=db_path)
    if current is None:
        raise ValueError(f"Incident with ID {incident_id} not found.")

    safe_actor = _normalize_actor(actor, default="Operator")

    clean_cat = clean_text(proposed_category) or None
    clean_sev = clean_text(proposed_severity) or None
    clean_dept = clean_text(proposed_department) or None
    clean_imp = clean_text(proposed_impact) or None
    clean_reason = clean_text(reason) or None

    parts = []
    if clean_cat:
        parts.append(f"category='{clean_cat}'")
    if clean_sev:
        parts.append(f"severity='{clean_sev}'")
    if clean_dept:
        parts.append(f"department='{clean_dept}'")
    if clean_imp:
        parts.append(f"impact='{clean_imp}'")

    summary_str = f" ({', '.join(parts)})" if parts else ""
    reason_str = f" Reason: {clean_reason}" if clean_reason else ""
    audit_notes = f"Operator rejected AI classification proposal{summary_str}.{reason_str}"

    log_id = db.add_audit_log(
        incident_id=incident_id,
        action="CLASSIFICATION_REJECTED",
        actor=safe_actor,
        notes=audit_notes,
        db_path=db_path,
    )
    return log_id is not None


# ==============================================================================
# 4. STATISTICS & AGGREGATIONS
# ==============================================================================

def get_incident_stats(
    db_path: Optional[Union[Path, str]] = None
) -> Dict[str, int]:
    """Retrieve live aggregated statistics for the dashboard cards.

    Returns:
        {
            "total": int,
            "open": int,
            "critical": int,
            "resolved": int
        }
    """
    return db.get_incident_stats(db_path=db_path)


# ==============================================================================
# 5. INVESTIGATION & AUDIT HISTORY
# ==============================================================================

def get_investigations_for_incident(
    incident_id: int,
    db_path: Optional[Union[Path, str]] = None
) -> List[Dict[str, Any]]:
    """Retrieve all investigation records for an incident, newest first."""
    if not isinstance(incident_id, int) or incident_id <= 0:
        raise ValueError(f"Invalid incident ID: {incident_id}. Must be a positive integer.")

    rows = db.get_investigations_for_incident(incident_id, db_path=db_path)
    return rows_to_dict_list(rows)


def get_audit_history_for_incident(
    incident_id: int,
    db_path: Optional[Union[Path, str]] = None
) -> List[Dict[str, Any]]:
    """Retrieve full audit history for an incident in chronological order."""
    if not isinstance(incident_id, int) or incident_id <= 0:
        raise ValueError(f"Invalid incident ID: {incident_id}. Must be a positive integer.")

    rows = db.get_audit_logs(incident_id, db_path=db_path)
    return rows_to_dict_list(rows)


def get_investigation_by_id(
    investigation_id: int,
    db_path: Optional[Union[Path, str]] = None,
) -> Optional[Dict[str, Any]]:
    """Retrieve an investigation by ID, deserialized into standard Python structures."""
    if not isinstance(investigation_id, int) or investigation_id <= 0:
        raise ValueError(f"Invalid investigation ID: {investigation_id}. Must be a positive integer.")

    row = db.get_investigation_by_id(investigation_id, db_path=db_path)
    if row is None:
        return None
    return deserialize_investigation(row)


def save_investigation_proposal(
    incident_id: int,
    investigation_data: Dict[str, Any],
    evidence_packets: Optional[List[Dict[str, Any]]] = None,
    actor: str = "System",
    db_path: Optional[Union[Path, str]] = None,
) -> int:
    """Validate, serialize provenance, and atomically persist an AI investigation proposal.

    Preserves all 9 chunk provenance attributes in investigations.evidence_citations JSON:
    chunk_id, document_id, document_name, page_number, chunk_index, rank, citation, citation_full, bm25_score.
    """
    if not isinstance(incident_id, int) or incident_id <= 0:
        raise ValueError(f"Invalid incident ID: {incident_id}. Must be a positive integer.")

    if not isinstance(investigation_data, dict):
        raise ValueError(f"investigation_data must be a dictionary, got {type(investigation_data).__name__}.")

    raw_ai_rec = investigation_data.get("ai_recommendation", "")
    ai_rec = raw_ai_rec.strip() if isinstance(raw_ai_rec, str) else str(raw_ai_rec).strip()
    if not ai_rec:
        raise ValueError("Investigation proposal missing required non-empty 'ai_recommendation'.")

    # Serialize confirmed_facts (JSON list of strings or empty list)
    facts = investigation_data.get("confirmed_facts", [])
    if isinstance(facts, list):
        clean_facts_list = [clean_text(f) for f in facts if isinstance(f, str) and clean_text(f)]
        confirmed_facts_str = json.dumps(clean_facts_list, ensure_ascii=False)
    elif isinstance(facts, str) and facts.strip():
        confirmed_facts_str = facts.strip()
    else:
        confirmed_facts_str = json.dumps([], ensure_ascii=False)

    # Serialize root_cause_hypotheses (JSON list of dicts)
    hypotheses = investigation_data.get("root_cause_hypotheses", [])
    if isinstance(hypotheses, list):
        root_cause_str = json.dumps(hypotheses, ensure_ascii=False)
    elif isinstance(hypotheses, str) and hypotheses.strip():
        root_cause_str = hypotheses.strip()
    else:
        root_cause_str = json.dumps([], ensure_ascii=False)

    # Serialize evidence_citations as structured bundle preserving provenance and assumptions
    citations = investigation_data.get("evidence_citations", [])
    if not isinstance(citations, list):
        citations = [str(citations)] if citations else []
    clean_citations = [clean_text(c) for c in citations if isinstance(c, str) and clean_text(c)]

    assumptions = investigation_data.get("assumptions", [])
    if not isinstance(assumptions, list):
        assumptions = []
    clean_assumptions = [clean_text(a) for a in assumptions if isinstance(a, str) and clean_text(a)]

    validation_summary = investigation_data.get("citation_validation_summary", {})
    if not isinstance(validation_summary, dict):
        validation_summary = {}

    # Extract provenance chunks from evidence_packets or investigation_data["retrieved_evidence"]
    raw_chunks = evidence_packets or investigation_data.get("retrieved_evidence", [])
    provenance_list = []
    if isinstance(raw_chunks, list):
        for chunk in raw_chunks:
            if isinstance(chunk, dict):
                provenance_list.append({
                    "chunk_id": chunk.get("chunk_id"),
                    "document_id": chunk.get("document_id"),
                    "document_name": chunk.get("document_name"),
                    "page_number": chunk.get("page_number", 1),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "rank": chunk.get("rank"),
                    "citation": chunk.get("citation"),
                    "citation_full": chunk.get("citation_full"),
                    "bm25_score": chunk.get("bm25_score"),
                    "relevance_score": chunk.get("relevance_score"),
                    "content": chunk.get("content", "")[:500] if chunk.get("content") else None,
                })

    evidence_bundle = {
        "citations": clean_citations,
        "assumptions": clean_assumptions,
        "provenance": provenance_list,
        "validation_summary": validation_summary,
        "mode": investigation_data.get("mode"),
        "is_fallback": bool(investigation_data.get("is_fallback") or investigation_data.get("mode") == "evidence_only_fallback"),
        "error_type": investigation_data.get("error_type"),
        "fallback_reason": investigation_data.get("fallback_reason"),
    }
    evidence_citations_str = json.dumps(evidence_bundle, ensure_ascii=False)

    # Extract confidence
    raw_conf = investigation_data.get("confidence")
    confidence_val = None
    if raw_conf is not None and not isinstance(raw_conf, bool) and isinstance(raw_conf, (int, float)):
        confidence_val = float(raw_conf)

    safe_actor = _normalize_actor(actor, default="System")

    return db.save_investigation_with_audit(
        incident_id=incident_id,
        ai_recommendation=ai_rec,
        confirmed_facts=confirmed_facts_str,
        root_cause_hypotheses=root_cause_str,
        evidence_citations=evidence_citations_str,
        confidence=confidence_val,
        human_decision="Pending",
        actor=safe_actor,
        db_path=db_path,
    )


def apply_investigation_decision(
    investigation_id: int,
    decision: str,
    final_action: Optional[str] = None,
    reviewer_notes: Optional[str] = None,
    actor: str = "Operator",
    is_operator_authored: Optional[bool] = None,
    audit_notes: Optional[str] = None,
    db_path: Optional[Union[Path, str]] = None,
) -> bool:
    """Validate and apply a human review decision to an investigation proposal atomically.

    Decisions:
    - 'Approved':
        - In fallback mode (operator-authored remediation): validates non-empty operator remediation,
          persists final_action, updates incident to 'Approved', and atomically records
          'RECOMMENDATION_APPROVED' with actor 'Operator' and explicit details indicating
          the approved remediation was operator-authored and not AI-generated.
        - In standard AI mode: approves AI recommendation as-is (action='RECOMMENDATION_APPROVED').
    - 'Edited': Operator modified recommendation (action='RECOMMENDATION_EDITED').
    - 'Rejected': Rejects recommendation; explicitly sets final_action = NULL and resets incident
      to 'Open' (action='RECOMMENDATION_REJECTED').

    Duplicate submissions for already-decided investigations are safely suppressed without
    creating duplicate audit records.
    """
    if not isinstance(investigation_id, int) or investigation_id <= 0:
        raise ValueError(f"Invalid investigation ID: {investigation_id}. Must be a positive integer.")

    clean_dec = clean_text(decision).title()
    if clean_dec not in {"Approved", "Edited", "Rejected"}:
        raise ValueError(
            f"Invalid investigation decision: '{decision}'. Must be one of 'Approved', 'Edited', 'Rejected'."
        )

    # Fetch investigation to verify existence and determine incident_id
    inv_row = db.get_investigation_by_id(investigation_id, db_path=db_path)
    if inv_row is None:
        raise ValueError(f"Investigation with ID {investigation_id} not found.")

    incident_id = inv_row["incident_id"]
    ai_rec_raw = inv_row["ai_recommendation"] or ""
    ev_cits_raw = inv_row["evidence_citations"] or ""

    is_fallback = (
        is_operator_authored is True
        or (
            is_operator_authored is not False
            and (
                "Automated AI diagnostic synthesis is currently unavailable" in ai_rec_raw
                or '"is_fallback": true' in ev_cits_raw.lower()
                or '"mode": "evidence_only_fallback"' in ev_cits_raw.lower()
            )
        )
    )

    clean_just = clean_text(reviewer_notes) or None

    if clean_dec == "Approved":
        human_decision = "Approved"
        action = "RECOMMENDATION_APPROVED"
        if is_fallback:
            clean_final = final_action.strip() if isinstance(final_action, str) else ""
            if not clean_final:
                raise ValueError("An operator remediation procedure is required before approval can be recorded in fallback mode.")
            resolved_final_action = clean_final
            if audit_notes and audit_notes.strip():
                final_audit_notes = audit_notes.strip()
            else:
                if clean_just:
                    final_audit_notes = (
                        f"Operator approved operator-authored remediation (offline fallback mode — not AI-generated) "
                        f"for Investigation #{investigation_id}. Justification: {clean_just}"
                    )
                else:
                    final_audit_notes = (
                        f"Operator approved operator-authored remediation (offline fallback mode — not AI-generated) "
                        f"for Investigation #{investigation_id}."
                    )
        else:
            resolved_final_action = final_action.strip() if (isinstance(final_action, str) and final_action.strip()) else inv_row["ai_recommendation"]
            if audit_notes and audit_notes.strip():
                final_audit_notes = audit_notes.strip()
            else:
                if clean_just:
                    final_audit_notes = f"Operator approved AI recommendation for Investigation #{investigation_id}. Justification: {clean_just}"
                else:
                    final_audit_notes = f"Operator approved AI recommendation for Investigation #{investigation_id}."

    elif clean_dec == "Edited":
        human_decision = "Approved"
        action = "RECOMMENDATION_EDITED"
        clean_final = final_action.strip() if isinstance(final_action, str) else ""
        if not clean_final:
            raise ValueError("final_action must be provided when editing recommendation.")
        resolved_final_action = clean_final
        if audit_notes and audit_notes.strip():
            final_audit_notes = audit_notes.strip()
        else:
            if clean_just:
                final_audit_notes = f"Operator edited and approved recommendation for Investigation #{investigation_id}. Justification: {clean_just}"
            else:
                final_audit_notes = f"Operator edited and approved recommendation for Investigation #{investigation_id}."

    else:  # Rejected
        human_decision = "Rejected"
        action = "RECOMMENDATION_REJECTED"
        resolved_final_action = None
        if audit_notes and audit_notes.strip():
            final_audit_notes = audit_notes.strip()
        else:
            if clean_just:
                final_audit_notes = f"Operator rejected recommendation for Investigation #{investigation_id}. Reason: {clean_just}"
            else:
                final_audit_notes = f"Operator rejected recommendation for Investigation #{investigation_id}."

    safe_actor = _normalize_actor(actor, default="Operator")

    return db.apply_investigation_decision_with_audit(
        investigation_id=investigation_id,
        incident_id=incident_id,
        human_decision=human_decision,
        final_action=resolved_final_action,
        reviewer_notes=reviewer_notes,
        actor=safe_actor,
        action=action,
        audit_notes=final_audit_notes,
        db_path=db_path,
    )


def deserialize_investigation(
    row_or_dict: Union[sqlite3.Row, Dict[str, Any]]
) -> Dict[str, Any]:
    """Safely unpack JSON-serialized fields in an investigation record into native Python structures.

    Handles:
    - confirmed_facts: JSON array or plain text lines.
    - root_cause_hypotheses: JSON array of dicts.
    - evidence_citations: Structured bundle dict containing citations, assumptions, provenance,
      and validation_summary, or legacy plain list / string.
    - Preserves backward compatibility with legacy database records.
    """
    if row_or_dict is None:
        return {}

    data = row_to_dict(row_or_dict) if isinstance(row_or_dict, sqlite3.Row) else dict(row_or_dict)

    # 1. Deserialize confirmed_facts
    raw_facts = data.get("confirmed_facts")
    clean_facts = []
    if isinstance(raw_facts, list):
        clean_facts = raw_facts
    elif isinstance(raw_facts, str) and raw_facts.strip():
        trimmed = raw_facts.strip()
        if trimmed.startswith("["):
            try:
                parsed = json.loads(trimmed)
                clean_facts = parsed if isinstance(parsed, list) else [str(parsed)]
            except Exception:
                clean_facts = [line.strip() for line in trimmed.splitlines() if line.strip()]
        else:
            clean_facts = [line.strip() for line in trimmed.splitlines() if line.strip()]

    # 2. Deserialize root_cause_hypotheses
    raw_hypo = data.get("root_cause_hypotheses")
    clean_hypo = []
    if isinstance(raw_hypo, list):
        clean_hypo = raw_hypo
    elif isinstance(raw_hypo, str) and raw_hypo.strip():
        trimmed = raw_hypo.strip()
        if trimmed.startswith("["):
            try:
                parsed = json.loads(trimmed)
                clean_hypo = parsed if isinstance(parsed, list) else []
            except Exception:
                clean_hypo = []

    # 3. Deserialize evidence_citations & unpack provenance / assumptions
    raw_cites = data.get("evidence_citations")
    citations_list: List[str] = []
    assumptions_list: List[str] = []
    provenance_list: List[Dict[str, Any]] = []
    validation_summary: Dict[str, Any] = {}
    mode_val: Optional[str] = None
    is_fallback_val: bool = False
    error_type_val: Optional[str] = None
    fallback_reason_val: Optional[str] = None

    if isinstance(raw_cites, dict):
        citations_list = raw_cites.get("citations", [])
        assumptions_list = raw_cites.get("assumptions", [])
        provenance_list = raw_cites.get("provenance", [])
        validation_summary = raw_cites.get("validation_summary", {})
        mode_val = raw_cites.get("mode")
        is_fallback_val = bool(raw_cites.get("is_fallback"))
        error_type_val = raw_cites.get("error_type")
        fallback_reason_val = raw_cites.get("fallback_reason")
    elif isinstance(raw_cites, list):
        citations_list = [str(c) for c in raw_cites]
    elif isinstance(raw_cites, str) and raw_cites.strip():
        trimmed = raw_cites.strip()
        if trimmed.startswith("{"):
            try:
                bundle = json.loads(trimmed)
                if isinstance(bundle, dict):
                    citations_list = bundle.get("citations", [])
                    assumptions_list = bundle.get("assumptions", [])
                    provenance_list = bundle.get("provenance", [])
                    validation_summary = bundle.get("validation_summary", {})
                    mode_val = bundle.get("mode")
                    is_fallback_val = bool(bundle.get("is_fallback"))
                    error_type_val = bundle.get("error_type")
                    fallback_reason_val = bundle.get("fallback_reason")
            except Exception:
                citations_list = [trimmed]
        elif trimmed.startswith("["):
            try:
                parsed = json.loads(trimmed)
                citations_list = parsed if isinstance(parsed, list) else [str(parsed)]
            except Exception:
                citations_list = [line.strip() for line in trimmed.splitlines() if line.strip()]
        else:
            citations_list = [line.strip() for line in trimmed.splitlines() if line.strip()]

    # Backward compatibility detection for legacy fallback proposals
    ai_rec_text = data.get("ai_recommendation", "") or ""
    if (
        is_fallback_val
        or mode_val == "evidence_only_fallback"
        or "Automated AI diagnostic synthesis is currently unavailable" in ai_rec_text
        or "EVIDENCE-ONLY FALLBACK" in ai_rec_text
    ):
        is_fallback_val = True
        mode_val = "evidence_only_fallback"
        if not error_type_val:
            error_type_val = "general"
        if data.get("confidence") == 0.0:
            data["confidence"] = None

    data["confirmed_facts"] = clean_facts
    data["root_cause_hypotheses"] = clean_hypo
    data["evidence_citations"] = citations_list
    data["assumptions"] = assumptions_list
    data["provenance"] = provenance_list
    data["citation_validation_summary"] = validation_summary
    data["mode"] = mode_val
    data["is_fallback"] = is_fallback_val
    data["error_type"] = error_type_val
    data["fallback_reason"] = fallback_reason_val

    return data
