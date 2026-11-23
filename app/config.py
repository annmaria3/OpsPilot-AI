"""OpsPilot AI - Central Application Configuration.

This module defines project-wide constants, filesystem paths, environment
variable loaders, and standard domain taxonomies (categories, severities,
departments, statuses).

No third-party SDKs with native dependencies (like Pydantic or Google GenAI)
are imported here to ensure compatibility and stability.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ==============================================================================
# 1. FILESYSTEM PATHS
# ==============================================================================

# Project root directory (OpsPilot/)
# __file__ is OpsPilot/app/config.py -> parent is OpsPilot/app -> parent.parent is OpsPilot
BASE_DIR = Path(__file__).resolve().parent.parent

# Core data directories
DATA_DIR = BASE_DIR / "data"
DATABASE_PATH = DATA_DIR / "opspilot.db"
KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base"
REPORTS_DIR = BASE_DIR / "reports"

# Ensure core directories exist on disk
DATA_DIR.mkdir(parents=True, exist_ok=True)
KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 2. ENVIRONMENT VARIABLES & AI SETTINGS
# ==============================================================================

# Load environment variables from the root .env file
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# Gemini REST API configuration
# Read securely from .env. Never print or log secret keys directly.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash").strip()
GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


def is_gemini_configured() -> bool:
    """Return True if a Gemini API key is present, False otherwise."""
    return bool(GEMINI_API_KEY)


# ==============================================================================
# 3. DOMAIN TAXONOMIES & CONSTANTS
# ==============================================================================

# Standard incident categories
INCIDENT_CATEGORIES = [
    "Payment / Transaction Failure",
    "Authentication / Access Issue",
    "Infrastructure / Server Downtime",
    "Database Connectivity / Query Failure",
    "Application Performance & Latency",
    "Data Inconsistency / Sync Issue",
    "Other",
]

# Severity levels in increasing order of urgency
SEVERITY_LEVELS = [
    "Low",
    "Medium",
    "High",
    "Critical",
]

# Departments that can be affected or own the incident
DEPARTMENTS = [
    "Payments & Billing",
    "IT & Security",
    "Customer Operations",
    "Engineering & DevOps",
    "Data Platform",
    "General Operations",
]

# Lifecycle statuses for incident tracking
INCIDENT_STATUSES = [
    "Open",
    "Investigating",
    "Pending Approval",
    "Approved",
    "Resolved",
    "Closed",
]

# Audit log action identifiers for tracking human and AI activities
AUDIT_ACTIONS = [
    "INCIDENT_CREATED",
    "AI_CLASSIFIED",
    "CLASSIFICATION_APPROVED",
    "CLASSIFICATION_REJECTED",
    "INVESTIGATION_RUN",
    "RECOMMENDATION_APPROVED",
    "RECOMMENDATION_REJECTED",
    "RECOMMENDATION_EDITED",
    "INCIDENT_ESCALATED",
    "STATUS_UPDATED",
    "GOVERNANCE_AUDIT_CORRECTION",
]


# ==============================================================================
# 4. KNOWLEDGE BASE & DOCUMENT INGESTION CONSTANTS
# ==============================================================================

# Supported document extensions for ingestion
SUPPORTED_DOCUMENT_EXTENSIONS = {".pdf", ".txt", ".md"}

# Maximum document file size limit: 10 MB
MAX_DOCUMENT_SIZE_BYTES = 10 * 1024 * 1024

# Text chunking defaults (characters)
DEFAULT_CHUNK_SIZE = 600
DEFAULT_CHUNK_OVERLAP = 100

# Knowledge retrieval defaults
DEFAULT_RETRIEVAL_LIMIT = 5
MAX_RETRIEVAL_LIMIT = 20
MIN_SEARCH_KEYWORD_LENGTH = 2


# ==============================================================================
# 5. AI INVESTIGATION CONSTANTS
# ==============================================================================

# Network read timeout in seconds for investigation generation
INVESTIGATION_TIMEOUT_SECONDS = 45.0

# Maximum number of knowledge chunks passed into investigation prompt
MAX_INVESTIGATION_CHUNKS = 5

# Generation parameters for investigation
INVESTIGATION_MODEL_TEMPERATURE = 0.1
INVESTIGATION_MAX_OUTPUT_TOKENS = 1500

