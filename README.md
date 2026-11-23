# OpsPilot AI

### Intelligent Business Incident Investigation & Decision Support Platform

OpsPilot AI is an AI-assisted incident investigation and decision-support platform designed to help technical teams investigate operational incidents using knowledge-base retrieval, structured AI analysis, evidence citations, and human-approved remediation recommendations.

The platform combines Retrieval-Augmented Generation (RAG), Gemini-powered investigation, and an auditable human-in-the-loop approval workflow to make incident analysis more structured, transparent, and accountable.

> **Project status:** Functional prototype with tested investigation, retrieval, governance, and approval workflows. Uses synthetic demonstration data and does not connect to live enterprise infrastructure.

---

## Problem Statement

Operational incidents often require engineers to investigate symptoms across technical documentation, identify possible causes, and decide on safe remediation steps.

Manual investigation can be time-consuming and inconsistent. AI-generated recommendations also introduce risks when assumptions are presented as verified facts or when actions are executed without appropriate oversight.

OpsPilot AI addresses these challenges by combining knowledge-grounded investigation with explicit human review and an audit trail.

## Key Features

- **Incident Management:** Register and manage operational incidents.

- **AI-Assisted Classification:** Generate structured incident classifications.

- **Knowledge Base Management:** Ingest PDF, TXT, and Markdown documents.

- **RAG-Based Retrieval:** Retrieve relevant knowledge-base passages using SQLite FTS5 and BM25 ranking.

- **AI Investigation:** Use the Gemini API to generate structured investigation reports grounded in retrieved evidence.

- **Evidence Citations:** Track retrieved document references and validate citation grounding.

- **Uncertainty Awareness:** Separate confirmed facts, assumptions, and root-cause hypotheses.

- **Human-in-the-Loop Governance:** Allow operators to review, edit, approve, or reject recommendations.

- **Audit Trail:** Record investigation and decision-related events.

- **Safe Remediation Workflow:** Preserve original AI recommendations when operators customize proposed actions.

## Technology Stack

| Component | Technology |

|---|---|

| Frontend | Streamlit |

| Backend | Python |

| Database | SQLite |

| Retrieval | SQLite FTS5, BM25 |

| AI Model | Google Gemini API |

| API Communication | Python Requests |

| Testing | Pytest |

| Version Control | Git and GitHub |

## Architecture

```text

&#x20;                ┌──────────────────────┐

&#x20;                │   Streamlit UI       │

&#x20;                │ Incident Management  │

&#x20;                └──────────┬───────────┘

&#x20;                           │

&#x20;                ┌──────────▼───────────┐

&#x20;                │ Application Services │

&#x20;                │ Incident & Workflow  │

&#x20;                └──────────┬───────────┘

&#x20;                           │

&#x20;            ┌──────────────┴──────────────┐

&#x20;            │                             │

&#x20;   ┌────────▼────────┐          ┌─────────▼────────┐

&#x20;   │ Knowledge Base  │          │ AI Investigation │

&#x20;   │ SQLite FTS5     │          │ Gemini API       │

&#x20;   │ BM25 Retrieval  │          │ Structured Output│

&#x20;   └────────┬────────┘          └─────────┬────────┘

&#x20;            │                             │

&#x20;            └──────────────┬──────────────┘

&#x20;                           │

&#x20;                ┌──────────▼───────────┐

&#x20;                │ Evidence & Citation  │

&#x20;                │ Validation           │

&#x20;                └──────────┬───────────┘

&#x20;                           │

&#x20;                ┌──────────▼───────────┐

&#x20;                │ Human Review         │

&#x20;                │ Approve / Edit /     │

&#x20;                │ Reject               │

&#x20;                └──────────┬───────────┘

&#x20;                           │

&#x20;                ┌──────────▼───────────┐

&#x20;                │ Audit Trail &        │

&#x20;                │ Decision Persistence│

&#x20;                └──────────────────────┘

```

## How RAG Works

OpsPilot AI uses a Retrieval-Augmented Generation workflow to provide relevant technical documentation to the AI model during an investigation.

1. **Document ingestion:** Technical runbooks are uploaded to the knowledge base.

2. **Text extraction:** Text is extracted from supported document formats.

3. **Chunking:** Documents are divided into smaller passages.

4. **Indexing:** Passages are stored and indexed using SQLite FTS5.

5. **Retrieval:** Incident descriptions are used to retrieve relevant passages using BM25 lexical ranking.

6. **Context construction:** Retrieved passages and their source information are supplied to the Gemini model.

7. **Investigation:** Gemini generates a structured report using the supplied context.

8. **Citation validation:** The application checks whether cited sources correspond to retrieved evidence.

**Important:** The current implementation uses lexical retrieval with BM25. It does not use vector embeddings or vector-database-based semantic search.

## Human-in-the-Loop Governance

OpsPilot AI is designed to assist operators, not replace them.

The investigation workflow distinguishes between:

- **Confirmed facts:** Information verified through available evidence.

- **Assumptions:** Details that have not yet been independently verified.

- **Root-cause hypotheses:** Possible explanations that require further investigation.

- **Remediation recommendations:** Proposed steps that require human review.

Operators can approve, customize, or reject recommendations. Customized actions are stored separately from the original AI recommendation, and decision-related events are recorded in the audit trail.

The platform does not automatically execute production remediation.

## Installation and Setup

### Prerequisites

- Python 3.10 or later

- Git

- A Google Gemini API key

### 1. Clone the repository

```bash

git clone https://github.com/annmaria3/OpsPilot-AI.git

cd OpsPilot-AI

```

### 2. Create a virtual environment

Windows:

```cmd

python -m venv venv

venv\Scripts\activate

```

### 3. Install dependencies

```cmd

pip install -r requirements.txt

```

### 4. Configure environment variables

Copy the example configuration:

```cmd

copy .env.example .env

```

Open `.env` and replace the placeholder with your Gemini API key:

```env

GEMINI_API_KEY=your_actual_api_key

GEMINI_MODEL=gemini-3.5-flash-lite

```

Keep your actual `.env` file private. Never commit API keys to GitHub.

### 5. Run the application

```cmd

streamlit run app/main.py

```

The application will open in your browser.

## Running Tests

Run the automated test suite:

```cmd

python -m pytest -v

```

The project has currently passed **211 tests and 11 subtests**, covering application services, AI integration and fallback behavior, retrieval, knowledge-base ingestion, investigation persistence, governance, and UI workflows.

## Current Limitations

- The knowledge base uses synthetic demonstration runbooks, not private enterprise documentation.

- Retrieval uses BM25 lexical ranking rather than embedding-based semantic search.

- The platform does not currently integrate with live production logs, monitoring systems, payment gateways, or enterprise databases.

- Root-cause hypotheses require independent verification.

- AI confidence scores are model-generated assessments, not calibrated probabilities.

- Remediation is reviewed and approved by a human; no production action is automatically executed.

- Gemini availability and response quality depend on the external API.

## Future Improvements

- Semantic retrieval using embedding models.

- Integration with enterprise monitoring and logging platforms.

- Support for larger and continuously updated knowledge bases.

- Improved evaluation of retrieval relevance and answer quality.

- Role-based access control and enhanced authentication.

- Expanded audit reporting and operational analytics.

- Integration with controlled remediation systems with additional safeguards.

## Author

**Ann Maria Joby**

GitHub: [@annmaria3](https://github.com/annmaria3)

---

*OpsPilot AI — Grounded investigation. Transparent decisions. Human-controlled remediation.*
