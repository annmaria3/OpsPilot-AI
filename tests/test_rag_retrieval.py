"""OpsPilot AI - Knowledge Retrieval Engine Unit & Integration Tests (Step 7.3).

Comprehensive test suite verifying:
1. Keyword extraction from incident title.
2. Stop-word and conversational noise filtering.
3. Preservation of technical tokens: HTTP error codes, CamelCase, and acronyms.
4. Category taxonomy domain keyword enrichment.
5. Query construction and keyword bounding.
6. Uniform citation formatting (compact and full).
7. Top-K chunk retrieval and BM25 rank ordering.
8. Empty, whitespace, and symbol-only query resilience.
9. Retrieval for incident by dictionary input.
10. Retrieval for incident by database incident ID.
11. Graceful handling of missing or non-existent incident IDs.
12. Graceful handling of queries with zero matching documents.
13. Complete presence and correctness of all 11 source attribution fields.
14. Strict validation and clamping of retrieval limit (including boolean defense).
15. Resilience to missing, None, or malformed incident dictionary attributes.
16. Strict verification of local-only execution (zero Gemini or network dependencies).
17. Production database isolation check (data/opspilot.db remains 0 rows).

All tests execute against isolated temporary SQLite databases and temporary directories.
"""

import os
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import config
from app.database import db
from app.services import incident_service, rag_service


class TestKnowledgeRetrievalEngine(unittest.TestCase):
    """Test suite for Step 7.3 knowledge retrieval engine and query formulation."""

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
        """Create isolated temporary storage directory and temporary SQLite database."""
        self.temp_dir = tempfile.mkdtemp(prefix="opspilot_retrieval_test_")
        self.temp_storage = os.path.join(self.temp_dir, "storage")
        os.makedirs(self.temp_storage, exist_ok=True)

        self.db_temp_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_temp_file.close()
        self.test_db_path = self.db_temp_file.name

        # Initialize schema, indexes, FTS5 virtual table, and triggers
        db.initialize_database(db_path=self.test_db_path)

    def tearDown(self):
        """Clean up temporary database and filesystem directories."""
        if os.path.exists(self.test_db_path):
            try:
                os.remove(self.test_db_path)
            except OSError:
                pass

        if os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except OSError:
                pass

    # --------------------------------------------------------------------------
    # 1. Keyword Extraction from Title
    # --------------------------------------------------------------------------
    def test_01_keyword_extraction_from_title(self):
        """Verify extraction of core technical keywords from an incident title."""
        title = "PostgreSQL connection pool exhausted in production"
        keywords = rag_service.extract_search_keywords(title=title)

        # Title terms should be present; stop words like 'in' should be filtered
        self.assertIn("PostgreSQL", keywords)
        self.assertIn("connection", keywords)
        self.assertIn("pool", keywords)
        self.assertIn("exhausted", keywords)
        self.assertIn("production", keywords)
        self.assertNotIn("in", keywords)

    # --------------------------------------------------------------------------
    # 2. Stop-word and Conversational Noise Filtering
    # --------------------------------------------------------------------------
    def test_02_stopword_filtering(self):
        """Verify that conversational noise and ticket filler words are stripped."""
        title = "Please help urgently we are having an issue with the database"
        description = "Hi team, we noticed a critical error happening since this morning. Please investigate."

        keywords = rag_service.extract_search_keywords(title=title, description=description)

        # Stop words should be absent
        for word in ["please", "help", "urgently", "we", "are", "having", "an", "issue", "with", "the", "hi", "team"]:
            self.assertNotIn(word, [k.lower() for k in keywords])

        # Technical subject term should be present
        self.assertIn("database", [k.lower() for k in keywords])

    # --------------------------------------------------------------------------
    # 3. Technical Token Preservation
    # --------------------------------------------------------------------------
    def test_03_technical_token_preservation(self):
        """Verify error codes (504), CamelCase (DeadlockDetected), and acronyms (OOM, K8S) are kept."""
        description = (
            "Service ingress-nginx returned 504 Gateway Timeout. "
            "Pod threw NullPointerException and OOMKilled in K8S cluster. "
            "Database showed DeadlockDetected on pg_stat_activity."
        )

        keywords = rag_service.extract_search_keywords(title="Ingress failure", description=description)

        self.assertIn("504", keywords)
        self.assertIn("NullPointerException", keywords)
        self.assertIn("OOMKilled", keywords)
        self.assertIn("K8S", keywords)
        self.assertIn("DeadlockDetected", keywords)
        self.assertIn("pg_stat_activity", keywords)

    # --------------------------------------------------------------------------
    # 4. Category Taxonomy Domain Keyword Enrichment
    # --------------------------------------------------------------------------
    def test_04_category_domain_enrichment(self):
        """Verify category domain taxonomy words augment the search query."""
        keywords = rag_service.extract_search_keywords(
            title="Payment gateway timeout",
            category="Database Connectivity / Query Failure",
        )

        self.assertIn("Payment", keywords)
        self.assertIn("gateway", keywords)
        self.assertIn("timeout", keywords)
        self.assertIn("Database", keywords)
        self.assertIn("Connectivity", keywords)
        self.assertIn("Query", keywords)
        self.assertIn("Failure", keywords)

    # --------------------------------------------------------------------------
    # 5. Query Construction and Keyword Bounding
    # --------------------------------------------------------------------------
    def test_05_query_construction_bounded(self):
        """Verify generated search query is bounded to top 15 terms without runaway length."""
        long_title = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
        long_desc = "lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega"

        query = rag_service.build_incident_search_query(title=long_title, description=long_desc)
        query_words = query.split()

        self.assertLessEqual(len(query_words), 15)
        self.assertGreaterEqual(len(query_words), 10)
        # First words should match the high-priority title terms
        self.assertEqual(query_words[0], "alpha")
        self.assertEqual(query_words[1], "beta")

    # --------------------------------------------------------------------------
    # 6. Uniform Source Citation Formatting
    # --------------------------------------------------------------------------
    def test_06_format_source_citation_full_and_compact(self):
        """Verify formatting of source citations in both compact and full representations."""
        # Full citation
        full = rag_service.format_source_citation(
            document_name="PostgreSQL_Runbook.pdf",
            page_number=2,
            chunk_index=3,
            compact=False,
        )
        self.assertEqual(full, "Runbook: PostgreSQL_Runbook.pdf (Page 2, Chunk 3)")

        # Compact citation
        compact = rag_service.format_source_citation(
            document_name="PostgreSQL_Runbook.pdf",
            page_number=2,
            chunk_index=3,
            compact=True,
        )
        self.assertEqual(compact, "[PostgreSQL_Runbook.pdf, p.2 #3]")

        # Fallback values on missing/invalid input
        fallback = rag_service.format_source_citation(document_name="", page_number=-1, chunk_index=-5)
        self.assertEqual(fallback, "Runbook: Unknown Document (Page 1, Chunk 0)")

    # --------------------------------------------------------------------------
    # 7. Top-K Chunk Retrieval and BM25 Rank Ordering
    # --------------------------------------------------------------------------
    def test_07_retrieve_knowledge_chunks_bm25_ordering(self):
        """Verify retrieved chunks are ordered by BM25 relevance with sequential ranks."""
        # Ingest test runbook with 2 distinct chunks
        doc_id = db.create_knowledge_document(
            document_name="Kafka_Diagnostics.txt",
            file_path="/docs/kafka.txt",
            db_path=self.test_db_path,
        )

        db.insert_knowledge_chunks(
            doc_id,
            [
                {
                    "page_number": 1,
                    "chunk_index": 0,
                    "content": "Kafka consumer group rebalance storm. Consumer rebalance timeout and partition lag.",
                },
                {
                    "page_number": 2,
                    "chunk_index": 1,
                    "content": "General introduction to streaming message broker architectures.",
                },
                {
                    "page_number": 3,
                    "chunk_index": 2,
                    "content": "Kafka partition lag metrics in Datadog and consumer rebalance alerts.",
                },
            ],
            db_path=self.test_db_path,
        )

        # Ingest background noise to give proper BM25 IDF weights
        noise_doc = db.create_knowledge_document("Noise.txt", "/noise.txt", db_path=self.test_db_path)
        for i in range(5):
            db.insert_knowledge_chunks(noise_doc, [{"page_number": 1, "chunk_index": i, "content": f"Unrelated server log {i}"}], db_path=self.test_db_path)

        packets = rag_service.retrieve_knowledge_chunks(
            query="Kafka consumer rebalance lag",
            limit=5,
            db_path=self.test_db_path,
        )

        self.assertGreaterEqual(len(packets), 2)

        # Ranks must be sequential 1, 2, ...
        for idx, p in enumerate(packets):
            self.assertEqual(p["rank"], idx + 1)

        # Most relevant chunk (highest frequency of query terms) should be rank 1
        top_packet = packets[0]
        self.assertEqual(top_packet["document_name"], "Kafka_Diagnostics.txt")
        self.assertIn("Kafka consumer group rebalance storm", top_packet["content"])

        # Raw BM25 score must be negative
        self.assertLess(top_packet["bm25_score"], 0.0)
        # relevance_score must be positive absolute magnitude
        self.assertEqual(top_packet["relevance_score"], round(abs(top_packet["bm25_score"]), 4))

    # --------------------------------------------------------------------------
    # 8. Empty, Whitespace, and Symbol-Only Query Handling
    # --------------------------------------------------------------------------
    def test_08_retrieve_knowledge_chunks_empty_or_symbol_query(self):
        """Verify empty, whitespace, symbol-only, or None queries return empty list safely."""
        for invalid_q in ["", "   ", "\t\n", None, "??? !!! @@@", "$$$ %%% ^^^"]:
            results = rag_service.retrieve_knowledge_chunks(query=invalid_q, db_path=self.test_db_path)
            self.assertEqual(results, [])

    # --------------------------------------------------------------------------
    # 9. Retrieval for Incident by Dictionary Input
    # --------------------------------------------------------------------------
    def test_09_retrieve_knowledge_for_incident_by_dict(self):
        """Verify retrieval given an incident dictionary returns full bundle with citations."""
        # Seed knowledge base with matching runbook
        rag_service.ingest_document(
            file_name="Redis_Failover_Guide.txt",
            file_bytes=b"Redis sentinel master failover procedure. When sentinel triggers failover, check replication status.",
            storage_dir=self.temp_storage,
            db_path=self.test_db_path,
        )

        incident_dict = {
            "id": 42,
            "title": "Redis Sentinel Master Failover Alert",
            "description": "Production Redis sentinel triggered a master failover. Replication lag spiking.",
            "category": "Infrastructure / Server Downtime",
        }

        bundle = rag_service.retrieve_knowledge_for_incident(
            incident=incident_dict,
            limit=5,
            db_path=self.test_db_path,
        )

        self.assertIsInstance(bundle, dict)
        self.assertEqual(bundle["incident_id"], 42)
        self.assertEqual(bundle["incident_title"], "Redis Sentinel Master Failover Alert")
        self.assertGreater(bundle["total_chunks_retrieved"], 0)
        self.assertEqual(len(bundle["chunks"]), bundle["total_chunks_retrieved"])

        top_chunk = bundle["chunks"][0]
        self.assertEqual(top_chunk["document_name"], "Redis_Failover_Guide.txt")
        self.assertIn("sentinel triggers failover", top_chunk["content"])
        self.assertEqual(top_chunk["citation"], "[Redis_Failover_Guide.txt, p.1 #0]")

    # --------------------------------------------------------------------------
    # 10. Retrieval for Incident by Database Incident ID
    # --------------------------------------------------------------------------
    def test_10_retrieve_knowledge_for_incident_by_id(self):
        """Verify retrieval given an integer incident ID loaded from SQLite database."""
        # Create incident in database
        inc_id = db.create_incident(
            title="Database Connection Pool Starvation",
            description="Active connections maxed out. Error 504 on payment API.",
            category="Database Connectivity / Query Failure",
            db_path=self.test_db_path,
        )

        # Seed matching document
        rag_service.ingest_document(
            file_name="Postgres_Connection_Pool.txt",
            file_bytes=b"Diagnosing connection pool starvation. Check max_connections and pg_stat_activity.",
            storage_dir=self.temp_storage,
            db_path=self.test_db_path,
        )

        bundle = rag_service.retrieve_knowledge_for_incident(
            incident=inc_id,
            limit=5,
            db_path=self.test_db_path,
        )

        self.assertEqual(bundle["incident_id"], inc_id)
        self.assertEqual(bundle["incident_title"], "Database Connection Pool Starvation")
        self.assertGreaterEqual(bundle["total_chunks_retrieved"], 1)
        self.assertIn("Postgres_Connection_Pool.txt", bundle["chunks"][0]["document_name"])

    # --------------------------------------------------------------------------
    # 11. Graceful Handling of Missing or Non-Existent Incident ID
    # --------------------------------------------------------------------------
    def test_11_retrieve_knowledge_missing_or_invalid_incident_id(self):
        """Verify non-existent or negative incident ID returns clean empty bundle gracefully."""
        # Non-existent ID
        bundle1 = rag_service.retrieve_knowledge_for_incident(incident=999999, db_path=self.test_db_path)
        self.assertEqual(bundle1["total_chunks_retrieved"], 0)
        self.assertEqual(bundle1["chunks"], [])
        self.assertEqual(bundle1["incident_id"], 999999)

        # Negative ID
        bundle2 = rag_service.retrieve_knowledge_for_incident(incident=-1, db_path=self.test_db_path)
        self.assertEqual(bundle2["total_chunks_retrieved"], 0)
        self.assertEqual(bundle2["chunks"], [])

    # --------------------------------------------------------------------------
    # 12. Graceful Handling of Queries with No Matches
    # --------------------------------------------------------------------------
    def test_12_no_matches_returns_empty_gracefully(self):
        """Verify query with no matching terms in database returns empty list."""
        # Database has no docs matching this obscure keyword
        packets = rag_service.retrieve_knowledge_chunks(
            query="XylophoneZebraUnicornNonExistentTerm999",
            limit=5,
            db_path=self.test_db_path,
        )
        self.assertEqual(packets, [])

        bundle = rag_service.retrieve_knowledge_for_incident(
            incident={"id": 1, "title": "XylophoneZebraUnicornNonExistentTerm999"},
            db_path=self.test_db_path,
        )
        self.assertEqual(bundle["total_chunks_retrieved"], 0)
        self.assertEqual(bundle["chunks"], [])

    # --------------------------------------------------------------------------
    # 13. Complete Presence of All 11 Source Attribution Fields
    # --------------------------------------------------------------------------
    def test_13_source_attribution_fields_complete(self):
        """Verify that every returned evidence packet contains all 11 required fields."""
        rag_service.ingest_document(
            file_name="Attribution_Verification.txt",
            file_bytes=b"Attribution verification content for completeness test.",
            storage_dir=self.temp_storage,
            db_path=self.test_db_path,
        )

        packets = rag_service.retrieve_knowledge_chunks(
            query="Attribution verification",
            limit=5,
            db_path=self.test_db_path,
        )
        self.assertEqual(len(packets), 1)

        packet = packets[0]
        required_fields = [
            "chunk_id",
            "document_id",
            "document_name",
            "page_number",
            "chunk_index",
            "content",
            "bm25_score",
            "relevance_score",
            "rank",
            "citation",
            "citation_full",
        ]

        for field in required_fields:
            self.assertIn(field, packet, f"Missing required attribution field: {field}")
            self.assertIsNotNone(packet[field], f"Attribution field '{field}' must not be None")

        self.assertEqual(packet["rank"], 1)
        self.assertEqual(packet["document_name"], "Attribution_Verification.txt")
        self.assertEqual(packet["page_number"], 1)
        self.assertEqual(packet["chunk_index"], 0)

    # --------------------------------------------------------------------------
    # 14. Strict Limit Parameter Validation and Clamping
    # --------------------------------------------------------------------------
    def test_14_limit_parameter_enforcement(self):
        """Verify limit parameter clamping and prevention of boolean types acting as integers."""
        # Seed 10 chunks in doc
        doc_id = db.create_knowledge_document("Scale_Doc.txt", "/scale.txt", db_path=self.test_db_path)
        chunks = [{"page_number": 1, "chunk_index": i, "content": f"Scale cluster node item {i}."} for i in range(10)]
        db.insert_knowledge_chunks(doc_id, chunks, db_path=self.test_db_path)

        # Standard limit
        p3 = rag_service.retrieve_knowledge_chunks(query="cluster", limit=3, db_path=self.test_db_path)
        self.assertEqual(len(p3), 3)

        # Negative limit clamps to 1
        p_neg = rag_service.retrieve_knowledge_chunks(query="cluster", limit=-5, db_path=self.test_db_path)
        self.assertEqual(len(p_neg), 1)

        # Limit > MAX_RETRIEVAL_LIMIT (20) clamps to max
        p_max = rag_service.retrieve_knowledge_chunks(query="cluster", limit=100, db_path=self.test_db_path)
        self.assertLessEqual(len(p_max), config.MAX_RETRIEVAL_LIMIT)

        # Boolean True should NOT be treated as limit=1 (must fallback to DEFAULT_RETRIEVAL_LIMIT = 5)
        p_bool_true = rag_service.retrieve_knowledge_chunks(query="cluster", limit=True, db_path=self.test_db_path)
        self.assertEqual(len(p_bool_true), 5)

        # Boolean False should NOT be treated as limit=0 (must fallback to DEFAULT_RETRIEVAL_LIMIT = 5)
        p_bool_false = rag_service.retrieve_knowledge_chunks(query="cluster", limit=False, db_path=self.test_db_path)
        self.assertEqual(len(p_bool_false), 5)

        # Non-numeric string should fallback to default 5
        p_str = rag_service.retrieve_knowledge_chunks(query="cluster", limit="invalid_number", db_path=self.test_db_path)
        self.assertEqual(len(p_str), 5)

        # float('inf'), float('-inf'), float('nan') must safely fallback to default 5
        p_inf = rag_service.retrieve_knowledge_chunks(query="cluster", limit=float('inf'), db_path=self.test_db_path)
        self.assertEqual(len(p_inf), 5)

        p_neg_inf = rag_service.retrieve_knowledge_chunks(query="cluster", limit=float('-inf'), db_path=self.test_db_path)
        self.assertEqual(len(p_neg_inf), 5)

        p_nan = rag_service.retrieve_knowledge_chunks(query="cluster", limit=float('nan'), db_path=self.test_db_path)
        self.assertEqual(len(p_nan), 5)

        # Verify retrieve_knowledge_for_incident inherits identical safe limit behavior
        b_inf = rag_service.retrieve_knowledge_for_incident(
            incident={"title": "cluster scaling"},
            limit=float('inf'),
            db_path=self.test_db_path,
        )
        self.assertEqual(len(b_inf["chunks"]), 5)

        b_neginf = rag_service.retrieve_knowledge_for_incident(
            incident={"title": "cluster scaling"},
            limit=float('-inf'),
            db_path=self.test_db_path,
        )
        self.assertEqual(len(b_neginf["chunks"]), 5)

        b_nan = rag_service.retrieve_knowledge_for_incident(
            incident={"title": "cluster scaling"},
            limit=float('nan'),
            db_path=self.test_db_path,
        )
        self.assertEqual(len(b_nan["chunks"]), 5)

    # --------------------------------------------------------------------------
    # 15. Resilience to Missing, None, or Malformed Incident Dictionary Attributes
    # --------------------------------------------------------------------------
    def test_15_incident_dictionary_missing_or_malformed_fields(self):
        """Verify missing title, None description, or non-dict types are handled safely."""
        # Missing title
        b1 = rag_service.retrieve_knowledge_for_incident(
            incident={"description": "Some description"},
            db_path=self.test_db_path,
        )
        self.assertIsInstance(b1, dict)
        self.assertEqual(b1["incident_title"], "")

        # Non-string fields
        b2 = rag_service.retrieve_knowledge_for_incident(
            incident={"id": 10, "title": 12345, "description": ["malformed", "list"]},
            db_path=self.test_db_path,
        )
        self.assertIsInstance(b2, dict)
        self.assertEqual(b2["incident_id"], 10)

        # Non-dict and non-int input (e.g. string or None)
        b3 = rag_service.retrieve_knowledge_for_incident(incident=None, db_path=self.test_db_path)
        self.assertEqual(b3["total_chunks_retrieved"], 0)
        self.assertEqual(b3["chunks"], [])

    # --------------------------------------------------------------------------
    # 16. Local-Only Execution (Zero Gemini or Network Dependency)
    # --------------------------------------------------------------------------
    def test_16_retrieval_independent_of_gemini(self):
        """Verify retrieval executes purely locally without any Gemini API or network calls."""
        # Ingest document
        rag_service.ingest_document(
            file_name="Local_Only.txt",
            file_bytes=b"Offline deterministic runbook procedure for local execution.",
            storage_dir=self.temp_storage,
            db_path=self.test_db_path,
        )

        # Mock requests.post to assert no HTTP requests are made
        with patch("requests.post") as mock_post:
            with patch("requests.get") as mock_get:
                bundle = rag_service.retrieve_knowledge_for_incident(
                    incident={"id": 1, "title": "Offline deterministic runbook"},
                    db_path=self.test_db_path,
                )
                self.assertGreaterEqual(bundle["total_chunks_retrieved"], 1)
                mock_post.assert_not_called()
                mock_get.assert_not_called()

    # --------------------------------------------------------------------------
    # 17. Production Database Isolation
    # --------------------------------------------------------------------------
    def test_17_production_database_isolation(self):
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
