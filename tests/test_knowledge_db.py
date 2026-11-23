"""OpsPilot AI - Knowledge Base Database Layer & SQLite FTS5 Tests (Step 7.2A & 7.2A-R).

Comprehensive test suite covering:
1. Safe initialization on fresh DB (tables, virtual table, index, triggers).
2. Repeated initialization idempotence.
3. Non-destructive migration of existing database schema.
4. Preservation of existing records during migration.
5. Document creation and retrieval.
6. Content hash lookup.
7. Chunk insertion and retrieval.
8. Accurate document chunk_count updates.
9. SQLite FTS5 keyword search.
10. BM25 result ordering (most relevant first).
11. Empty search handling (safe handling of empty strings, symbols, None).
12. Updating chunk synchronizes FTS index via trigger.
13. Deleting document cascades deletion to chunks.
14. FTS cleanup after document deletion via trigger.
15. Missing document handling.
16. Invalid input handling.
17. Transaction rollback on chunk batch failure.
18. Document renaming updates search results (live document name freshness).
19. Boolean keywords (AND, OR, NOT, NEAR) do not crash search.
20. Special characters and operational error message punctuation safety.
21. Unicode search behavior handled safely.
22. BM25 score semantics: negative raw score, ordering, and relevance magnitude.
23. Production database isolation check.

All tests operate on isolated temporary SQLite databases.
"""


import os
import sqlite3
import tempfile
import unittest

from app import config
from app.database import db


class TestKnowledgeDatabase(unittest.TestCase):
    """Test suite for Knowledge Base relational and FTS5 operations."""

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

        # Initialize schema, triggers, and foreign keys in isolated test DB
        db.initialize_database(db_path=self.test_db_path)

    def tearDown(self):
        """Clean up temporary test database file."""
        if os.path.exists(self.test_db_path):
            try:
                os.remove(self.test_db_path)
            except OSError:
                pass

    # --------------------------------------------------------------------------
    # 1. Safe Initialization on Fresh DB
    # --------------------------------------------------------------------------
    def test_01_safe_initialization_on_fresh_db(self):
        """Verify that initialize_database creates all tables, FTS virtual table, index, and triggers."""
        conn = db.get_connection(self.test_db_path)
        try:
            cursor = conn.cursor()

            # Verify standard tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = {row[0] for row in cursor.fetchall()}
            expected_tables = {
                "incidents",
                "investigations",
                "knowledge_documents",
                "knowledge_chunks",
                "audit_logs",
                "knowledge_chunks_fts",
            }
            self.assertTrue(expected_tables.issubset(tables), f"Missing tables: {expected_tables - tables}")

            # Verify index on knowledge_chunks(document_id)
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_chunks_document_id';")
            self.assertIsNotNone(cursor.fetchone(), "Index idx_chunks_document_id not found.")

            # Verify triggers
            cursor.execute("SELECT name FROM sqlite_master WHERE type='trigger';")
            triggers = {row[0] for row in cursor.fetchall()}
            expected_triggers = {
                "sync_knowledge_chunk_insert",
                "sync_knowledge_chunk_update",
                "sync_knowledge_chunk_delete",
                "sync_knowledge_doc_delete_fts",
            }
            self.assertTrue(expected_triggers.issubset(triggers), f"Missing triggers: {expected_triggers - triggers}")
        finally:
            conn.close()

    # --------------------------------------------------------------------------
    # 2. Repeated Initialization Idempotence
    # --------------------------------------------------------------------------
    def test_02_repeated_initialization_idempotence(self):
        """Verify that calling initialize_database multiple times does not error or corrupt schema."""
        # Call it two more times
        db.initialize_database(db_path=self.test_db_path)
        db.initialize_database(db_path=self.test_db_path)

        # Confirm database is fully operational
        doc_id = db.create_knowledge_document(
            document_name="Runbook_Alpha.txt",
            file_path="/storage/docs/Runbook_Alpha.txt",
            file_size_bytes=1024,
            content_hash="abc123hash",
            document_type="TXT",
            db_path=self.test_db_path,
        )
        self.assertIsInstance(doc_id, int)
        self.assertGreater(doc_id, 0)

    # --------------------------------------------------------------------------
    # 3. Migration of Existing Database Schema
    # --------------------------------------------------------------------------
    def test_03_migration_of_existing_database_schema(self):
        """Verify that initialize_database safely adds missing columns to an older schema."""
        # Create a fresh temporary file with legacy knowledge_documents schema
        old_temp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        old_temp.close()
        old_db_path = old_temp.name

        try:
            conn = sqlite3.connect(old_db_path)
            conn.execute("""
                CREATE TABLE knowledge_documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_name TEXT NOT NULL UNIQUE,
                    file_path TEXT NOT NULL,
                    uploaded_at TEXT
                );
            """)
            conn.commit()
            conn.close()

            # Run initialize_database which contains safe migrations
            db.initialize_database(db_path=old_db_path)

            # Inspect columns
            conn = db.get_connection(old_db_path)
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(knowledge_documents);")
            columns = {row["name"] for row in cursor.fetchall()}
            conn.close()

            expected_columns = {
                "id",
                "document_name",
                "file_path",
                "file_size_bytes",
                "content_hash",
                "document_type",
                "chunk_count",
                "status",
                "uploaded_at",
            }
            self.assertTrue(expected_columns.issubset(columns), f"Missing migrated columns: {expected_columns - columns}")
        finally:
            if os.path.exists(old_db_path):
                os.remove(old_db_path)

    # --------------------------------------------------------------------------
    # 4. Preservation of Existing Records During Migration
    # --------------------------------------------------------------------------
    def test_04_preservation_of_existing_records_during_migration(self):
        """Verify that pre-existing document rows survive schema migration intact."""
        old_temp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        old_temp.close()
        old_db_path = old_temp.name

        try:
            conn = sqlite3.connect(old_db_path)
            conn.execute("""
                CREATE TABLE knowledge_documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_name TEXT NOT NULL UNIQUE,
                    file_path TEXT NOT NULL,
                    uploaded_at TEXT
                );
            """)
            conn.execute("""
                INSERT INTO knowledge_documents (document_name, file_path, uploaded_at)
                VALUES ('Legacy_Standard.pdf', '/docs/legacy.pdf', '2026-01-01 10:00:00');
            """)
            conn.commit()
            conn.close()

            # Migrate
            db.initialize_database(db_path=old_db_path)

            # Verify existing record preserved with new column defaults
            doc = db.get_knowledge_document_by_id(1, db_path=old_db_path)
            self.assertIsNotNone(doc)
            self.assertEqual(doc["document_name"], "Legacy_Standard.pdf")
            self.assertEqual(doc["file_path"], "/docs/legacy.pdf")
            self.assertEqual(doc["uploaded_at"], "2026-01-01 10:00:00")
            self.assertEqual(doc["chunk_count"], 0)
            self.assertEqual(doc["status"], "Indexed")
        finally:
            if os.path.exists(old_db_path):
                os.remove(old_db_path)

    # --------------------------------------------------------------------------
    # 5. Document Creation and Retrieval
    # --------------------------------------------------------------------------
    def test_05_document_creation_and_retrieval(self):
        """Verify create_knowledge_document inserts and retrieval functions return full metadata."""
        doc_id = db.create_knowledge_document(
            document_name="Network_Topology.pdf",
            file_path="/storage/Network_Topology.pdf",
            file_size_bytes=2048500,
            content_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            document_type="PDF",
            chunk_count=10,
            status="Indexed",
            db_path=self.test_db_path,
        )
        self.assertGreater(doc_id, 0)

        # Retrieve by ID
        doc = db.get_knowledge_document_by_id(doc_id, db_path=self.test_db_path)
        self.assertIsNotNone(doc)
        self.assertEqual(doc["document_name"], "Network_Topology.pdf")
        self.assertEqual(doc["file_path"], "/storage/Network_Topology.pdf")
        self.assertEqual(doc["file_size_bytes"], 2048500)
        self.assertEqual(doc["content_hash"], "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
        self.assertEqual(doc["document_type"], "PDF")
        self.assertEqual(doc["chunk_count"], 10)
        self.assertEqual(doc["status"], "Indexed")
        self.assertTrue(doc["uploaded_at"])

        # Retrieve all
        all_docs = db.get_all_knowledge_documents(db_path=self.test_db_path)
        self.assertEqual(len(all_docs), 1)
        self.assertEqual(all_docs[0]["id"], doc_id)

    # --------------------------------------------------------------------------
    # 6. Content Hash Lookup
    # --------------------------------------------------------------------------
    def test_06_content_hash_lookup(self):
        """Verify get_knowledge_document_by_hash finds document by hash or returns None."""
        test_hash = "abcdef1234567890deadbeef"
        doc_id = db.create_knowledge_document(
            document_name="Database_Runbook.md",
            file_path="/docs/db_runbook.md",
            content_hash=test_hash,
            document_type="MD",
            db_path=self.test_db_path,
        )

        match = db.get_knowledge_document_by_hash(test_hash, db_path=self.test_db_path)
        self.assertIsNotNone(match)
        self.assertEqual(match["id"], doc_id)
        self.assertEqual(match["document_name"], "Database_Runbook.md")

        # Non-matching hash
        self.assertIsNone(db.get_knowledge_document_by_hash("nonexistent_hash", db_path=self.test_db_path))
        self.assertIsNone(db.get_knowledge_document_by_hash("", db_path=self.test_db_path))
        self.assertIsNone(db.get_knowledge_document_by_hash(None, db_path=self.test_db_path))

    # --------------------------------------------------------------------------
    # 7. Chunk Insertion and Retrieval
    # --------------------------------------------------------------------------
    def test_07_chunk_insertion_and_retrieval(self):
        """Verify insert_knowledge_chunks inserts batch and get_chunks_for_document orders correctly."""
        doc_id = db.create_knowledge_document(
            document_name="Service_Catalog.txt",
            file_path="/docs/service_catalog.txt",
            db_path=self.test_db_path,
        )

        chunks_data = [
            {"page_number": 2, "chunk_index": 1, "content": "Payment API Gateway v2 endpoints."},
            {"page_number": 1, "chunk_index": 0, "content": "Overview of enterprise microservices architecture."},
            {"page_number": 1, "chunk_index": 1, "content": "Authentication and authorization protocols."},
        ]

        count = db.insert_knowledge_chunks(doc_id, chunks_data, db_path=self.test_db_path)
        self.assertEqual(count, 3)

        retrieved = db.get_chunks_for_document(doc_id, db_path=self.test_db_path)
        self.assertEqual(len(retrieved), 3)

        # Ordering check: page 1 index 0, page 1 index 1, page 2 index 1
        self.assertEqual(retrieved[0]["page_number"], 1)
        self.assertEqual(retrieved[0]["chunk_index"], 0)
        self.assertIn("Overview of enterprise", retrieved[0]["content"])

        self.assertEqual(retrieved[1]["page_number"], 1)
        self.assertEqual(retrieved[1]["chunk_index"], 1)
        self.assertIn("Authentication and authorization", retrieved[1]["content"])

        self.assertEqual(retrieved[2]["page_number"], 2)
        self.assertEqual(retrieved[2]["chunk_index"], 1)
        self.assertIn("Payment API Gateway", retrieved[2]["content"])

    # --------------------------------------------------------------------------
    # 8. Correct Chunk Count Updates
    # --------------------------------------------------------------------------
    def test_08_correct_chunk_count_updates(self):
        """Verify that inserting chunks updates the parent document's chunk_count column."""
        doc_id = db.create_knowledge_document(
            document_name="Incident_Playbook.pdf",
            file_path="/docs/playbook.pdf",
            db_path=self.test_db_path,
        )

        doc_before = db.get_knowledge_document_by_id(doc_id, db_path=self.test_db_path)
        self.assertEqual(doc_before["chunk_count"], 0)

        # Insert 3 chunks
        db.insert_knowledge_chunks(
            doc_id,
            [
                {"page_number": 1, "chunk_index": 0, "content": "Step 1: Check monitoring dashboard."},
                {"page_number": 1, "chunk_index": 1, "content": "Step 2: Alert on-call engineer."},
                {"page_number": 2, "chunk_index": 0, "content": "Step 3: Mitigate traffic if DDoS."},
            ],
            db_path=self.test_db_path,
        )
        doc_after_first = db.get_knowledge_document_by_id(doc_id, db_path=self.test_db_path)
        self.assertEqual(doc_after_first["chunk_count"], 3)

        # Insert 2 more chunks
        db.insert_knowledge_chunks(
            doc_id,
            [
                {"page_number": 2, "chunk_index": 1, "content": "Step 4: Conduct post-incident review."},
                {"page_number": 3, "chunk_index": 0, "content": "Step 5: File post-mortem report."},
            ],
            db_path=self.test_db_path,
        )
        doc_after_second = db.get_knowledge_document_by_id(doc_id, db_path=self.test_db_path)
        self.assertEqual(doc_after_second["chunk_count"], 5)

    # --------------------------------------------------------------------------
    # 9. FTS5 Keyword Search
    # --------------------------------------------------------------------------
    def test_09_fts5_keyword_search(self):
        """Verify FTS5 search matches domain keywords and returns source metadata."""
        doc_id = db.create_knowledge_document(
            document_name="Postgres_Runbook.txt",
            file_path="/docs/postgres.txt",
            db_path=self.test_db_path,
        )

        db.insert_knowledge_chunks(
            doc_id,
            [
                {
                    "page_number": 1,
                    "chunk_index": 0,
                    "content": "PostgreSQL connection pool exhaustion occurs when max_connections is reached.",
                },
                {
                    "page_number": 2,
                    "chunk_index": 0,
                    "content": "Kubernetes cluster memory leak causing OOMKilled errors across ingress nodes.",
                },
            ],
            db_path=self.test_db_path,
        )

        # Search for database terms
        results = db.search_knowledge_chunks_fts("connection pool exhaustion", limit=5, db_path=self.test_db_path)
        self.assertGreaterEqual(len(results), 1)
        top_match = results[0]
        self.assertEqual(top_match["document_id"], doc_id)
        self.assertEqual(top_match["document_name"], "Postgres_Runbook.txt")
        self.assertEqual(top_match["page_number"], 1)
        self.assertEqual(top_match["chunk_index"], 0)
        self.assertIn("max_connections", top_match["content"])
        self.assertIn("bm25_score", top_match)
        self.assertIn("relevance_score", top_match)

        # Search for Kubernetes terms
        k8s_results = db.search_knowledge_chunks_fts("kubernetes OOMKilled", limit=5, db_path=self.test_db_path)
        self.assertGreaterEqual(len(k8s_results), 1)
        self.assertEqual(k8s_results[0]["page_number"], 2)
        self.assertIn("ingress nodes", k8s_results[0]["content"])

    # --------------------------------------------------------------------------
    # 10. BM25 Result Ordering
    # --------------------------------------------------------------------------
    def test_10_bm25_result_ordering(self):
        """Verify BM25 ranking orders more relevant chunks before less relevant ones."""
        doc_id = db.create_knowledge_document(
            document_name="Redis_Handbook.txt",
            file_path="/docs/redis.txt",
            db_path=self.test_db_path,
        )

        # Chunk A has high term frequency for "timeout", Chunk B has a single casual mention
        db.insert_knowledge_chunks(
            doc_id,
            [
                {
                    "page_number": 1,
                    "chunk_index": 0,
                    "content": "General architecture notes for in-memory caching layers.",
                },
                {
                    "page_number": 2,
                    "chunk_index": 0,
                    "content": "Redis client timeout issues. Timeout handling timeout threshold and timeout retries.",
                },
                {
                    "page_number": 3,
                    "chunk_index": 0,
                    "content": "Redis memory eviction policies when maxmemory is hit, sometimes resulting in a timeout.",
                },
            ],
            db_path=self.test_db_path,
        )

        results = db.search_knowledge_chunks_fts("timeout", limit=5, db_path=self.test_db_path)
        self.assertGreaterEqual(len(results), 2)

        # Chunk with high frequency should rank first
        self.assertEqual(results[0]["page_number"], 2)
        self.assertIn("timeout threshold", results[0]["content"])

    # --------------------------------------------------------------------------
    # 11. Empty Search Handling
    # --------------------------------------------------------------------------
    def test_11_empty_search_handling(self):
        """Verify empty, whitespace, symbol-only, or None searches return [] without error."""
        queries = ["", "   ", "\t\n", None, "??? !!! @@@", "$$$ %%% ^^^", "   ***   ", "()[]{}"]
        for q in queries:
            results = db.search_knowledge_chunks_fts(q, db_path=self.test_db_path)
            self.assertEqual(results, [], f"Expected empty result for query {repr(q)}")

    # --------------------------------------------------------------------------
    # 12. Updating Chunk Synchronizes FTS
    # --------------------------------------------------------------------------
    def test_12_updating_chunk_synchronizes_fts(self):
        """Verify trigger sync_knowledge_chunk_update keeps FTS index synchronized on direct chunk update."""
        doc_id = db.create_knowledge_document(
            document_name="Queue_Service.txt",
            file_path="/docs/queue.txt",
            db_path=self.test_db_path,
        )

        db.insert_knowledge_chunks(
            doc_id,
            [{"page_number": 1, "chunk_index": 0, "content": "ActiveMQ message queue dispatch latency."}],
            db_path=self.test_db_path,
        )

        # Matches ActiveMQ, but not RabbitMQ
        self.assertEqual(len(db.search_knowledge_chunks_fts("ActiveMQ", db_path=self.test_db_path)), 1)
        self.assertEqual(len(db.search_knowledge_chunks_fts("RabbitMQ", db_path=self.test_db_path)), 0)

        # Directly update the chunk row in knowledge_chunks
        conn = db.get_connection(self.test_db_path)
        try:
            conn.execute("""
                UPDATE knowledge_chunks
                SET content = 'RabbitMQ cluster partition detected in production.'
                WHERE document_id = ?;
            """, (doc_id,))
            conn.commit()
        finally:
            conn.close()

        # FTS index should now match RabbitMQ and NOT ActiveMQ
        self.assertEqual(len(db.search_knowledge_chunks_fts("RabbitMQ", db_path=self.test_db_path)), 1)
        self.assertEqual(len(db.search_knowledge_chunks_fts("ActiveMQ", db_path=self.test_db_path)), 0)

    # --------------------------------------------------------------------------
    # 13. Deleting Document Cascades Chunk Deletion
    # --------------------------------------------------------------------------
    def test_13_deleting_document_cascades_chunk_deletion(self):
        """Verify delete_knowledge_document deletes the document and cascades to chunks."""
        doc_id = db.create_knowledge_document(
            document_name="Ephemeral_Doc.txt",
            file_path="/docs/ephemeral.txt",
            db_path=self.test_db_path,
        )

        db.insert_knowledge_chunks(
            doc_id,
            [
                {"page_number": 1, "chunk_index": 0, "content": "Ephemeral content chunk 1."},
                {"page_number": 1, "chunk_index": 1, "content": "Ephemeral content chunk 2."},
            ],
            db_path=self.test_db_path,
        )

        self.assertEqual(len(db.get_chunks_for_document(doc_id, db_path=self.test_db_path)), 2)

        deleted = db.delete_knowledge_document(doc_id, db_path=self.test_db_path)
        self.assertTrue(deleted)

        # Verify document is gone
        self.assertIsNone(db.get_knowledge_document_by_id(doc_id, db_path=self.test_db_path))

        # Verify chunks are gone
        self.assertEqual(len(db.get_chunks_for_document(doc_id, db_path=self.test_db_path)), 0)

    # --------------------------------------------------------------------------
    # 14. FTS Cleanup After Document Deletion
    # --------------------------------------------------------------------------
    def test_14_fts_cleanup_after_document_deletion(self):
        """Verify deleting a document cleans up all associated FTS entries via trigger."""
        doc_id = db.create_knowledge_document(
            document_name="Zookeeper_Doc.txt",
            file_path="/docs/zookeeper.txt",
            db_path=self.test_db_path,
        )

        db.insert_knowledge_chunks(
            doc_id,
            [{"page_number": 1, "chunk_index": 0, "content": "ZookeeperElectionQuorumLossException occurred."}],
            db_path=self.test_db_path,
        )

        # Verify indexed
        self.assertEqual(len(db.search_knowledge_chunks_fts("ZookeeperElectionQuorumLossException", db_path=self.test_db_path)), 1)

        # Delete document
        db.delete_knowledge_document(doc_id, db_path=self.test_db_path)

        # Verify FTS index cleaned up
        self.assertEqual(len(db.search_knowledge_chunks_fts("ZookeeperElectionQuorumLossException", db_path=self.test_db_path)), 0)

        # Direct check on FTS table
        conn = db.get_connection(self.test_db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM knowledge_chunks_fts WHERE document_id = ?;", (doc_id,))
            self.assertEqual(cursor.fetchone()[0], 0)
        finally:
            conn.close()

    # --------------------------------------------------------------------------
    # 15. Missing Document Handling
    # --------------------------------------------------------------------------
    def test_15_missing_document_handling(self):
        """Verify operations on non-existent document IDs return None/empty or raise ValueError."""
        non_existent_id = 999999

        # Retrieval of missing doc
        self.assertIsNone(db.get_knowledge_document_by_id(non_existent_id, db_path=self.test_db_path))
        self.assertEqual(db.get_chunks_for_document(non_existent_id, db_path=self.test_db_path), [])

        # Inserting chunks for missing doc raises ValueError
        with self.assertRaises(ValueError):
            db.insert_knowledge_chunks(
                non_existent_id,
                [{"page_number": 1, "chunk_index": 0, "content": "Orphan content."}],
                db_path=self.test_db_path,
            )

        # Deleting missing doc returns False
        self.assertFalse(db.delete_knowledge_document(non_existent_id, db_path=self.test_db_path))

    # --------------------------------------------------------------------------
    # 16. Invalid Input Handling
    # --------------------------------------------------------------------------
    def test_16_invalid_input_handling(self):
        """Verify invalid document names, paths, or chunk data raise appropriate errors."""
        # Empty document name
        with self.assertRaises(ValueError):
            db.create_knowledge_document("", "/valid/path.txt", db_path=self.test_db_path)

        # Empty file path
        with self.assertRaises(ValueError):
            db.create_knowledge_document("Valid_Name.txt", "", db_path=self.test_db_path)

        # Non-integer / invalid IDs for deletion
        self.assertFalse(db.delete_knowledge_document(-1, db_path=self.test_db_path))
        self.assertFalse(db.delete_knowledge_document("invalid_id", db_path=self.test_db_path))

        # Invalid IDs for retrieval
        self.assertIsNone(db.get_knowledge_document_by_id(-1, db_path=self.test_db_path))
        self.assertEqual(db.get_chunks_for_document(-1, db_path=self.test_db_path), [])

        # Create valid doc for chunk validation
        doc_id = db.create_knowledge_document("Valid.txt", "/valid.txt", db_path=self.test_db_path)

        # Empty content in chunk
        with self.assertRaises(ValueError):
            db.insert_knowledge_chunks(doc_id, [{"page_number": 1, "content": ""}], db_path=self.test_db_path)

        # Non-dictionary chunk element
        with self.assertRaises(ValueError):
            db.insert_knowledge_chunks(doc_id, ["invalid_chunk_type"], db_path=self.test_db_path)

        # Empty chunk list returns 0 without error
        self.assertEqual(db.insert_knowledge_chunks(doc_id, [], db_path=self.test_db_path), 0)

    # --------------------------------------------------------------------------
    # 17. Transaction Rollback on Chunk Failure
    # --------------------------------------------------------------------------
    def test_17_transaction_rollback_on_chunk_failure(self):
        """Verify that if any chunk in a batch fails, the entire transaction rolls back."""
        doc_id = db.create_knowledge_document(
            document_name="Batch_Rollback_Test.txt",
            file_path="/docs/rollback.txt",
            db_path=self.test_db_path,
        )

        # Batch where chunk 2 has empty content
        malformed_batch = [
            {"page_number": 1, "chunk_index": 0, "content": "Valid chunk 1 content."},
            {"page_number": 1, "chunk_index": 1, "content": "   "},  # Malformed
            {"page_number": 1, "chunk_index": 2, "content": "Valid chunk 3 content."},
        ]

        with self.assertRaises(ValueError):
            db.insert_knowledge_chunks(doc_id, malformed_batch, db_path=self.test_db_path)

        # Confirm 0 chunks were inserted
        chunks = db.get_chunks_for_document(doc_id, db_path=self.test_db_path)
        self.assertEqual(len(chunks), 0)

        # Confirm chunk_count remains 0
        doc = db.get_knowledge_document_by_id(doc_id, db_path=self.test_db_path)
        self.assertEqual(doc["chunk_count"], 0)

    # --------------------------------------------------------------------------
    # 18. Document Renaming Updates Search Results
    # --------------------------------------------------------------------------
    def test_18_document_renaming_updates_search_results(self):
        """Verify that renaming a document updates its displayed name in FTS search results."""
        doc_id = db.create_knowledge_document(
            document_name="Network_Topology_v1.pdf",
            file_path="/storage/Network_Topology_v1.pdf",
            db_path=self.test_db_path,
        )

        db.insert_knowledge_chunks(
            doc_id,
            [{"page_number": 1, "chunk_index": 0, "content": "Core router BGP peering failure on interface ge-0/0/1."}],
            db_path=self.test_db_path,
        )

        # Pre-rename search returns original name
        results_before = db.search_knowledge_chunks_fts("router BGP", db_path=self.test_db_path)
        self.assertEqual(len(results_before), 1)
        self.assertEqual(results_before[0]["document_name"], "Network_Topology_v1.pdf")

        # Rename document in relational table
        conn = db.get_connection(self.test_db_path)
        try:
            conn.execute(
                "UPDATE knowledge_documents SET document_name = ? WHERE id = ?;",
                ("Network_Topology_v2_Updated.pdf", doc_id),
            )
            conn.commit()
        finally:
            conn.close()

        # Post-rename search must return updated document name
        results_after = db.search_knowledge_chunks_fts("router BGP", db_path=self.test_db_path)
        self.assertEqual(len(results_after), 1)
        self.assertEqual(results_after[0]["document_name"], "Network_Topology_v2_Updated.pdf")

    # --------------------------------------------------------------------------
    # 19. Boolean Keywords Do Not Crash Search
    # --------------------------------------------------------------------------
    def test_19_boolean_keywords_do_not_crash_search(self):
        """Verify that boolean-looking terms (AND, OR, NOT, NEAR) do not crash SQLite FTS5."""
        doc_id = db.create_knowledge_document(
            document_name="Logic_Guide.txt",
            file_path="/docs/logic.txt",
            db_path=self.test_db_path,
        )

        db.insert_knowledge_chunks(
            doc_id,
            [
                {
                    "page_number": 1,
                    "chunk_index": 0,
                    "content": "Operational guidance on AND gate logic, OR condition failures, and NOT ready signals NEAR edge nodes.",
                }
            ],
            db_path=self.test_db_path,
        )

        boolean_queries = [
            "AND",
            "OR",
            "NOT",
            "NEAR",
            "AND OR NOT NEAR",
            "AND AND AND",
            "server AND router NOT switch OR firewall NEAR database",
        ]

        for q in boolean_queries:
            results = db.search_knowledge_chunks_fts(q, db_path=self.test_db_path)
            self.assertIsInstance(results, list, f"Query '{q}' failed to return a list.")
            self.assertGreaterEqual(len(results), 1, f"Query '{q}' expected to match chunk.")

    # --------------------------------------------------------------------------
    # 20. Special Characters and Punctuation Safety
    # --------------------------------------------------------------------------
    def test_20_special_characters_and_punctuation_safety(self):
        """Verify that error messages with complex punctuation, code, or symbols do not crash search."""
        doc_id = db.create_knowledge_document(
            document_name="Error_Catalog.txt",
            file_path="/docs/errors.txt",
            db_path=self.test_db_path,
        )

        db.insert_knowledge_chunks(
            doc_id,
            [
                {
                    "page_number": 1,
                    "chunk_index": 0,
                    "content": "[FATAL] 504 Gateway Timeout: upstream (10.0.1.25:8080) -> connection refused! status='FAILED'",
                }
            ],
            db_path=self.test_db_path,
        )

        punct_queries = [
            "[FATAL] 504 Gateway Timeout: upstream (10.0.1.25:8080) -> connection refused!",
            "status='FAILED' && error_code == 0x7F",
            "SELECT * FROM users WHERE '1'='1' -- sql injection attempt",
            "JSONDecodeError: Expecting ',' delimiter: line 1 column 12 (char 11)",
            "key: 'val', arr: [1, 2, 3], nested: {a: (b)}",
            "http://api.service.internal:8080/v1/health?check=true#frag",
        ]

        for q in punct_queries:
            results = db.search_knowledge_chunks_fts(q, db_path=self.test_db_path)
            self.assertIsInstance(results, list, f"Query '{q}' failed to return a list.")

    # --------------------------------------------------------------------------
    # 21. Unicode Search Behavior Handled Safely
    # --------------------------------------------------------------------------
    def test_21_unicode_search_behavior_handled_safely(self):
        """Verify that unicode, international characters, and accented text search safely."""
        doc_id = db.create_knowledge_document(
            document_name="International_Logs.txt",
            file_path="/docs/intl.txt",
            db_path=self.test_db_path,
        )

        db.insert_knowledge_chunks(
            doc_id,
            [
                {
                    "page_number": 1,
                    "chunk_index": 0,
                    "content": "Systemdiagnose: Überprüfung der Netzwerkkonfiguration und Fehlerberichte in München.",
                },
                {
                    "page_number": 2,
                    "chunk_index": 0,
                    "content": "Résolution d'incident pour le serveur d'authentification principal à Montréal.",
                },
                {
                    "page_number": 3,
                    "chunk_index": 0,
                    "content": "Database latency spike in São Paulo (sa-east-1) datacenter.",
                },
            ],
            db_path=self.test_db_path,
        )

        unicode_queries = [
            ("Überprüfung", "München"),
            ("Résolution", "Montréal"),
            ("São Paulo", "datacenter"),
        ]

        for query_term, expected_match in unicode_queries:
            results = db.search_knowledge_chunks_fts(query_term, db_path=self.test_db_path)
            self.assertGreaterEqual(len(results), 1, f"Expected match for unicode term '{query_term}'")
            self.assertIn(expected_match, results[0]["content"])

        # Non-Latin characters execute safely without error
        safe_non_latin = db.search_knowledge_chunks_fts("こんにちは 故障", db_path=self.test_db_path)
        self.assertIsInstance(safe_non_latin, list)

    # --------------------------------------------------------------------------
    # 22. BM25 Ordering Semantics and Properties
    # --------------------------------------------------------------------------
    def test_22_bm25_ordering_semantics_and_properties(self):
        """Verify BM25 score semantics: negative raw score, ordering, and relevance magnitude."""
        doc_id = db.create_knowledge_document(
            document_name="Diagnostics_Runbook.txt",
            file_path="/docs/diagnostics.txt",
            db_path=self.test_db_path,
        )

        # Include 2 target chunks with different term frequency plus background chunks
        # to ensure standard positive IDF scaling in SQLite FTS5 BM25
        chunks = [
            {
                "page_number": 1,
                "chunk_index": 0,
                "content": "Database deadlock detected. Deadlock resolver failed. Multiple deadlock threads locked in deadlock.",
            },
            {
                "page_number": 2,
                "chunk_index": 0,
                "content": "Routine daily maintenance log: one transient deadlock occurred earlier.",
            },
        ]
        for idx in range(8):
            chunks.append({
                "page_number": 3 + idx,
                "chunk_index": 0,
                "content": f"Unrelated network packet loss statistics across datacenter rack {idx}.",
            })

        db.insert_knowledge_chunks(doc_id, chunks, db_path=self.test_db_path)

        results = db.search_knowledge_chunks_fts("deadlock", limit=5, db_path=self.test_db_path)
        self.assertEqual(len(results), 2, "Expected exactly 2 matching chunks.")

        top_match = results[0]
        second_match = results[1]

        # Top match has higher term frequency
        self.assertEqual(top_match["page_number"], 1)
        self.assertEqual(second_match["page_number"], 2)

        # Raw BM25 scores are negative, where smaller (more negative) indicates higher relevance
        self.assertLess(top_match["bm25_score"], 0.0)
        self.assertLess(second_match["bm25_score"], 0.0)
        self.assertLess(
            top_match["bm25_score"],
            second_match["bm25_score"],
            "Top match must have a more negative raw BM25 score than second match.",
        )

        # relevance_score is abs(raw_bm25), so higher is more relevant
        self.assertGreater(
            top_match["relevance_score"],
            second_match["relevance_score"],
            "Top match must have a greater positive relevance_score than second match.",
        )

        # Verify all expected result keys are present
        for key in ["chunk_id", "document_id", "document_name", "page_number", "chunk_index", "content", "bm25_score", "relevance_score"]:
            self.assertIn(key, top_match)

    # --------------------------------------------------------------------------
    # 23. Atomic Document Creation With Chunks
    # --------------------------------------------------------------------------
    def test_23_create_knowledge_document_with_chunks_atomic(self):
        """Verify create_knowledge_document_with_chunks atomically creates doc and persists chunks."""
        chunks = [
            {"page_number": 1, "chunk_index": 0, "content": "Atomic chunk 1."},
            {"page_number": 1, "chunk_index": 1, "content": "Atomic chunk 2."},
            {"page_number": 2, "chunk_index": 0, "content": "Atomic chunk 3."},
        ]

        result = db.create_knowledge_document_with_chunks(
            document_name="Atomic_Doc.txt",
            file_path="/docs/atomic.txt",
            file_size_bytes=2048,
            content_hash="hash_atomic_123",
            document_type="TXT",
            chunks=chunks,
            status="Indexed",
            db_path=self.test_db_path,
        )

        self.assertIsInstance(result, dict)
        doc_id = result["document_id"]
        self.assertGreater(doc_id, 0)
        self.assertEqual(result["chunk_count"], 3)
        self.assertEqual(result["status"], "Indexed")

        # Verify chunks exist in DB and FTS
        db_chunks = db.get_chunks_for_document(doc_id, db_path=self.test_db_path)
        self.assertEqual(len(db_chunks), 3)

        fts_hits = db.search_knowledge_chunks_fts("Atomic chunk", db_path=self.test_db_path)
        self.assertEqual(len(fts_hits), 3)

    # --------------------------------------------------------------------------
    # 24. Atomic Rollback on Invalid Chunk in Batch
    # --------------------------------------------------------------------------
    def test_24_create_knowledge_document_with_chunks_rollback(self):
        """Verify that a malformed chunk in create_knowledge_document_with_chunks rolls back entirely."""
        malformed_chunks = [
            {"page_number": 1, "chunk_index": 0, "content": "Valid chunk."},
            {"page_number": 1, "chunk_index": 1, "content": "   "},  # Empty content
        ]

        with self.assertRaises(ValueError):
            db.create_knowledge_document_with_chunks(
                document_name="Failing_Doc.txt",
                file_path="/docs/fail.txt",
                file_size_bytes=100,
                content_hash="fail_hash_123",
                document_type="TXT",
                chunks=malformed_chunks,
                db_path=self.test_db_path,
            )

        # Confirm 0 documents and 0 chunks were committed
        conn = db.get_connection(self.test_db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM knowledge_documents WHERE document_name = 'Failing_Doc.txt';")
            self.assertEqual(cursor.fetchone()[0], 0)
            cursor.execute("SELECT COUNT(*) FROM knowledge_chunks;")
            self.assertEqual(cursor.fetchone()[0], 0)
        finally:
            conn.close()

    # --------------------------------------------------------------------------
    # 25. Production Database Isolation
    # --------------------------------------------------------------------------
    def test_25_production_database_isolation(self):
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


