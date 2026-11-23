# OpsPilot Runbook OPS-702: Database Connection Saturation

## Context
Standard operational guidance for database connection exhaustion and query timeout mitigation.

## Indicators
- Error signature: `FATAL: remaining connection slots are reserved for non-replication superuser connections`
- Metric: Active client connections reach 100% of maximum pool capacity.
- Symptoms: Cascading request queueing across backend application services.

## Verification Steps
1. Inspect active connection states in PostgreSQL connection pool metrics.
2. Identify long-running uncommitted transactions holding row-level locks.
3. Check application replica routing to confirm read queries use read replicas.

## Remediation Protocol
1. Terminate idle-in-transaction connections older than 180 seconds.
2. Enable pgBouncer connection pooler transaction pooling mode.
3. Scale read replica cluster instances to offload analytical query traffic.
4. Verify application connection pool settings are aligned with backend limits.
