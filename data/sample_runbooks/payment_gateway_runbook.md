# OpsPilot Runbook OPS-701: Payment Transaction Timeout Handling

## Context
Procedures for managing checkout latency and timeout alerts in the OpsPilot order pipeline.

## Indicators
- Error code: `PAYMENT_GATEWAY_TIMEOUT_504`
- Service: `payment-processor`
- Symptoms: Latency exceeds threshold, pending payment queue growth.

## Verification Steps
1. Review gateway latency metrics across upstream payment endpoints.
2. Check connection pool allocation status on payment-processor instances.
3. Query pending transactions requiring idempotency confirmation.

## Remediation Protocol
1. Enable secondary payment gateway route via configuration toggle.
2. Expand connection pool limit from 100 to 250 connections.
3. Execute automated reconciliation job for unconfirmed checkout sessions.
4. Notify operations support desk with affected transaction identifiers.
