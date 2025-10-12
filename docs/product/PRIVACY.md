# Privacy Policy (Beta)

This policy covers the Oscillink SDK and the optional hosted Cloud API.

SDK (local):
- No telemetry or network transmission by default. All computation is in-process.

Cloud API (beta):
- Data processed: Only the data you send in requests. No training or retention beyond request lifecycle unless an operator enables caching.
- Logs: We log metadata (request ID, timings, status codes). We do not log request bodies by default.
- Security: HTTPS enforced; API key required. Webhooks require signature verification and timestamp freshness checks.
- Subprocessors: Stripe for billing. Optional: Google Cloud (Cloud Run/Firestore) if you use our hosted service.
- Data location: If using our hosted service, data is processed in the region of deployment noted in release notes.
- Retention: If caching is enabled, derived artifacts are stored for TTL only; see operator documentation.
- Your rights: Contact contact@oscillink.com for access/deletion requests related to any stored account data.

Contact: privacy@oscillink.com
