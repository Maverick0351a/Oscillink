This document has moved.

New location: [docs/ops/OBSERVABILITY.md](./ops/OBSERVABILITY.md)
# Observability: Prometheus + Grafana

This repo exposes Prometheus metrics at `/metrics` via the FastAPI service (`cloud/app/main.py`).
Use the provided Grafana dashboard to visualize request rates, latency, usage, and Stripe webhook health.

## Metrics exposed

- oscillink_settle_requests_total{status}
- oscillink_settle_latency_seconds (histogram)
- oscillink_settle_last_N (gauge)
- oscillink_settle_last_D (gauge)
- oscillink_usage_nodes_total (counter)
- oscillink_usage_node_dim_units_total (counter)
- oscillink_job_queue_depth (gauge)
- oscillink_stripe_webhook_events_total{result}

## Prometheus scrape example

If you run Prometheus yourself, add a scrape job like:

```
scrape_configs:
  - job_name: "oscillink-cloud"
    scrape_interval: 10s
    metrics_path: /metrics
    static_configs:
      - targets: ["localhost:8000"]
```

Cloud Run note: Put Prometheus behind a private ingress or use an agent that can scrape your service securely.

## Grafana dashboard

Import `assets/grafana/oscillink_dashboard.json` into Grafana and select your Prometheus datasource when prompted.
The dashboard includes:

- Requests per second and error rate
- Latency quantiles (p50/p95/p99) in ms
- Units processed (node·dim) per second
- Job queue depth
- Stripe webhook events (processed/ignored/duplicate)

## Local quick start (dev)

1. Start the API locally: `uvicorn cloud.app.main:app --port 8000`
2. Point Prometheus at `http://localhost:8000/metrics`
3. Import the dashboard JSON in Grafana and set the Prometheus datasource.

That’s it — you’ll get live visibility after the first requests hit the API.

## Cloud Run + managed observability (prod)

There are two common options on GCP:

1) Google Managed Service for Prometheus (GMP)
  - Deploy the managed collector in your project/cluster and configure it to scrape your Cloud Run service’s `/metrics` endpoint.
  - Pros: Native GCP auth/metrics pipeline, scales well; integrates with Cloud Monitoring and managed Grafana.
  - Docs: https://cloud.google.com/stackdriver/docs/managed-prometheus

2) Self‑hosted Prometheus/Grafana
  - Point Prometheus to your public Cloud Run URL (or custom domain) over HTTPS.
  - Minimal scrape config example (public endpoint):

```
scrape_configs:
  - job_name: "oscillink-cloudrun"
   scrape_interval: 15s
   metrics_path: /metrics
   scheme: https
   static_configs:
    - targets: ["api2.odinprotocol.dev"]  # or <service>-<hash>-<project>.us-central1.run.app
```

Notes:
- If you restrict ingress (private), use a private agent/collector with appropriate access.
- For multiple regions/services, add multiple targets or discover them dynamically.
- Import `assets/grafana/oscillink_dashboard.json` into Grafana (managed or self‑hosted) and select your Prometheus datasource.
