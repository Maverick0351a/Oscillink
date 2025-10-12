# Licensed Container Pricing (Launch Proposal)

This proposal balances early traction with enterprise credibility and keeps packaging simple (per-container licensing).

## TL;DR — Recommended Launch Pricing (Customer-managed containers)

| Tier         | Who it’s for                 | License unit              | Monthly            | What’s included |
|--------------|-------------------------------|---------------------------|--------------------|-----------------|
| Dev (free)   | Individual eval, labs         | 1 container (non-prod)    | $0                 | Single instance, capped inputs, community support |
| Starter      | Solo devs / small teams       | per container             | $49                | 1 container in prod; basic receipts; email support |
| Team         | Small prod apps               | per container             | $199               | 1 container; advanced receipts, diffusion gating; priority email |
| Scale Pack   | Mid-size                      | 5-container bundle        | $699               | Up to 5 containers; SSO (OIDC), usage dashboards |
| Enterprise   | Regulated / large             | per cluster (contract)    | $2,000–$6,000/mo   | Unlimited containers in one cluster, SSO/SAML, HA guidance, support SLA; optional air-gap licensing |

Annual discount: 2 months free (~17%) if paid yearly.
Founding cohort: first 100 buyers lock pricing for 12 months.

## Market anchors (context)

- Pinecone: publicly lists minimum monthly commitments (Standard $50/mo, Enterprise $500/mo) — managed cloud, not self-managed.
- Qdrant Cloud (Hybrid): as low as ~$0.014/hour (~$10.2/month always-on tiny cluster) — very low entry exists but is metered and resource-bound.
- Weaviate Enterprise Cloud (AWS Marketplace): $10,000/yr commit + $0.285 per 1M vector dimensions — shows enterprise budgets and scale-based expectations.

Takeaway: $49 starter sits under the common $50 paid floor; $199 leaves room for support/features; $699 bundle encourages small clusters; $2–$6k/mo enterprise sits below many contract-only offers.

## Packaging & licensing model

- License unit: per running container (license heartbeat) with optional air-gapped monthly key renewal.
- Not billed by usage in v1: keep licensing simple; collect telemetry for insights/ROI only.
- Scope: {tier, container_count or bundle, features}.

## What to meter (insights only)

- Node·dim processed, ΔH distributions, CG iterations, settle_ms, null-point density.
- Use for dashboards & ROI narratives, not billing.

## Entitlements by tier

- Dev (free)
  - Limits: N≤5k, D≤1,024; community support
  - No server auth/SSO
  - Optional watermark in receipts ("Not for production")
- Starter ($49)
  - One production container
  - Basic receipts (ΔH, timings)
  - Email support; rolling key renewal
- Team ($199)
  - Adds diffusion gating, chain-verification receipts, signed receipts (HMAC)
  - Grafana dashboard JSON
  - Priority email
- Scale Pack ($699)
  - Up to 5 containers
  - SSO (OIDC), multi-container usage roll-ups, Prometheus metrics
  - Advisory tuning reports
- Enterprise ($2–$6k/mo)
  - Cluster/site license: unlimited containers in one cluster
  - SSO/SAML, audit logging, HA/DR playbooks, private registry
  - Named support with SLA; optional air-gap & offline metering

## Support SLAs

- Starter/Team: email, 3-business-day target
- Scale: next-business-day
- Enterprise: 8×5 or 24×7 (contracted)

## Implementation checklist (to enforce pricing)

- License validation
  - Online: short heartbeat with grace window; scope includes tier & features
  - Offline/air-gap: time-boxed license file (e.g., 30 days), manual renewal
- Feature gates by tier
  - Diffusion gating, chain priors, signed receipts, SSO, dashboards, HA helper endpoints
- No hard pay-per-use in v1
  - Keep pricing per container for traction; use telemetry solely for recommendations & ROI

## Positioning & ROI at purchase

- Cheaper than RAG over-provisioning: even 10–20% reduction in re-prompts/token spend can exceed $199/mo on moderate workloads.
- Auditability value: deterministic receipts reduce incident time & compliance risk.
- On-prem comfort: per-container pricing is simpler to approve than opaque usage meters.
