# Oscillink Helm examples

This folder contains example manifests and values to help you enable optional production add-ons (NetworkPolicy, PDB, HPA, and Ingress) and wire an admin secret for the `/admin/*` endpoints.

## Files

- `admin-secret.yaml` — Example Secret to expose sensitive envs via `envFrom` in the Deployment (e.g., `OSCILLINK_ADMIN_SECRET`).
- `values-with-addons.yaml` — Example values enabling NetworkPolicy, PDB, HPA, and Ingress with sensible defaults.
- `values-gke.yaml` — GKE-friendly defaults (nginx ingress class, TLS secret wiring, HPA/PDB, NetworkPolicy).
- `values-eks.yaml` — EKS-friendly defaults (ALB ingress class + annotations, TLS secret wiring, HPA/PDB, NetworkPolicy).
- `values-aks.yaml` — AKS-friendly defaults (nginx ingress class, TLS secret wiring, HPA/PDB, NetworkPolicy).
- `values-private.yaml` — Private cluster defaults (no public ingress, strict egress, optional proxy envs).
- `values-proxy.yaml` — Proxy-only tweak to layer on outbound proxy env vars.
- `cert-manager/cluster-issuer.yaml` and `cert-manager/certificate.yaml` — Sample manifests for automatic TLS.
- `cert-manager/cluster-issuer-eks.yaml` and `cert-manager/cluster-issuer-aks.yaml` — Controller-specific Issuer samples.
- `values-security.yaml` — Opinionated security-first defaults (metrics protected, JSON logs sampled, strict egress, no public ingress).
- `values-ingress-privacy.yaml` — Overlay to disable access logs on ingress and limit body size.

## Usage

1) Create the admin secret (optional but recommended if you use `/admin/introspect`). Adjust the secret name or values as needed.

```powershell
kubectl apply -n <namespace> -f deploy/helm/oscillink/examples/admin-secret.yaml
```

2) Install the chart using the example values. Override image/tag and host to match your environment.

```powershell
helm upgrade --install oscillink deploy/helm/oscillink `
  --namespace <namespace> `
  --create-namespace `
  -f deploy/helm/oscillink/examples/values-with-addons.yaml
For private clusters or outbound proxies:

```powershell
helm upgrade --install oscillink deploy/helm/oscillink `
  --namespace <namespace> `
  --create-namespace `
  -f deploy/helm/oscillink/examples/values-private.yaml
```

Proxy only (combine with your cluster preset values):

```powershell
helm upgrade --install oscillink deploy/helm/oscillink `
  --namespace <namespace> `
  --create-namespace `
  -f deploy/helm/oscillink/examples/values-gke.yaml `
  -f deploy/helm/oscillink/examples/values-proxy.yaml
```

TLS with cert-manager:

```powershell
kubectl apply -f deploy/helm/oscillink/examples/cert-manager/cluster-issuer.yaml
kubectl apply -f deploy/helm/oscillink/examples/cert-manager/certificate.yaml
```
```

3) Verify readiness and health:

- Readiness: `/license/status` (Deployment readinessProbe)
- Liveness: `/health`

## Notes

- NetworkPolicy: The example allows ingress to port 8080 and egress to JWKS endpoints on ports 443/80 plus DNS. Tighten `egressCIDR` to your egress gateway/network.
- PDB: Ensures at least one pod is available during voluntary disruptions.
- HPA: Scales from 2 to 6 replicas at ~65% CPU. Adjust based on your workload.
- Ingress: Configured with a placeholder host and TLS secret; integrate with your ingress controller (NGINX, ALB, etc.) and cert-manager for automation.
