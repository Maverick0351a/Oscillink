This document has moved.

New location: [docs/ops/NETWORKING.md](./ops/NETWORKING.md)
# Networking & egress strategy

Kubernetes NetworkPolicy cannot match FQDNs. To restrict egress by domain, use one of:

- Egress gateway (preferred): route all outbound traffic through a controlled egress gateway or proxy; allow only gateway CIDRs in NetworkPolicy.
- Proxy overlay: set HTTP(S)_PROXY envs; allow only proxy IPs in NetworkPolicy. See values-proxy.yaml.
- Service mesh: define a ServiceEntry or equivalent for the JWKS/licensing endpoints; allow only mesh egress in NetworkPolicy.
- Static IP allowlist: if your JWKS endpoint has a stable IP range, restrict egress CIDRs accordingly.

Notes:
- Allow DNS (UDP/TCP 53) when restricting egress, or use node-local DNS policies.
- JWKS caching + offline grace: the container supports `OSCILLINK_JWKS_TTL` and `OSCILLINK_JWKS_OFFLINE_GRACE` to reduce dependence on continuous external egress.
