This document has moved.

New location: [docs/ops/IMAGE_SIGNING.md](./ops/IMAGE_SIGNING.md)
# Image signing and digest pinning

This guide shows how to sign your container images with Sigstore cosign and pin images by digest for stronger supply-chain guarantees.

## Cosign signing

Prereqs:
- Install cosign: https://docs.sigstore.dev/cosign/installation/
- Use keyless (OIDC) or key-based signing. Keyless works great on CI with GitHub OIDC.

### Sign locally (keyless)

```bash
cosign sign <your-registry>/oscillink-licensed:1.0.0
```

### Verify

```bash
cosign verify <your-registry>/oscillink-licensed:1.0.0
```

Use `--certificate-identity` and `--certificate-oidc-issuer` to scope verification to your org.

## Digest pinning

After pushing, get the digest and use it in Helm values or your deployment manifests.

```bash
# Get the digest
crane digest <your-registry>/oscillink-licensed:1.0.0
# sha256:deadbeef...
```

Helm values example:

```yaml
image:
  repository: <your-registry>/oscillink-licensed
  tag: 1.0.0
  digest: sha256:deadbeef...
```

Notes:
- Kubernetes will pull the exact content-addressed image when `digest` is set.
- Keep `tag` for human readability; `digest` enforces the precise image.
- Combine this with SBOM and vulnerability scanning (pip-audit/Trivy) for better coverage.
