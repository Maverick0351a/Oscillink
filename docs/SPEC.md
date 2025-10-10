# Oscillink Lattice — Math Spec (Phase 1)

**Energy (per lattice)**

\[
H(U)=\lambda_G\,\|U-Y\|_F^2 + \lambda_C\,\mathrm{tr}(U^\top L_{\mathrm{sym}}U) +
\lambda_Q\,\mathrm{tr}((U-\mathbf{1}\psi^\top)^\top B (U-\mathbf{1}\psi^\top)) + \lambda_P\,\mathrm{tr}(U^\top L_{\text{path}}U)
\]

- \(L_{\mathrm{sym}} = I - D^{-1/2} A D^{-1/2}\), PSD.
- \(B=\mathrm{diag}(b)\), \(b\ge 0\); \(L_{\text{path}}\) is the normalized Laplacian over a chain prior (PSD).
- With \(\lambda_G>0\), the system matrix \(M=\lambda_G I + \lambda_C L_{\mathrm{sym}} + \lambda_Q B + \lambda_P L_{\text{path}}\) is **SPD**.

**Stationarity**

\[
MU^* = \lambda_G Y + \lambda_Q B\,\mathbf{1}\psi^\top
\]

**Implicit settle (resolvent)**

\[
(I+\Delta t M)U^+ = U + \Delta t(\lambda_G Y + \lambda_Q B\,\mathbf{1}\psi^\top)
\]

**Receipts**

\[
\Delta H = \mathrm{tr}((U-U^*)^\top M (U-U^*)) \ge 0
\]

Per‑node coherence drop uses normalized differences (consistent with \(L_{\mathrm{sym}}\)).
Null points are z‑scored residuals on edges.

**Chain prior**

Path Laplacian keeps SPD; chain receipts report verdict, weakest link (max z), and chain coherence gain.
