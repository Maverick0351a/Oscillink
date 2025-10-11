import numpy as np

from oscillink.core.lattice import OscillinkLattice


def main():
    N, D = 120, 128
    Y = np.random.randn(N, D).astype(np.float32)
    psi = (Y[:20].mean(axis=0) / (np.linalg.norm(Y[:20].mean(axis=0)) + 1e-12)).astype(np.float32)

    lat = OscillinkLattice(Y, kneighbors=6, lamG=1.0, lamC=0.5, lamQ=4.0)
    lat.set_query(psi=psi)

    chain = [2, 5, 7, 9]
    lat.add_chain(chain=chain, lamP=0.2)

    diag = lat.settle(dt=1.0, max_iters=12, tol=1e-3)
    print("settle:", diag)

    rec = lat.receipt()
    print("receipt(deltaH):", rec["deltaH_total"], "nulls:", len(rec["null_points"]))

    crec = lat.chain_receipt(chain=chain)
    print("chain verdict:", crec["verdict"], "weakest:", crec["weakest_link"])

    bundle = lat.bundle(k=6)
    print("bundle top-3:", bundle[:3])


if __name__ == "__main__":
    main()
