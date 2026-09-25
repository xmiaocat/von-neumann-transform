import numpy as np
import pytest

from von_neumann_transform import (
    BasisMethod,
    MatVecMethod,
    SolverMethod,
    VonNeumannTransform,
)
from von_neumann_transform.basis import _get_grid
from von_neumann_transform.overlap import _get_ovlp_linop


@pytest.mark.parametrize("k", [4, 6, 16, 64, 128])
def test_banded_matvec_matches_full_fourier_blocks(k):
    _, _, w_n_arr, t_n_arr, _, alpha = _get_grid(k * k, 0.2, 5.0)
    full, full_precon = _get_ovlp_linop(
        alpha, w_n_arr, t_n_arr, MatVecMethod.TOEPLITZ_MATMUL
    )
    banded, banded_precon = _get_ovlp_linop(
        alpha, w_n_arr, t_n_arr, MatVecMethod.TOEPLITZ_BANDED
    )

    rng = np.random.default_rng(k)
    for _ in range(3):
        x = rng.normal(size=k * k) + 1j * rng.normal(size=k * k)
        np.testing.assert_allclose(
            banded @ x, full @ x, rtol=1e-12, atol=1e-13
        )
        np.testing.assert_allclose(
            banded_precon @ x,
            full_precon @ x,
            rtol=1e-10,
            atol=1e-11,
        )


def test_banded_transform_matches_dense_solve():
    k = 16
    vnt = VonNeumannTransform(k * k, 0.2, 5.0)
    rng = np.random.default_rng(42)
    signal = rng.normal(size=k * k) + 1j * rng.normal(size=k * k)

    q_banded = vnt.transform(
        signal,
        basis_method=BasisMethod.FFT,
        matvec_method=MatVecMethod.TOEPLITZ_BANDED,
        solver_method=SolverMethod.CG,
        rtol=1e-11,
    )
    q_direct = vnt.transform(
        signal,
        basis_method=BasisMethod.FFT,
        matvec_method=MatVecMethod.DIRECT,
        solver_method=SolverMethod.DIRECT,
    )

    np.testing.assert_allclose(q_banded, q_direct, rtol=1e-8, atol=1e-9)
