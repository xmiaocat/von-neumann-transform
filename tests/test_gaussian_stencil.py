import numpy as np
import pytest

from von_neumann_transform import (
    BasisMethod,
    MatVecMethod,
    PrecondMethod,
    SolverMethod,
    VonNeumannTransform,
)
from von_neumann_transform.basis import _get_grid
from von_neumann_transform.overlap import (
    _get_ovlp_direct,
    _get_ovlp_linop,
    _get_ovlp_stencil,
)


@pytest.mark.parametrize("k", [2, 4, 5])
def test_stencil_is_exact_when_radius_covers_grid(k):
    _, _, wn, tn, _, alpha = _get_grid(k * k, 0.2, 5.0)
    stencil = _get_ovlp_stencil(alpha, wn, tn)
    dense = _get_ovlp_direct(alpha, wn, tn)

    assert stencil.nnz == k**4
    np.testing.assert_allclose(
        stencil.toarray(), dense, rtol=1e-14, atol=1e-14
    )


@pytest.mark.parametrize("k", [16, 64, 128])
def test_stencil_matvec_matches_full_overlap(k):
    _, _, wn, tn, _, alpha = _get_grid(k * k, 0.2, 5.0)
    stencil, precon = _get_ovlp_linop(
        alpha,
        wn,
        tn,
        MatVecMethod.GAUSSIAN_STENCIL,
        PrecondMethod.NONE,
    )
    full, _ = _get_ovlp_linop(alpha, wn, tn, MatVecMethod.TOEPLITZ_MATMUL)

    expected_nnz = (k * 9 - 20) ** 2
    assert _get_ovlp_stencil(alpha, wn, tn).nnz == expected_nnz
    rng = np.random.default_rng(k)
    for _ in range(3):
        x = rng.normal(size=k * k) + 1j * rng.normal(size=k * k)
        np.testing.assert_allclose(
            stencil @ x, full @ x, rtol=1e-12, atol=1e-13
        )
        np.testing.assert_allclose(stencil.H @ x, stencil @ x, atol=1e-13)
        np.testing.assert_array_equal(precon @ x, x)


def test_stencil_matvec_uses_no_fft(monkeypatch):
    k = 16
    _, _, wn, tn, _, alpha = _get_grid(k * k, 0.2, 5.0)

    def no_fft(*args, **kwargs):
        raise AssertionError("Gaussian stencil must not use an FFT")

    monkeypatch.setattr(np.fft, "fft", no_fft)
    stencil, _ = _get_ovlp_linop(alpha, wn, tn, MatVecMethod.GAUSSIAN_STENCIL)
    x = np.ones(k * k, dtype=np.complex128)
    assert np.isfinite(stencil @ x).all()


def test_stencil_transform_matches_dense_solve():
    k = 8
    vnt = VonNeumannTransform(k * k, 0.2, 5.0)
    rng = np.random.default_rng(42)
    signal = rng.normal(size=k * k) + 1j * rng.normal(size=k * k)

    q_stencil = vnt.transform(
        signal,
        basis_method=BasisMethod.FFT,
        matvec_method=MatVecMethod.GAUSSIAN_STENCIL,
        solver_method=SolverMethod.CG,
        rtol=1e-11,
    )
    q_direct = vnt.transform(
        signal,
        basis_method=BasisMethod.FFT,
        matvec_method=MatVecMethod.DIRECT,
        solver_method=SolverMethod.DIRECT,
    )

    np.testing.assert_allclose(q_stencil, q_direct, rtol=1e-8, atol=1e-9)
