import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cg

from von_neumann_transform import (
    MatVecMethod,
    PrecondMethod,
    SolverMethod,
    VonNeumannTransform,
    overlap,
)
from von_neumann_transform.basis import _get_grid
from von_neumann_transform.overlap import _get_ic0_factor, _get_ovlp_linop

MATVEC_METHODS = (
    MatVecMethod.TOEPLITZ_MATMUL,
    MatVecMethod.TOEPLITZ_EINSUM,
    MatVecMethod.TOEPLITZ_BANDED,
    MatVecMethod.GAUSSIAN_STENCIL,
)
PRECON_DEFAULTS = (
    (MatVecMethod.TOEPLITZ_MATMUL, PrecondMethod.CIRCULANT_DENSE),
    (MatVecMethod.TOEPLITZ_EINSUM, PrecondMethod.CIRCULANT_DENSE),
    (MatVecMethod.TOEPLITZ_BANDED, PrecondMethod.CIRCULANT_BANDED),
    (
        MatVecMethod.GAUSSIAN_STENCIL,
        PrecondMethod.INCOMPLETE_CHOLESKY,
    ),
)


@pytest.mark.parametrize("matvec_method,precond_method", PRECON_DEFAULTS)
def test_auto_preserves_preconditioner_pairing(matvec_method, precond_method):
    k = 16
    _, _, wn, tn, _, alpha = _get_grid(k * k, 0.2, 5.0)
    auto_op, auto_precon = _get_ovlp_linop(alpha, wn, tn, matvec_method)
    explicit_op, explicit_precon = _get_ovlp_linop(
        alpha, wn, tn, matvec_method, precond_method
    )
    rng = np.random.default_rng(42)
    x = rng.normal(size=k * k) + 1j * rng.normal(size=k * k)

    np.testing.assert_allclose(auto_op @ x, explicit_op @ x)
    np.testing.assert_allclose(auto_precon @ x, explicit_precon @ x)


@pytest.mark.parametrize("matvec_method", MATVEC_METHODS)
@pytest.mark.parametrize(
    "precond_method,reference_matvec",
    [
        (PrecondMethod.CIRCULANT_DENSE, MatVecMethod.TOEPLITZ_MATMUL),
        (PrecondMethod.CIRCULANT_BANDED, MatVecMethod.TOEPLITZ_BANDED),
        (
            PrecondMethod.INCOMPLETE_CHOLESKY,
            MatVecMethod.GAUSSIAN_STENCIL,
        ),
    ],
)
def test_preconditioner_is_independent_of_matvec(
    matvec_method, precond_method, reference_matvec
):
    k = 16
    _, _, wn, tn, _, alpha = _get_grid(k * k, 0.2, 5.0)
    operator, precon = _get_ovlp_linop(
        alpha, wn, tn, matvec_method, precond_method
    )
    default_operator, _ = _get_ovlp_linop(alpha, wn, tn, matvec_method)
    _, reference_precon = _get_ovlp_linop(alpha, wn, tn, reference_matvec)
    rng = np.random.default_rng(43)
    x = rng.normal(size=k * k) + 1j * rng.normal(size=k * k)

    np.testing.assert_allclose(operator @ x, default_operator @ x)
    np.testing.assert_allclose(precon @ x, reference_precon @ x)


@pytest.mark.parametrize("matvec_method", MATVEC_METHODS)
def test_none_preconditioner_is_identity(matvec_method):
    k = 8
    _, _, wn, tn, _, alpha = _get_grid(k * k, 0.2, 5.0)
    _, precon = _get_ovlp_linop(
        alpha, wn, tn, matvec_method, PrecondMethod.NONE
    )
    x = np.arange(k * k, dtype=np.complex128) + 1j
    np.testing.assert_array_equal(precon @ x, x)


@pytest.mark.parametrize(
    "precond_method",
    (
        PrecondMethod.NONE,
        PrecondMethod.CIRCULANT_DENSE,
        PrecondMethod.CIRCULANT_BANDED,
        PrecondMethod.INCOMPLETE_CHOLESKY,
    ),
)
def test_preconditioner_is_positive_on_random_vectors(precond_method):
    k = 16
    _, _, wn, tn, _, alpha = _get_grid(k * k, 0.2, 5.0)
    _, precon = _get_ovlp_linop(
        alpha, wn, tn, MatVecMethod.GAUSSIAN_STENCIL, precond_method
    )
    rng = np.random.default_rng(45)
    for _ in range(3):
        x = rng.normal(size=k * k) + 1j * rng.normal(size=k * k)
        value = np.vdot(x, precon @ x)
        assert value.real > 0.0
        assert abs(value.imag) < 1e-12 * value.real


@pytest.mark.parametrize(
    "precond_method",
    (
        PrecondMethod.NONE,
        PrecondMethod.CIRCULANT_DENSE,
        PrecondMethod.CIRCULANT_BANDED,
        PrecondMethod.INCOMPLETE_CHOLESKY,
    ),
)
def test_stencil_transform_with_selected_preconditioner(precond_method):
    k = 16
    vnt = VonNeumannTransform(k * k, 0.2, 5.0)
    rng = np.random.default_rng(44)
    signal = rng.normal(size=k * k) + 1j * rng.normal(size=k * k)

    result = vnt.transform(
        signal,
        matvec_method=MatVecMethod.GAUSSIAN_STENCIL,
        precond_method=precond_method,
        solver_method=SolverMethod.CG,
        rtol=1e-11,
    )
    reference = vnt.transform(
        signal,
        matvec_method=MatVecMethod.DIRECT,
        solver_method=SolverMethod.DIRECT,
    )
    np.testing.assert_allclose(result, reference, rtol=1e-8, atol=1e-9)


def test_direct_method_rejects_circulant_preconditioner():
    vnt = VonNeumannTransform(16, 0.2, 5.0)
    with pytest.raises(ValueError, match="does not use a preconditioner"):
        vnt.get_ovlp(
            vnt.alpha,
            vnt.w_n_arr,
            vnt.t_n_arr,
            MatVecMethod.DIRECT,
            PrecondMethod.CIRCULANT_BANDED,
        )


def test_ic0_is_exact_for_tridiagonal_matrix():
    matrix = csr_matrix(
        np.array(
            [
                [3, 1 + 1j, 0],
                [1 - 1j, 4, 1 + 2j],
                [0, 1 - 2j, 5],
            ],
            dtype=np.complex128,
        )
    )
    factor = _get_ic0_factor(matrix)
    np.testing.assert_allclose(
        (factor @ factor.conj().T).toarray(), matrix.toarray()
    )
    assert factor.nnz == 5


def test_ic0_reports_nonpositive_pivot():
    matrix = csr_matrix(np.array([[1, 2], [2, 1]], dtype=np.complex128))
    with pytest.raises(np.linalg.LinAlgError, match="non-positive pivot"):
        _get_ic0_factor(matrix)


def test_ic0_reduces_stencil_cg_iterations():
    k = 32
    _, _, wn, tn, _, alpha = _get_grid(k * k, 0.2, 5.0)
    operator, ic0 = _get_ovlp_linop(
        alpha, wn, tn, MatVecMethod.GAUSSIAN_STENCIL
    )
    _, identity = _get_ovlp_linop(
        alpha, wn, tn, MatVecMethod.GAUSSIAN_STENCIL, PrecondMethod.NONE
    )
    rng = np.random.default_rng(20260925)
    rhs = rng.normal(size=k * k) + 1j * rng.normal(size=k * k)

    def solve_with(preconditioner):
        iterations = 0

        def count_iteration(_):
            nonlocal iterations
            iterations += 1

        solution, info = cg(
            operator,
            rhs,
            M=preconditioner,
            rtol=1e-8,
            callback=count_iteration,
        )
        assert info == 0
        assert np.linalg.norm(operator @ solution - rhs) <= (
            1e-8 * np.linalg.norm(rhs)
        )
        return iterations

    assert solve_with(ic0) < solve_with(identity)


@pytest.mark.parametrize(
    "matvec_method,precond_method,expected_counts",
    [
        (
            MatVecMethod.TOEPLITZ_MATMUL,
            PrecondMethod.CIRCULANT_DENSE,
            (1, 0),
        ),
        (
            MatVecMethod.TOEPLITZ_BANDED,
            PrecondMethod.CIRCULANT_BANDED,
            (0, 1),
        ),
        (
            MatVecMethod.TOEPLITZ_MATMUL,
            PrecondMethod.CIRCULANT_BANDED,
            (1, 1),
        ),
        (
            MatVecMethod.GAUSSIAN_STENCIL,
            PrecondMethod.NONE,
            (0, 0),
        ),
        (
            MatVecMethod.GAUSSIAN_STENCIL,
            PrecondMethod.INCOMPLETE_CHOLESKY,
            (0, 0),
        ),
    ],
)
def test_shared_fourier_setup(
    monkeypatch, matvec_method, precond_method, expected_counts
):
    counts = [0, 0]
    original_dense = overlap._get_ovlp_fft_dense
    original_band = overlap._get_ovlp_fft_band

    def dense(*args, **kwargs):
        counts[0] += 1
        return original_dense(*args, **kwargs)

    def band(*args, **kwargs):
        counts[1] += 1
        return original_band(*args, **kwargs)

    monkeypatch.setattr(overlap, "_get_ovlp_fft_dense", dense)
    monkeypatch.setattr(overlap, "_get_ovlp_fft_band", band)
    _, _, wn, tn, _, alpha = _get_grid(8 * 8, 0.2, 5.0)
    _get_ovlp_linop(alpha, wn, tn, matvec_method, precond_method)
    assert tuple(counts) == expected_counts
