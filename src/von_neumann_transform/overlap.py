import numpy as np
from scipy.linalg import cho_solve_banded, cholesky_banded, solve_triangular
from scipy.sparse import coo_matrix, csr_matrix, tril
from scipy.sparse.linalg import (
    LinearOperator,
    aslinearoperator,
    spsolve_triangular,
)

from .methods import MatVecMethod, PrecondMethod


def _chol_solve_batch(l_mat: np.ndarray, b_mat: np.ndarray) -> None:
    solve_triangular(
        l_mat,
        b_mat,
        lower=True,
        trans=0,
        overwrite_b=True,
        check_finite=False,
    )  # forward
    solve_triangular(
        l_mat.conj().swapaxes(-1, -2),
        b_mat,
        lower=False,
        trans=0,
        overwrite_b=True,
        check_finite=False,
    )  # backward


def _get_ovlp_direct(
    alpha: float,
    w_n_arr: np.ndarray,
    t_n_arr: np.ndarray,
) -> np.ndarray:
    dw = w_n_arr[1] - w_n_arr[0]
    dt = t_n_arr[1] - t_n_arr[0]
    k = len(w_n_arr)

    # construct distance matrix
    small_dist_mat = np.zeros((k, k), dtype=np.complex128)
    for i in range(1, k):
        np.fill_diagonal(small_dist_mat[:, i:], i)
    small_dist_mat += -small_dist_mat.T

    # construct sum matrix
    small_sum_mat = np.add.outer(
        np.arange(k, dtype=np.complex128),
        np.arange(k, dtype=np.complex128),
    )

    tmp = (
        -0.5
        * alpha
        * (
            np.repeat(np.repeat(small_dist_mat**2, k, axis=0), k, axis=1)
            * dw**2
        )
    )
    tmp += -(1.0 / (8.0 * alpha)) * (
        np.tile(small_dist_mat**2, (k, k)) * dt**2
    )
    tmp += (
        0.5j
        * (np.repeat(np.repeat(small_dist_mat, k, axis=0), k, axis=1) * dw)
        * (np.tile(small_sum_mat, (k, k)) * dt + 2.0 * t_n_arr.min())
    )
    s = np.exp(tmp)

    return s


def _get_ovlp_block(
    alpha: float,
    w_n_arr: np.ndarray,
    t_n_arr: np.ndarray,
) -> np.ndarray:
    k = len(w_n_arr)
    dw = w_n_arr[1] - w_n_arr[0]
    dt = t_n_arr[1] - t_n_arr[0]

    # matrices that depend only on *within-block* indices m, n
    idx = np.arange(k, dtype=np.complex128)
    diff_mat = idx[:, np.newaxis] - idx[np.newaxis, :]  # m - n
    diff2_mat = diff_mat**2
    sum_mat = idx[:, np.newaxis] + idx[np.newaxis, :]  # m + n

    col_blocks = np.empty((k, k, k), dtype=np.complex128)
    for j in range(k):  # j = 0 … k−1  ⇒  block column index
        col_blocks[j] = _get_ovlp_block_values(
            alpha, dw, dt, t_n_arr.min(), j, diff2_mat, sum_mat
        )

    return col_blocks


def _get_ovlp_block_values(
    alpha: float,
    dw: float,
    dt: float,
    t_min: float,
    block_offset: int,
    inner_diff2: np.ndarray | int,
    inner_sum: np.ndarray,
) -> np.ndarray:
    """Evaluate entries of a block from its outer and inner offsets."""
    exponent = (
        -0.5 * alpha * block_offset**2 * dw**2
        - (1.0 / (8.0 * alpha)) * inner_diff2 * dt**2
        + 0.5j * (-block_offset * dw) * (inner_sum * dt + 2.0 * t_min)
    )
    return np.exp(exponent)


def _get_ovlp_fft_band(
    alpha: float,
    w_n_arr: np.ndarray,
    t_n_arr: np.ndarray,
    bandwidth: int = 4,
) -> np.ndarray:
    """Lower bands of the real Fourier-domain circulant blocks.

    The array uses SciPy's lower banded layout: entry [mode, offset, n]
    represents the Fourier block entry [n + offset, n].
    """
    k = len(w_n_arr)
    nc = 2 * k
    bandwidth = min(bandwidth, k - 1)
    dw = w_n_arr[1] - w_n_arr[0]
    dt = t_n_arr[1] - t_n_arr[0]

    band_fft = np.zeros((nc, bandwidth + 1, k), dtype=np.float64)
    for offset in range(bandwidth + 1):
        n = np.arange(k - offset)
        m = n + offset
        circ = np.zeros((nc, k - offset), dtype=np.complex128)
        for d in range(k):
            circ[d] = _get_ovlp_block_values(
                alpha, dw, dt, t_n_arr.min(), d, offset**2, m + n
            )
        circ[k + 1 :] = circ[1:k][::-1].conj()
        # The signed block offsets pair by conjugation, so the FFT is real.
        band_fft[:, offset, : k - offset] = np.fft.fft(circ, axis=0).real

    return band_fft


def _get_ovlp_fft_dense(
    alpha: float,
    w_n_arr: np.ndarray,
    t_n_arr: np.ndarray,
) -> np.ndarray:
    k = len(w_n_arr)
    blocks = _get_ovlp_block(alpha, w_n_arr, t_n_arr)
    circulant = np.zeros((2 * k, k, k), dtype=np.complex128)
    circulant[:k] = blocks
    circulant[k + 1 :] = blocks[1:][::-1].transpose((0, 2, 1)).conj()
    return np.fft.fft(circulant, axis=0)


def _get_ovlp_stencil(
    alpha: float,
    w_n_arr: np.ndarray,
    t_n_arr: np.ndarray,
    radius: int = 4,
) -> csr_matrix:
    """Store the exact overlap entries within a Gaussian R-neighborhood.

    The outer and inner offsets are both truncated to ``[-radius, radius]``.
    No circulant embedding or FFT is involved in the resulting matvec.
    """
    k = len(w_n_arr)
    radius = min(radius, k - 1)
    dw = w_n_arr[1] - w_n_arr[0]
    dt = t_n_arr[1] - t_n_arr[0]
    t_min = t_n_arr.min()

    offsets = range(-radius, radius + 1)
    count_per_axis = sum(k - abs(offset) for offset in offsets)
    nnz = count_per_axis**2
    index_dtype = np.int32 if k * k <= np.iinfo(np.int32).max else np.int64
    rows = np.empty(nnz, dtype=index_dtype)
    cols = np.empty(nnz, dtype=index_dtype)
    values = np.empty(nnz, dtype=np.complex128)
    cursor = 0
    for d in range(-radius, radius + 1):
        p_start, p_stop = max(d, 0), min(k + d, k)
        p_count = p_stop - p_start
        p = np.arange(p_start, p_stop)
        for ell in range(-radius, radius + 1):
            m_start, m_stop = max(ell, 0), min(k + ell, k)
            m_count = m_stop - m_start
            m = np.arange(m_start, m_stop)
            n = m - ell
            weights = _get_ovlp_block_values(
                alpha, dw, dt, t_min, d, ell**2, m + n
            )
            next_cursor = cursor + p_count * m_count
            row_view = rows[cursor:next_cursor].reshape(p_count, m_count)
            row_view[:] = p[:, None] * k + m
            cols[cursor:next_cursor] = rows[cursor:next_cursor] - d * k - ell
            values[cursor:next_cursor].reshape(p_count, m_count)[:] = weights
            cursor = next_cursor

    matrix = coo_matrix(
        (values, (rows, cols)),
        shape=(k * k, k * k),
    ).tocsr()
    matrix.sort_indices()
    return matrix


def _get_ic0_factor(matrix: csr_matrix) -> csr_matrix:
    """Incomplete Cholesky factor with no fill beyond the lower pattern.

    The natural row-major ordering of the Gaussian stencil is retained.
    A non-positive pivot is reported rather than silently changing the
    preconditioner with a diagonal shift.
    """
    lower = tril(matrix, format="csr")
    lower.sort_indices()
    indices = lower.indices
    indptr = lower.indptr
    values = lower.data.copy()

    for i in range(lower.shape[0]):
        start, stop = indptr[i : i + 2]
        if start == stop or indices[stop - 1] != i:
            raise np.linalg.LinAlgError(
                f"IC(0) requires a diagonal entry in row {i}."
            )
        diagonal_position = stop - 1
        positions = {
            int(indices[position]): position
            for position in range(start, diagonal_position)
        }

        for position in range(start, diagonal_position):
            j = int(indices[position])
            correction = 0j
            for prior in range(indptr[j], indptr[j + 1] - 1):
                matching = positions.get(int(indices[prior]))
                if matching is not None:
                    correction += values[matching] * values[prior].conjugate()
            values[position] = (values[position] - correction) / values[
                indptr[j + 1] - 1
            ]

        diagonal = values[diagonal_position]
        pivot = (
            diagonal.real
            - np.vdot(
                values[start:diagonal_position],
                values[start:diagonal_position],
            ).real
        )
        if not np.isfinite(pivot) or pivot <= 0:
            raise np.linalg.LinAlgError(
                f"IC(0) failed at row {i}: non-positive pivot {pivot}."
            )
        values[diagonal_position] = np.sqrt(pivot)

    return csr_matrix((values, indices, indptr), shape=lower.shape)


def _get_ovlp_linop(
    alpha: float,
    w_n_arr: np.ndarray,
    t_n_arr: np.ndarray,
    matvec_method: MatVecMethod = MatVecMethod.TOEPLITZ_MATMUL,
    precond_method: PrecondMethod = PrecondMethod.AUTO,
) -> tuple[LinearOperator, LinearOperator]:
    k = len(w_n_arr)
    nc = 2 * k

    if matvec_method is MatVecMethod.DIRECT:
        raise RuntimeError(
            "DIRECT method for matrix-vector multiplication "
            "can only be used with the get_ovlp_direct method."
        )

    if precond_method is PrecondMethod.AUTO:
        if matvec_method in (
            MatVecMethod.TOEPLITZ_MATMUL,
            MatVecMethod.TOEPLITZ_EINSUM,
        ):
            precond_method = PrecondMethod.CIRCULANT_DENSE
        elif matvec_method is MatVecMethod.TOEPLITZ_BANDED:
            precond_method = PrecondMethod.CIRCULANT_BANDED
        elif matvec_method is MatVecMethod.GAUSSIAN_STENCIL:
            precond_method = PrecondMethod.INCOMPLETE_CHOLESKY
        else:
            raise ValueError(f"Unknown matvec method: {matvec_method!r}")
    elif precond_method not in (
        PrecondMethod.NONE,
        PrecondMethod.CIRCULANT_DENSE,
        PrecondMethod.CIRCULANT_BANDED,
        PrecondMethod.INCOMPLETE_CHOLESKY,
    ):
        raise ValueError(f"Unknown preconditioner method: {precond_method!r}")

    need_dense_fft = (
        matvec_method
        in (
            MatVecMethod.TOEPLITZ_MATMUL,
            MatVecMethod.TOEPLITZ_EINSUM,
        )
        or precond_method is PrecondMethod.CIRCULANT_DENSE
    )
    need_band_fft = (
        matvec_method is MatVecMethod.TOEPLITZ_BANDED
        or precond_method is PrecondMethod.CIRCULANT_BANDED
    )
    s_fft = (
        _get_ovlp_fft_dense(alpha, w_n_arr, t_n_arr)
        if need_dense_fft
        else None
    )
    s_band = (
        _get_ovlp_fft_band(alpha, w_n_arr, t_n_arr) if need_band_fft else None
    )

    stencil = (
        _get_ovlp_stencil(alpha, w_n_arr, t_n_arr)
        if (
            matvec_method is MatVecMethod.GAUSSIAN_STENCIL
            or precond_method is PrecondMethod.INCOMPLETE_CHOLESKY
        )
        else None
    )

    if matvec_method is MatVecMethod.GAUSSIAN_STENCIL:
        assert stencil is not None
        s_op = aslinearoperator(stencil)
    else:
        if matvec_method is MatVecMethod.TOEPLITZ_BANDED:
            band_fft = s_band
            assert band_fft is not None

            def contract(x, y):
                y.fill(0.0)
                for offset in range(band_fft.shape[1]):
                    diagonal = band_fft[:, offset, : k - offset]
                    y[:, offset:, 0] += diagonal * x[:, : k - offset, 0]
                    if offset:
                        y[:, : k - offset, 0] += diagonal * x[:, offset:, 0]

        elif matvec_method is MatVecMethod.TOEPLITZ_MATMUL:
            dense_fft = s_fft
            assert dense_fft is not None

            def contract(x, y):
                np.matmul(dense_fft, x, out=y)

        elif matvec_method is MatVecMethod.TOEPLITZ_EINSUM:
            dense_fft = s_fft
            assert dense_fft is not None

            def contract(x, y):
                np.einsum("kij,kjp->kip", dense_fft, x, optimize=True, out=y)

        else:
            raise ValueError(f"Unknown matvec method: {matvec_method!r}")

        x_pad = np.zeros((nc, k), dtype=np.complex128)
        x_hat = np.empty((nc, k, 1), dtype=np.complex128)
        y_hat = np.empty((nc, k, 1), dtype=np.complex128)

        def mv(x):
            x_pad[:k] = x.reshape(k, k)
            np.fft.fft(x_pad, axis=0, out=x_hat[..., 0])
            contract(x_hat, y_hat)
            return np.fft.ifft(y_hat[..., 0], axis=0)[:k].ravel()

        s_op = LinearOperator(
            (k * k, k * k), dtype=np.complex128, matvec=mv, rmatvec=mv
        )

    if precond_method is PrecondMethod.NONE:
        m_op = LinearOperator(
            (k * k, k * k),
            dtype=np.complex128,
            matvec=lambda x: x.copy(),
            rmatvec=lambda x: x.copy(),
        )
    elif precond_method is PrecondMethod.INCOMPLETE_CHOLESKY:
        assert stencil is not None
        factor = _get_ic0_factor(stencil)
        upper = factor.conj().T.tocsr()

        def precon(r):
            y = spsolve_triangular(factor, r, lower=True)
            return spsolve_triangular(upper, y, lower=False)

        m_op = LinearOperator(
            (k * k, k * k),
            dtype=np.complex128,
            matvec=precon,
            rmatvec=precon,
        )
    else:
        if precond_method is PrecondMethod.CIRCULANT_DENSE:
            dense_fft = s_fft
            assert dense_fft is not None
            cho_dense = np.linalg.cholesky(dense_fft)

            def solve_precon(r):
                _chol_solve_batch(cho_dense, r)

        else:
            band_fft = s_band
            assert band_fft is not None
            cho_band = np.stack(
                [
                    cholesky_banded(band, lower=True, check_finite=False)
                    for band in band_fft
                ]
            )

            def solve_precon(r):
                for mode in range(nc):
                    r[mode, :, 0] = cho_solve_banded(
                        (cho_band[mode], True),
                        r[mode, :, 0],
                        check_finite=False,
                    )

        r_pad = np.zeros((nc, k), dtype=np.complex128)
        r_hat = np.empty((nc, k, 1), dtype=np.complex128)

        def precon(r):
            r_pad[:k] = r.reshape(k, k)
            np.fft.fft(r_pad, axis=0, out=r_hat[..., 0])
            solve_precon(r_hat)
            return np.fft.ifft(r_hat[..., 0], axis=0)[:k].ravel()

        m_op = LinearOperator(
            (k * k, k * k),
            dtype=np.complex128,
            matvec=precon,
            rmatvec=precon,
        )

    return s_op, m_op
