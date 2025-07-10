import numpy as np
from scipy.linalg import solve_triangular
from scipy.sparse.linalg import LinearOperator
from .methods import MatVecMethod


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
        # expression valid for every element (m, n) inside the current block
        tmp = (
            -0.5 * alpha * (j**2) * dw**2
            - (1.0 / (8.0 * alpha)) * diff2_mat * dt**2
            + 0.5j * ((-j) * dw) * (sum_mat * dt + 2.0 * t_n_arr.min())
        )
        block = np.exp(tmp)
        col_blocks[j] = block

    return col_blocks


def _get_ovlp_linop(
    alpha: float,
    w_n_arr: np.ndarray,
    t_n_arr: np.ndarray,
    matvec_method: MatVecMethod = MatVecMethod.TOEPLITZ_MATMUL,
) -> tuple[LinearOperator, LinearOperator]:
    k = len(w_n_arr)
    nc = 2 * k  # size of the circulant matrix

    s_block = _get_ovlp_block(alpha, w_n_arr, t_n_arr)
    s_circ = np.zeros((nc, k, k), dtype=np.complex128)
    s_circ[:k] = s_block
    s_circ[k + 1 :] = s_block[1:][::-1].transpose((0, 2, 1)).conj()

    s_fft = np.fft.fft(s_circ, axis=0)  # (k, k, 2k)
    s_cho_l = np.linalg.cholesky(s_fft)

    if matvec_method is MatVecMethod.DIRECT:
        raise RuntimeError(
            "DIRECT method for matrix-vector multiplication "
            "can only be used with the get_ovlp_direct method."
        )
    elif matvec_method is MatVecMethod.TOEPLITZ_MATMUL:

        def contract(mat, x, y):
            np.matmul(mat, x, out=y)

    elif matvec_method is MatVecMethod.TOEPLITZ_EINSUM:

        def contract(mat, x, y):
            np.einsum("kij,kjp->kip", mat, x, optimize=True, out=y)

    elif matvec_method is MatVecMethod.TOEPLITZ_HANKEL:
        raise NotImplementedError(
            "TOEPLITZ_HANKEL method for matrix-vector multiplication "
            "is not implemented yet."
        )
    else:
        assert (
            False
        ), f"Unknown matvec method: {matvec_method!r}"  # pragma: no cover

    x_pad = np.zeros((nc, k), dtype=np.complex128)
    x_hat = np.empty((nc, k, 1), dtype=np.complex128)
    y_hat = np.empty((nc, k, 1), dtype=np.complex128)

    r_pad = np.zeros((nc, k), dtype=np.complex128)
    r_hat = np.empty((nc, k, 1), dtype=np.complex128)

    def mv(x):
        x_blocks = x.reshape(k, k)
        x_pad[:k] = x_blocks
        np.fft.fft(x_pad, axis=0, out=x_hat[..., 0])  # (2k, k)

        contract(s_fft, x_hat, y_hat)
        y_blocks = np.fft.ifft(y_hat[..., 0], axis=0)[:k]

        return y_blocks.ravel()

    def precon(r):
        r_blocks = r.reshape(k, k)
        r_pad[:k] = r_blocks
        np.fft.fft(r_pad, axis=0, out=r_hat[..., 0])  # (2k, k)

        _chol_solve_batch(s_cho_l, r_hat)
        z_blocks = np.fft.ifft(r_hat[..., 0], axis=0)[:k]

        return z_blocks.ravel()

    s_op = LinearOperator(
        (k * k, k * k), dtype=np.complex128, matvec=mv, rmatvec=mv
    )
    m_op = LinearOperator((k * k, k * k), dtype=np.complex128, matvec=precon)

    return s_op, m_op
