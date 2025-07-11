from pathlib import Path
import numpy as np
from scipy.sparse.linalg import eigs
import pytest

from von_neumann_transform.basis import _get_grid
from von_neumann_transform.overlap import (
    _get_ovlp_direct,
    _get_ovlp_block,
    _get_ovlp_linop,
)

DATA_DIR = Path(__file__).parent / "data"

NPOINTS_LIST = [4, 64, 1024]
RTOL = 1e-12
ATOL = 1e-14


@pytest.fixture(scope="module")
def ref_overlaps():
    rng = np.random.default_rng(42)

    alpha_list = []
    w_n_arr_list = []
    t_n_arr_list = []
    ovlp_ref = []
    for npoints in NPOINTS_LIST:
        w_min, w_max = 1000.0 * np.sort(rng.random(2))
        if w_max - w_min < 1e-3:
            w_max += 1e-3
        w_grid, t_span, w_n_arr, t_n_arr, k, alpha = _get_grid(
            npoints,
            w_min,
            w_max,
        )
        alpha_list.append(alpha)
        w_n_arr_list.append(w_n_arr)
        t_n_arr_list.append(t_n_arr)

        ovlp = _get_ovlp_direct(alpha, w_n_arr, t_n_arr)
        ovlp_ref.append(ovlp)

    return alpha_list, w_n_arr_list, t_n_arr_list, ovlp_ref


def test_overlap_block_consistency(ref_overlaps):
    alpha_list, w_n_arr_list, t_n_arr_list, ovlp_ref = ref_overlaps
    for alpha, w_n_arr, t_n_arr, ovlp in zip(
        alpha_list,
        w_n_arr_list,
        t_n_arr_list,
        ovlp_ref,
    ):
        k = len(w_n_arr)
        ovlp_block = _get_ovlp_block(alpha, w_n_arr, t_n_arr)

        s_col0 = np.concatenate(ovlp_block)
        s_row0 = s_col0.T.conj()
        s_toeplitz = np.zeros_like(ovlp)
        for i in range(k):
            for j in range(k):
                diff = j - i
                if diff >= 0:
                    s_toeplitz[i * k : (i + 1) * k, j * k : (j + 1) * k] = (
                        s_row0[:, diff * k : (diff + 1) * k]
                    )
                else:
                    diff = -diff
                    s_toeplitz[i * k : (i + 1) * k, j * k : (j + 1) * k] = (
                        s_col0[diff * k : (diff + 1) * k, :]
                    )

        np.testing.assert_allclose(
            s_toeplitz,
            ovlp,
            rtol=RTOL,
            atol=ATOL,
        )


def test_overlap_linop_consistency(ref_overlaps):
    alpha_list, w_n_arr_list, t_n_arr_list, ovlp_ref = ref_overlaps
    for alpha, w_n_arr, t_n_arr, ovlp in zip(
        alpha_list,
        w_n_arr_list,
        t_n_arr_list,
        ovlp_ref,
    ):
        k = len(w_n_arr)
        s_op, _ = _get_ovlp_linop(alpha, w_n_arr, t_n_arr)

        for _ in range(10):
            x_re = np.random.default_rng().random(k * k)
            x_im = np.random.default_rng().random(k * k)
            x = x_re + 1j * x_im
            s_op_x = s_op.matvec(x)

            s_x_ref = ovlp @ x

            np.testing.assert_allclose(
                s_op_x,
                s_x_ref,
                rtol=RTOL,
                atol=ATOL,
            )


@pytest.mark.skip(reason="Difficult convergence in eigs")
def test_overlap_precon_consistency(ref_overlaps):
    alpha_list, w_n_arr_list, t_n_arr_list, ovlp_ref = ref_overlaps
    for alpha, w_n_arr, t_n_arr, ovlp in zip(
        alpha_list,
        w_n_arr_list,
        t_n_arr_list,
        ovlp_ref,
    ):
        s_op, m_op = _get_ovlp_linop(alpha, w_n_arr, t_n_arr)

        e_ref, _ = np.linalg.eigh(ovlp)
        assert np.all(e_ref > 0.0)
        e_ref_min, e_ref_max = e_ref[0], e_ref[-1]

        p_op = m_op @ s_op
        e_precon_sm, _ = eigs(p_op, k=1, which="SM")
        e_precon_lm, _ = eigs(p_op, k=1, which="LM")
        e_precon_min = np.abs(e_precon_sm[0])
        e_precon_max = np.abs(e_precon_lm[0])

        print(e_ref_min, e_ref_max)
        print(e_precon_min, e_precon_max)

        assert e_ref_max / e_ref_min >= e_precon_max / e_precon_min
