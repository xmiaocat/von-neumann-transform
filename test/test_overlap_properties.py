from pathlib import Path
import numpy as np
import pytest

from von_neumann_transform.basis import _get_grid
from von_neumann_transform.overlap import _get_ovlp_direct

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


def test_overlap_hermitian(ref_overlaps):
    alpha_list, w_n_arr_list, t_n_arr_list, ovlp_ref = ref_overlaps
    for alpha, w_n_arr, t_n_arr, ovlp in zip(
        alpha_list,
        w_n_arr_list,
        t_n_arr_list,
        ovlp_ref,
    ):
        # Check that the overlap is Hermitian
        np.testing.assert_allclose(
            ovlp,
            ovlp.conj().T,
            rtol=RTOL,
            atol=ATOL,
        )


def test_overlap_block_toeplitz(ref_overlaps):
    alpha_list, w_n_arr_list, t_n_arr_list, ovlp_ref = ref_overlaps
    for alpha, w_n_arr, t_n_arr, ovlp in zip(
        alpha_list,
        w_n_arr_list,
        t_n_arr_list,
        ovlp_ref,
    ):
        k = len(w_n_arr)
        ovlp_block = ovlp.reshape((k, k, k, k)).swapaxes(1, 2)
        np.testing.assert_allclose(
            ovlp_block[1:, 1:],
            ovlp_block[:-1, :-1],
            rtol=RTOL,
            atol=ATOL,
        )


def test_overlap_pmtbs(ref_overlaps):
    alpha_list, w_n_arr_list, t_n_arr_list, ovlp_ref = ref_overlaps
    for alpha, w_n_arr, t_n_arr, ovlp in zip(
        alpha_list,
        w_n_arr_list,
        t_n_arr_list,
        ovlp_ref,
    ):
        k = len(w_n_arr)
        ovlp_block = ovlp.reshape((k, k, k, k)).swapaxes(1, 2)
        for i in range(k):
            for j in range(k):
                tmp_magnitude = np.abs(ovlp_block[i, j])
                np.testing.assert_allclose(
                    tmp_magnitude[1:, 1:],
                    tmp_magnitude[:-1, :-1],
                    rtol=RTOL,
                    atol=ATOL,
                )

                z1 = ovlp_block[i, j][1:, :-1]
                z2 = ovlp_block[i, j][:-1, 1:]
                mag_mask = (np.abs(z1) > ATOL) & (np.abs(z2) > ATOL)
                phase_diff = np.angle(z1[mag_mask] / z2[mag_mask])
                np.testing.assert_allclose(
                    np.abs(phase_diff),
                    0.0,
                    rtol=RTOL,
                    atol=ATOL,
                )
