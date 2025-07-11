import numpy as np
import pytest

from von_neumann_transform.basis import _get_grid, _evaluate_basis_functions
from von_neumann_transform.projection import (
    _project_signal,
    _get_signal_projection_factorise,
    _get_signal_projection_fft,
)

NPOINTS_LIST = [16, 256, 4096]
RTOL = 1e-10
ATOL = 1e-12


@pytest.fixture(scope="module")
def ref_projection():
    rng = np.random.default_rng(42)

    w_grid_list = []
    w_n_arr_list = []
    t_n_arr_list = []
    alpha_list = []
    signal_list = []
    projections_ref = []
    for npoints in NPOINTS_LIST:
        w_min, w_max = 1000.0 * np.sort(rng.random(2))
        if w_max - w_min < 1e-3:
            w_max += 1e-3
        w_grid, t_span, w_n_arr, t_n_arr, k, alpha = _get_grid(
            npoints,
            w_min,
            w_max,
        )
        w_grid_list.append(w_grid)
        w_n_arr_list.append(w_n_arr)
        t_n_arr_list.append(t_n_arr)
        alpha_list.append(alpha)

        alpha_nmo = _evaluate_basis_functions(
            w_grid,
            w_n_arr,
            t_n_arr,
            alpha,
        )

        signal = rng.random(npoints) * np.exp(1.0j * rng.random(npoints))
        signal_list.append(signal)

        dw = w_grid[1] - w_grid[0]
        proj_ref = _project_signal(alpha_nmo, signal, dw)
        projections_ref.append(proj_ref)

    return (
        w_grid_list,
        w_n_arr_list,
        t_n_arr_list,
        alpha_list,
        signal_list,
        projections_ref,
    )


@pytest.mark.parametrize(
    "proj_func",
    [
        _get_signal_projection_factorise,
        _get_signal_projection_fft,
    ],
)
def test_projection_method_consistency(ref_projection, proj_func):
    (
        w_grid_list,
        w_n_arr_list,
        t_n_arr_list,
        alpha_list,
        signal_list,
        projections_ref,
    ) = ref_projection

    for npoints, w_grid, w_n_arr, t_n_arr, alpha, signal, proj_ref in zip(
        NPOINTS_LIST,
        w_grid_list,
        w_n_arr_list,
        t_n_arr_list,
        alpha_list,
        signal_list,
        projections_ref,
    ):
        proj = proj_func(w_grid, w_n_arr, t_n_arr, alpha, signal)
        np.testing.assert_allclose(
            proj,
            proj_ref,
            rtol=RTOL,
            atol=ATOL,
        )
