from pathlib import Path
import numpy as np
import pytest

from von_neumann_transform.basis import _get_grid

DATA_DIR = Path(__file__).parent / "data"

NPOINTS_LIST = [
    4,
    16,
    64,
    256,
    1024,
    4096,
    16384,
    65536,
    262144,
    1048576,
]
NPOINTS_LIST_XFAIL = [32, 128, 512, 2048]
RTOL = 1e-12
ATOL = 1e-14


@pytest.fixture(scope="module")
def ref_grid():
    with np.load(DATA_DIR / "example_grids.npz") as data:
        k_list = data["k_list"]
        assert [k**2 for k in k_list] == NPOINTS_LIST

        w_min_list = data["w_min_list"]
        w_max = data["w_max"]

        ref_dict = {}
        for k, v in data.items():
            try:
                int(k.split("_")[-1])
            except ValueError:
                continue
            ref_dict[k] = v

    return w_min_list, w_max, ref_dict


@pytest.mark.parametrize("npoints", NPOINTS_LIST)
def test_grid_examples(ref_grid, npoints):
    w_min_list, w_max, ref_dict = ref_grid

    w_grid_list = []
    t_span_list = []
    w_n_arr_list = []
    t_n_arr_list = []
    k_list = []
    alpha_list = []

    for w_min in w_min_list:
        w_grid, t_span, w_n_arr, t_n_arr, k, alpha = _get_grid(
            npoints,
            w_min,
            w_max,
        )
        w_grid_list.append(w_grid)
        t_span_list.append(t_span)
        w_n_arr_list.append(w_n_arr)
        t_n_arr_list.append(t_n_arr)
        k_list.append(k)
        alpha_list.append(alpha)

    np.testing.assert_allclose(
        w_grid_list,
        ref_dict[f"w_grid_{npoints}"],
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        t_span_list,
        ref_dict[f"t_span_{npoints}"],
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        w_n_arr_list,
        ref_dict[f"w_n_arr_{npoints}"],
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        t_n_arr_list,
        ref_dict[f"t_n_arr_{npoints}"],
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_equal(
        k_list,
        ref_dict[f"k_{npoints}"].astype(int),
    )
    np.testing.assert_allclose(
        alpha_list,
        ref_dict[f"alpha_{npoints}"],
        rtol=RTOL,
        atol=ATOL,
    )


@pytest.mark.parametrize("npoints", NPOINTS_LIST_XFAIL)
def test_example_grids_xfail(ref_grid, npoints):
    w_min_list, w_max, ref_dict = ref_grid
    for w_min in w_min_list:
        with pytest.raises(ValueError):
            _get_grid(npoints, w_min, w_max)
