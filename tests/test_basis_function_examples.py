from pathlib import Path
import numpy as np
import pytest

from von_neumann_transform.basis import _evaluate_basis_functions

DATA_DIR = Path(__file__).parent / "data"

NPOINTS_LIST = [4, 64, 1024]
RTOL = 1e-12
ATOL = 1e-14


@pytest.fixture(scope="module")
def ref_basis():
    with np.load(DATA_DIR / "example_basis_functions.npz") as data:
        npoints_list = data["npoints_list"]
        assert list(npoints_list) == NPOINTS_LIST

        ref_dict = {}
        for k, v in data.items():
            try:
                int(k.split("_")[-1])
            except ValueError:
                continue
            ref_dict[k] = v

    return ref_dict


@pytest.mark.parametrize("npoints", NPOINTS_LIST)
def test_basis_examples(ref_basis, npoints):
    ref_dict = ref_basis

    alpha_nmo_list = []
    for w_grid, w_n_arr, t_n_arr, alpha in zip(
        ref_dict[f"w_grid_{npoints}"],
        ref_dict[f"w_n_arr_{npoints}"],
        ref_dict[f"t_n_arr_{npoints}"],
        ref_dict[f"alpha_{npoints}"],
    ):
        alpha_nmo = _evaluate_basis_functions(
            w_grid,
            w_n_arr,
            t_n_arr,
            alpha,
        )
        alpha_nmo_list.append(alpha_nmo)

    np.testing.assert_allclose(
        alpha_nmo_list,
        ref_dict[f"alpha_nmo_{npoints}"],
        rtol=RTOL,
        atol=ATOL,
    )
