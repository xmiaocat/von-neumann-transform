from pathlib import Path
import numpy as np
import pytest

from von_neumann_transform.overlap import _get_ovlp_direct, _get_ovlp_block

DATA_DIR = Path(__file__).parent / "data"

NPOINTS_LIST = [4, 64, 1024]
RTOL = 1e-12
ATOL = 1e-14


@pytest.fixture(scope="module")
def ref_overlaps():
    with np.load(DATA_DIR / "example_overlaps.npz") as data:
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
def test_overlap_direct_examples(ref_overlaps, npoints):
    ref_dict = ref_overlaps

    ovlp_list = []
    for alpha, w_n_arr, t_n_arr in zip(
        ref_dict[f"alpha_{npoints}"],
        ref_dict[f"w_n_arr_{npoints}"],
        ref_dict[f"t_n_arr_{npoints}"],
    ):
        ovlp = _get_ovlp_direct(alpha, w_n_arr, t_n_arr)
        ovlp_list.append(ovlp)

    np.testing.assert_allclose(
        ovlp_list,
        ref_dict[f"ovlp_{npoints}"],
        rtol=RTOL,
        atol=ATOL,
    )


@pytest.mark.parametrize("npoints", NPOINTS_LIST)
def test_overlap_block_examples(ref_overlaps, npoints):
    ref_dict = ref_overlaps

    ovlp_col_list = []
    ref_ovlp_col_list = []
    for alpha, w_n_arr, t_n_arr, k, s_ref in zip(
        ref_dict[f"alpha_{npoints}"],
        ref_dict[f"w_n_arr_{npoints}"],
        ref_dict[f"t_n_arr_{npoints}"],
        ref_dict[f"k_{npoints}"],
        ref_dict[f"ovlp_{npoints}"],
    ):
        ovlp_block = _get_ovlp_block(alpha, w_n_arr, t_n_arr)
        ovlp_col = np.concatenate(ovlp_block)
        ovlp_col_list.append(ovlp_col)

        ref_ovlp_col = s_ref[:, :k]
        ref_ovlp_col_list.append(ref_ovlp_col)

    np.testing.assert_allclose(
        ovlp_col_list,
        ref_ovlp_col_list,
        rtol=RTOL,
        atol=ATOL,
    )
