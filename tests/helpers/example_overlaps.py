#!/usr/bin/env python

from pathlib import Path
import numpy as np
from von_neumann_transform.overlap import _get_ovlp_direct

DATA_DIR = Path(__file__).parents[1] / "data"

NPOINTS_LIST = [4, 64, 1024]

with np.load(DATA_DIR / "example_grids.npz") as data:
    grid_dict = {}
    for n in NPOINTS_LIST:
        grid_dict[f"w_n_arr_{n}"] = data[f"w_n_arr_{n}"]
        grid_dict[f"t_n_arr_{n}"] = data[f"t_n_arr_{n}"]
        grid_dict[f"alpha_{n}"] = data[f"alpha_{n}"]
        grid_dict[f"k_{n}"] = data[f"k_{n}"]

ovlp_dict = {}
for n in NPOINTS_LIST:
    ovlp_list = []
    for w_n_arr, t_n_arr, alpha in zip(
        grid_dict[f"w_n_arr_{n}"],
        grid_dict[f"t_n_arr_{n}"],
        grid_dict[f"alpha_{n}"],
    ):
        ovlp = _get_ovlp_direct(alpha, w_n_arr, t_n_arr)
        ovlp_list.append(ovlp)
    ovlp_dict[f"ovlp_{n}"] = np.array(ovlp_list)

np.savez(
    DATA_DIR / "example_overlaps.npz",
    **grid_dict,
    **ovlp_dict,
    npoints_list=NPOINTS_LIST,
)
