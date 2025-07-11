#!/usr/bin/env python

from pathlib import Path
import numpy as np
from von_neumann_transform.basis import _evaluate_basis_functions

DATA_DIR = Path(__file__).parents[1] / "data"

NPOINTS_LIST = [4, 64, 1024]

with np.load(DATA_DIR / "example_grids.npz") as data:
    grid_dict = {}
    for n in NPOINTS_LIST:
        grid_dict[f"w_grid_{n}"] = data[f"w_grid_{n}"]
        grid_dict[f"w_n_arr_{n}"] = data[f"w_n_arr_{n}"]
        grid_dict[f"t_n_arr_{n}"] = data[f"t_n_arr_{n}"]
        grid_dict[f"alpha_{n}"] = data[f"alpha_{n}"]

basis_dict = {}
for n in NPOINTS_LIST:
    alpha_nmo_list = []
    for w_grid, w_n_arr, t_n_arr, alpha in zip(
        grid_dict[f"w_grid_{n}"],
        grid_dict[f"w_n_arr_{n}"],
        grid_dict[f"t_n_arr_{n}"],
        grid_dict[f"alpha_{n}"],
    ):
        alpha_nmo = _evaluate_basis_functions(
            w_grid,
            w_n_arr,
            t_n_arr,
            alpha,
        )
        alpha_nmo_list.append(alpha_nmo)
    basis_dict[f"alpha_nmo_{n}"] = np.array(alpha_nmo_list)

np.savez(
    DATA_DIR / "example_basis_functions.npz",
    **grid_dict,
    **basis_dict,
    npoints_list=NPOINTS_LIST,
)
