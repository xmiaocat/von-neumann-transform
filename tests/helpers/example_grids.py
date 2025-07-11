#!/usr/bin/env python

from pathlib import Path
import numpy as np
from von_neumann_transform.basis import _get_grid

DATA_DIR = Path(__file__).parents[1] / "data"

K_LIST = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
W_MIN_LIST = [0.0, 1.0, 2.0, 3.0, 4.0]
W_MAX = 5.0

array_dict = {}
for k in K_LIST:
    n = k**2
    w_grid_list = []
    t_span_list = []
    w_n_arr_list = []
    t_n_arr_list = []
    k_list = []
    alpha_list = []

    for w_min in W_MIN_LIST:
        w_grid, t_span, w_n_arr, t_n_arr, k, alpha = _get_grid(
            n,
            w_min,
            W_MAX,
        )
        w_grid_list.append(w_grid)
        t_span_list.append(t_span)
        w_n_arr_list.append(w_n_arr)
        t_n_arr_list.append(t_n_arr)
        k_list.append(k)
        alpha_list.append(alpha)

    array_dict[f"w_grid_{n}"] = w_grid_list
    array_dict[f"t_span_{n}"] = t_span_list
    array_dict[f"w_n_arr_{n}"] = w_n_arr_list
    array_dict[f"t_n_arr_{n}"] = t_n_arr_list
    array_dict[f"k_{n}"] = k_list
    array_dict[f"alpha_{n}"] = alpha_list

np.savez(
    DATA_DIR / "example_grids.npz",
    **array_dict,
    k_list=K_LIST,
    w_min_list=W_MIN_LIST,
    w_max=W_MAX,
)
