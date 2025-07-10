import numpy as np


def _get_grid(
    npoints: int,
    w_min: float,
    w_max: float,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, int, float]:

    # trimmed grid for signal in the frequency domain
    w_grid, dw_grid = np.linspace(
        w_min,
        w_max,
        npoints,
        retstep=True,
    )

    # grid in the von Neumann plane
    w_span = w_grid.max() - w_grid.min()
    t_span = 2.0 * np.pi / dw_grid
    k = round(np.sqrt(npoints))
    if k**2 != npoints:
        raise ValueError(
            "Number of points must be a perfect square "
            "(k^2) for the von Neumann transform."
        )

    dw = w_span / k
    dt = t_span / k
    w_n_arr = w_min + (np.arange(k) + 0.5) * dw
    t_n_arr = -t_span / 2.0 + (np.arange(k) + 0.5) * dt

    # width of the basis functions
    alpha = t_span / (2.0 * w_span)

    return w_grid, t_span, w_n_arr, t_n_arr, k, alpha


def _evaluate_basis_functions(
    w_grid: np.ndarray,
    w_n_arr: np.ndarray,
    t_n_arr: np.ndarray,
    alpha: float,
) -> np.ndarray:
    k = len(w_n_arr)
    norm = (2.0 * alpha / np.pi) ** 0.25

    alpha_nmo = np.zeros((k, k, k * k), dtype=np.complex128)
    for i in range(0, k):
        for j in range(0, k):
            alpha_nmo[i, j] = np.exp(
                -alpha * (w_grid - w_n_arr[i]) ** 2
                - 1.0j * t_n_arr[j] * (w_grid - w_n_arr[i]),
            )
    alpha_nmo *= norm

    return alpha_nmo
