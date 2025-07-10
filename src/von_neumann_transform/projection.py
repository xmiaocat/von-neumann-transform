import numpy as np


def _project_signal(
    alpha_nmo: np.ndarray,
    signal: np.ndarray,
    dw: float,
) -> np.ndarray:
    alpha_nm = (
        np.einsum(
            "nmo,o->nm",
            alpha_nmo.conj(),
            signal,
            optimize=True,
        )
        * dw
    )
    return alpha_nm


def _get_signal_projection_factorise(
    w_grid: np.ndarray,
    w_n_arr: np.ndarray,
    t_n_arr: np.ndarray,
    alpha: float,
    signal: np.ndarray,
) -> np.ndarray:
    dw = w_grid[1] - w_grid[0]
    norm = (2.0 * alpha / np.pi) ** 0.25

    # (k × N): each row n is the Gaussian window centered at w_n_arr[n]
    alpha_w = norm * np.exp(-alpha * np.subtract.outer(w_n_arr, w_grid) ** 2)

    # (k × N): each row m is signal(w) multiplied by
    # the modulation for t_n_arr[m]
    alpha_t = np.exp(1.0j * np.outer(t_n_arr, w_grid)) * signal[np.newaxis, :]

    # alpha_w @ alpha_t.T is (k × k) with
    # alpha_nm[i, m] = \sum_o alpha_w[i, o] * alpha_t[m, o]
    alpha_nm = alpha_w @ alpha_t.T * dw

    # include extra phase shift e^{-i t_m w_n}
    phasor = np.exp(-1.0j * np.outer(w_n_arr, t_n_arr))
    alpha_nm *= phasor

    return alpha_nm


def _get_signal_projection_fft(
    w_grid: np.ndarray,
    w_n_arr: np.ndarray,
    t_n_arr: np.ndarray,
    alpha: float,
    signal: np.ndarray,
) -> np.ndarray:
    k2 = len(w_grid)
    dw = w_grid[1] - w_grid[0]
    norm = (2.0 * alpha / np.pi) ** 0.25

    # transform signal(w) * exp(-alpha * (w - w_i)^2)
    # via batched IFFT
    w_diff2 = np.subtract.outer(w_n_arr, w_grid) ** 2  # (k, k2)
    f_tmp = norm * np.exp(-alpha * w_diff2) * signal[np.newaxis, :]
    f_tmp = np.fft.ifft(f_tmp, axis=1) * (k2 * dw)  # (k, k2)

    # build the full IFFT time grid
    t_grid = 2 * np.pi * np.fft.fftfreq(k2, d=dw)  # (k2,)

    # the von Neumann time grid is coarser than the FFT grid,
    # so the closest frequency bin in the FFT grid is used
    idx_cols = np.array(
        [np.abs(t_grid - t).argmin() for t in t_n_arr],
        dtype=int,
    )  # (k,)

    # slice out the k columns corresponding to t_n_arr
    alpha_nm = f_tmp[:, idx_cols]

    # correct for global phase shift
    alpha_nm *= np.exp(1.0j * t_n_arr * w_grid[0])[np.newaxis, :]

    # apply the final phase correction e^{-i t_m w_n}
    phasor = np.exp(-1.0j * np.outer(w_n_arr, t_n_arr))
    alpha_nm *= phasor

    return alpha_nm
