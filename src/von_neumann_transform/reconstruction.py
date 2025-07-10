import numpy as np


def _reconstruct_signal(
    q_nm: np.ndarray,
    q_nmo: np.ndarray,
) -> np.ndarray:
    return np.einsum("nmo,nm->o", q_nmo, q_nm, optimize=True)


def _reconstruct_signal_factorise(
    q_nm: np.ndarray,
    w_grid: np.ndarray,
    w_n_arr: np.ndarray,
    t_n_arr: np.ndarray,
    alpha: float,
) -> np.ndarray:
    norm = (2.0 * alpha / np.pi) ** 0.25

    # (k × N): each row n is the Gaussian window centered at w_n_arr[n]
    alpha_w = norm * np.exp(-alpha * np.subtract.outer(w_n_arr, w_grid) ** 2)

    # (k × N): each row m is the time modulation for t_n_arr[m]
    alpha_t = np.exp(-1.0j * np.outer(t_n_arr, w_grid))

    # extra phase shift e^{i t_m w_n}

    # apply the phase shift e^{i t_m w_n}
    phasor = np.exp(1.0j * np.outer(w_n_arr, t_n_arr))
    tmp = q_nm * phasor

    # tmp @ alpha_t is (k × N) with
    # tmp[n, m] = \sum_m q_nm[n, m] * alpha_t[m, o]
    tmp = tmp @ alpha_t

    # sum over n: signal[o] = \sum_n tmp[n, o] * alpha_w[n, o]
    signal = np.sum(tmp * alpha_w, axis=0)

    return signal


def _reconstruct_signal_fft(
    q_nm: np.ndarray,
    w_grid: np.ndarray,
    w_n_arr: np.ndarray,
    t_n_arr: np.ndarray,
    alpha: float,
) -> np.ndarray:
    k, k2 = len(w_n_arr), len(w_grid)
    dw = w_grid[1] - w_grid[0]
    norm = (2.0 * alpha / np.pi) ** 0.25

    # apply the phase shift e^{i t_m w_n}
    phasor = np.exp(1.0j * np.outer(w_n_arr, t_n_arr))
    tmp = q_nm * phasor  # (k, k)

    # correct for global phase shift
    tmp *= np.exp(-1.0j * t_n_arr * w_grid[0])[np.newaxis, :]

    # the von Neumann time grid is coarser than the FFT grid,
    # so each row of f_tmp is embedded into the FFT grid
    t_grid = 2 * np.pi * np.fft.fftfreq(k2, d=dw)  # (k2,)
    idx_cols = np.array(
        [np.abs(t_grid - t).argmin() for t in t_n_arr],
        dtype=int,
    )

    q_tmp = np.zeros((k, k2), dtype=np.complex128)
    q_tmp[:, idx_cols] = tmp  # (k, k2)

    # compute f_tmp[n, o] = \sum_m q_tmp[n, m] * e^{i t_m w_o}
    # via batched FFT
    f_tmp = np.fft.fft(q_tmp, axis=1)

    # apply the Gaussian window
    # signal[o] = \sum_n f_tmp[n, o] * alpha_w[n, o]
    alpha_w = norm * np.exp(-alpha * np.subtract.outer(w_n_arr, w_grid) ** 2)
    signal = np.sum(f_tmp * alpha_w, axis=0)

    return signal
