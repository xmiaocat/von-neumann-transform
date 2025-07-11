# Von Neumann Transform

[![Tests](https://github.com/xmiaocat/von-neumann-transform/actions/workflows/tests.yml/badge.svg)](https://github.com/xmiaocat/von-neumann-transform/actions/workflows/tests.yml)
[![Flake8](https://github.com/xmiaocat/von-neumann-transform/actions/workflows/flake8.yml/badge.svg)](https://github.com/xmiaocat/von-neumann-transform/actions/workflows/flake8.yml)
[![Mypy](https://github.com/xmiaocat/von-neumann-transform/actions/workflows/mypy.yml/badge.svg)](https://github.com/xmiaocat/von-neumann-transform/actions/workflows/mypy.yml)
[![Codecov](https://codecov.io/gh/xmiaocat/von-neumann-transform/branch/main/graph/badge.svg)](https://codecov.io/gh/xmiaocat/von-neumann-transform)
[![PyPI version](https://img.shields.io/pypi/v/von-neumann-transform.svg)](https://pypi.org/project/von-neumann-transform/)

A Python package for efficient computation of the von Neumann representation 
of a signal given in the frequency domain.
The von Neumann representation is a joint time-frequency representation 
defined in
- S. Fechner, F. Dimler, T. Brixner, G. Gerber, J. Tannor,
  *Opt. Express* **2007**, *15*, 15387–15401.
- F. Dimler, S. Fechner, A. Rodenberg, T. Brixner, J. Tannor,
  *New J. Phys.* **2009**, *11*, 105052.

## TODO
- [ ] Add consistency tests for solvers.
- [ ] Add consistency tests for inverse transform.

## Features
- Grid generation: Build uniform time-frequency grids in the
  von Neumann plane.
- Signal projection: Compute the projection of the frequency-domain 
  signal onto the von Neumann basis functions *via*:
  - Direct method: precompute and store the basis functions.
  - Factorisation method: sequentially compute projections using
    a factorisation of the basis.
  - FFT-based method: compute the projection with the help of FFT.
- Overlap assembly & solvers: Solve for von Neumann coefficients
  accounting for basis overlap using:
  - Direct solve: assemble the overlap matrix and apply a
    direct linear solver.
  - Iterative solve: employ a matrix-vector operator and
    iterative solver routines.
- Signal reconstruction: Reconstruct the original frequency-domain
  signal from von Neumann coefficients.
- Type-safe API: Enums (`BasisMethod`, `MatVecMethod`, `SolverMethod`)
  select algorithms, all functions and methods include type hints.

## Installation

Install from PyPI:
```bash
pip install von-neumann-transform
```

Install the latest development version from GitHub:
```bash
pip install git+https://github.com/xmiaocat/von-neumann-transform.git
```

To install in development mode including dev dependencies, 
clone the repository and run:
```bash
pip install -e ".[dev]"
```

## Quickstart
```python
import numpy as np
from von_neumann_transform import VonNeumannTransform

NPOINTS = 4096  # length of the signal
W_MIN = 0.0  # minimum angular frequency
W_MAX = 5.0  # maximum angular frequency

# Your signal in the frequency domain
signal = np.random.rand(NPOINTS) + 1.0j * np.random.rand(NPOINTS)

# Create a Von Neumann Transform instance
vnt = VonNeumannTransform(NPOINTS, W_MIN, W_MAX)

# Compute the von Neumann representation of the signal
q_nm = vnt.transform(signal)

# Reconstruct the original signal from the von Neumann coefficients
signal_recon = vnt.inverse_transform(q_nm)
```

## Algorithmic Details
This section provides an overview of each method and its computational complexity.
Suppose the signal has length $N$. Then $k = \sqrt{N}$ is chosen, so the
von Neumann representation is a $k \times k$ matrix.

### Signal Projection
The signal projection is defined as
```math
  \langle \alpha_{\omega_i t_j} | \epsilon \rangle
  = \int_{-\infty}^{\infty} \alpha^*_{\omega_i t_j}(\omega) \epsilon(\omega) \mathrm{d}\omega
```
where $\alpha_{\omega_i t_j}(\omega)$ are the von Neumann basis functions
```math
  \alpha_{\omega_n t_m}(\omega) 
  = \left(\frac{2\alpha}{\pi}\right)^{1/4} 
    \exp \left[ -\alpha (\omega - \omega_n)^2 - \mathrm{i} t_m (\omega - \omega_n) \right]
```
and $\epsilon(\omega)$ is the signal in the frequency domain.

In the discrete world, the basis functions $\alpha_{\omega_n t_m}(\omega)$
are sampled at $N$ points in the frequency domain, and therefore has the
shape `(k, k, N)`. The projection is then computed as
```math
  \alpha_{nm} = \sum_{p=0}^{N-1} \alpha_{\omega_n t_m}(\omega_p) \epsilon(\omega_p)\,.
```
The time complexity of this operation is $\mathcal{O}(k^2 N)$.
The space complexity is also $\mathcal{O}(k^2 N)$ if the basis functions
are precomputed and stored. 

The basis functions can be factorised as
```math
  \begin{align}
  \alpha_{\omega_n t_m}(\omega) 
  &= \left(\frac{2\alpha}{\pi}\right)^{1/4} 
    \exp \left[ -\alpha (\omega - \omega_n)^2 \right] 
    \exp \left[ -\mathrm{i} t_m \omega \right]
    \exp \left[ \mathrm{i} t_m \omega_n \right] \\
  &=: \left(\frac{2\alpha}{\pi}\right)^{1/4} 
    \alpha_{\omega_n}(\omega) \alpha_{t_m}(\omega) \alpha_{\omega_n t_m}\,.
  \end{align}
```
The discretised version of the factors $\alpha_{\omega_n}(\omega)$,
$\alpha_{t_m}(\omega)$ and $\alpha_{\omega_n t_m}$ have the shapes
`(k, N)`, `(k, N)` and `(k, k)` respectively. This way, 
although the time complexity of the projection is still
$\mathcal{O}(k^2 N)$, the space complexity can be reduced to
$\mathcal{O}(k N)$.

The factorisation shows another way to compute the projection.
Because of the factor $\exp \left[ \mathrm{i} t_m \omega_n \right]$,
this projection can be viewed as the (inverse) Fourier transform of the
signal $\epsilon(\omega)$ multiplied by the factor $\alpha_{\omega_n}(\omega)$
with a phase correction afterwards.
This reduces the time complexity to $\mathcal{O}(k N \log N)$.

The computational complexity of all three methods is summarised in the table below:
| Method          | Time Complexity   | Space Complexity |
|-----------------|-------------------|------------------|
| direct (`BasisMethod.DIRECT`) | $\mathcal{O}(k^2 N) = \mathcal{O}(N^2)$ | $\mathcal{O}(k^2 N) = \mathcal{O}(N^2)$ |
| factorisation (`BasisMethod.FACTORISE`) | $\mathcal{O}(k^2 N) = \mathcal{O}(N^2)$ | $\mathcal{O}(k N) = \mathcal{O}(N^{3/2})$ |
| FFT-based (`BasisMethod.FFT`) | $\mathcal{O}(k N \log N) = \mathcal{O}(N^{3/2} \log N)$ | $\mathcal{O}(k N) = \mathcal{O}(N^{3/2})$ |

### Overlap Assembly & Solvers
Since there are in total $k\times k = N$ basis functions, the overlap
matrix
```math
  S_{(n,m),(i,j)} = \sqrt{\frac{2\alpha}{\pi}}
    \exp \left[ -\frac{\alpha}{2}(\omega_n - \omega_i)^2
                -\frac{1}{8\alpha}(t_j - t_m)^2 
                +\frac{\mathrm{i}}{2}(\omega_i - \omega_n)(t_j + t_m) \right]
```
has the dimension $N \times N$. Precomputing it and then solving the
linear system
```math
  \sum_{(i,j)} S_{(n,m),(i,j)} q_{(i,j)} = \langle \alpha_{nm} | \epsilon \rangle
```
directly would have a time complexity of $\mathcal{O}(N^3)$ 
and a space complexity of $\mathcal{O}(N^2)$.
This can become quite expensive for signals with a large number of points,

In practice, the space is often the limiting factor. Therefore, the usual
approach would be to implement a matrix-vector operator
that computes the overlap-matrix-vector product on-the-fly
by contracting each row, then feeds that result into an iterative 
solver like the conjugate gradient method to solve the linear system.
This approach has a time complexity of $\mathcal{O}(N^2)$ per iteration
and a space complexity of $\mathcal{O}(N)$, which is much more feasible
for large signals.
However, the repeated evaluation of the overlap matrix and
the matrix-vector product becomes very time-consuming.
This method is not implemented in this package.

Luckily, the overlap matrix has some structures that can be exploited.
This matrix is actually a **H**ermitian **P**ositive **D**efinite 
**B**lock **T**oeplitz matrix with **T**oeplitz-**H**ankel
Hadamard Product **B**locks (HPDBTTHB).
The block Toeplitz structure means that we only need the first
block row and the first block column of the matrix to construct the
entire matrix. Because of the hermiticity, we even only need the
first block column of the dimension $N \times k$.
Even better, the block Toeplitz structure allows one to efficiently compute
the matrix-vector product by embedding the the matrix into a larger
$2N \times 2N$ block circulant matrix and then using batch FFTs to compute
the product in $\mathcal{O}(k^2 \log k)$ time.
This way, the expensive "ordinary" matrix-vector product only needs to be
computed on the much smaller blocks, thus reducing the time complexity
of the matrix-vector product to $\mathcal{O}(k^3)$.
Overall, the time complexity of this method is
$\mathcal{O}(k^3)$ and the space complexity is $\mathcal{O}(k^3)$.

In theory, the Toeplitz-Hankel Hadamard product structure of the blocks
can be exploited further to reduce the time complexity of the 
matrix-vector product of blocks to $\mathcal{O}(k^2 \log k)$,
but this is not implemented in this package yet.

To accelerate the convergence, a circulant preconditioner is used 
in the iterative solver, which does not increase the time complexity 
for the iterative part but adds an additional $\mathcal{O}(k^4)$ 
overhead for the preparation of the preconditioner.

The computational complexity of the overlap assembly and solvers is summarised 
in the table below:
| Method          | Time Complexity   | Space Complexity |
|-----------------|-------------------|------------------|
| direct (`MatVecMethod.DIRECT`) | $\mathcal{O}(N^3)$ | $\mathcal{O}(N^2)$ |
| rows of overlap + iterative solver (not implemented) | $\mathcal{O}(m\cdot N^2)$ | $\mathcal{O}(N)$ |
| Toeplitz + iterative solver + preconditioner (`MatVecMethod.TOEPLITZ_MATMUL` or `MatVecMethod.TOEPLITZ_EINSUM`) | $\mathcal{O}(N^2 + m\cdot N^{3/2})$ | $\mathcal{O}(N^{3/2})$ |

The variable $m$ is the number of iterations of the iterative solver.

### Signal Reconstruction
The signal reconstruction is simply the inverse of the signal projection,
and thus the same assortment of methods with the same computational
complexity applies.


## API Reference

### Class `VonNeumannTransform`

`VonNeumannTransform(npoints: int, omega_min: float, omega_max: float)`

Initialise a Von Neumann Transform instance.

- **Parameters:**
  - `npoints`: Number of points in the frequency domain signal.
  - `omega_min`: Minimum angular frequency.
  - `omega_max`: Maximum angular frequency.

```
transform(
    signal: np.ndarray,
    basis_method: BasisMethod = BasisMethod.FFT,
    matvec_method: MatVecMethod = MatVecMethod.TOEPLITZ_MATMUL,
    solver_method: SolverMethod = SolverMethod.CG,
    rtol: float = 1e-10,
    atol: float = 0.0,
    maxiter: int = 1000,
) -> np.ndarray
```

Computes the von Neumann representation of the signal.

- Parameters:
    - signal (np.ndarray): Input signal in the frequency domain.
    - basis_method (BasisMethod): Method to compute the projection
        of the signal onto the basis functions.
        Possible values are:
        - `BasisMethod.DIRECT`: Directly compute the projection
          by precomputing and storing the basis functions.
        - `BasisMethod.FACTORISE`: Use the factorisation of the basis
          functions to compute the projection.
        - `BasisMethod.FFT`: Use the FFT to compute the projection.
    - matvec_method (MatVecMethod): Method to compute the overlap matrix.
        Possible values are:
        - `MatVecMethod.DIRECT`: Directly assemble the overlap matrix.
           Requires `SolverMethod.DIRECT`.
        - `MatVecMethod.TOEPLITZ_MATMUL`: Use the Toeplitz structure
          to compute the matrix-vector product.
        - `MatVecMethod.TOEPLITZ_EINSUM`: Use the Toeplitz structure
          to compute the matrix-vector product with einsum.
        The latter two methods require an iterative solver
        (`SolverMethod.CG`, `SolverMethod.BICGSTAB`, or `SolverMethod.LGMRES`).
    - solver_method (SolverMethod): Method to solve the linear system.
        - `SolverMethod.DIRECT`: Use a direct solver. Requires
          `MatVecMethod.DIRECT`.
        - `SolverMethod.CG`: Use the conjugate gradient method.
        - `SolverMethod.BICGSTAB`: Use the biconjugate gradient
          stabilised method.
        - `SolverMethod.LGMRES`: Use the LGMRES method.
        The iterative solvers require
        `MatVecMethod.TOEPLITZ_MATMUL` or `MatVecMethod.TOEPLITZ_EINSUM`.
    - rtol, atol (float): Relative and absolute tolerances for the
        iterative solver.
    - maxiter (int): Maximum number of iterations for the iterative
        solver.

- Returns:
    - q_nm (np.ndarray): Von Neumann coefficients, solution of the
        linear system S * q_nm = alpha_nm.

```
inverse_transform(
    q_nm: np.ndarray,
    method: BasisMethod = BasisMethod.FFT,
) -> np.ndarray
```

Reconstructs the signal from the von Neumann coefficients.

- Parameters:
    - q_nm (np.ndarray): Von Neumann coefficients.
    - method (BasisMethod): Method to compute the inverse projection.
        Possible values are:
        - `BasisMethod.DIRECT`: Directly reconstruct the signal
          by precomputing and storing the basis functions.
        - `BasisMethod.FACTORISE`: Use the factorisation of the basis
          functions to reconstruct the signal.
        - `BasisMethod.FFT`: Use the FFT to reconstruct the signal.
- Returns:
    - signal (np.ndarray): Reconstructed signal in the frequency domain.

### Enums

`BasisMethod`
Selects how basis functions are handled in the projection and reconstruction:
- `BasisMethod.DIRECT`: Precompute and store the basis functions.
- `BasisMethod.FACTORISE`: Use the factorisation of the basis functions.
- `BasisMethod.FFT`: Use the FFT to compute the projection and reconstruction.

`MatVecMethod`
Selects how the overlap operator is applied:
- `MatVecMethod.DIRECT`: Directly assemble the overlap matrix and multiply.
- `MatVecMethod.TOEPLITZ_MATMUL`: Use the Toeplitz structure to compute
  the matrix-vector product.
- `MatVecMethod.TOEPLITZ_EINSUM`: Use the Toeplitz structure to compute
  the matrix-vector product with einsum.
- `MatVecMethod.TOEPLITZ_HANKEL`: Use the Toeplitz-Hankel structure
  to compute the matrix-vector product. **Not implemented yet.**

`SolverMethod`
Selects the linear solver for overlap inversion:
- `SolverMethod.DIRECT`: Use a direct solver.
- `SolverMethod.CG`: Use the conjugate gradient method.
- `SolverMethod.BICGSTAB`: Use the biconjugate gradient stabilised method.
- `SolverMethod.LGMRES`: Use the LGMRES method.

## License
Distributed under the Apache License 2.0.
See `LICENSE` for more information.

