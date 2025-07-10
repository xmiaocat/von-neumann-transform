from enum import Enum, auto


class BasisMethod(Enum):
    DIRECT = auto()
    FACTORISE = auto()
    FFT = auto()


class MatVecMethod(Enum):
    DIRECT = auto()
    TOEPLITZ_MATMUL = auto()
    TOEPLITZ_EINSUM = auto()
    TOEPLITZ_HANKEL = auto()


class SolverMethod(Enum):
    DIRECT = auto()
    CG = auto()
    BICGSTAB = auto()
    LGMRES = auto()
