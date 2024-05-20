import numpy as np
import tt

from tt_utils import map_on_modes
from incidence import Incidence
from grating import Grating


if __name__ == '__main__':
    # kx, kz and diagonal of matrix P
    kx0 = 0.1
    period = 6.28

    d = 15
    n_np = np.arange(2 ** d) - 2 ** (d - 1)
    n = tt.xfun(2, d) - 2 ** (d - 1) * tt.ones(2, d)

    f1 = lambda x: kx0 + x * 2 * np.pi / period
    f2 = lambda x: np.sqrt(1 - f1(x) ** 2, dtype=complex)
    # f3 = lambda x: 1 / f2(x)
    f = f2

    # grating = Grating.multiscale_lamellar_random(2 * np.pi, 1, 0.3, 2.1, seed=1)
    # f = grating.permittivity_fourier

    y_np = f(n_np)
    # y = tt.multifuncrs([n], f, eps=1e-6, y0=tt.ones(2, d), verb=0)
    # print(np.linalg.norm(y_np - y.full(asvector=True)))

    y_svd = tt.vector(y_np)
    y_svd = y_svd.round(1e-6)
    print(np.linalg.norm(y_np - y_svd.full(asvector=True)))

    # exponent in matrices R and T
    dl = 2
    h = 0.5
    dh = h / (2 ** dl)

    kz = y_svd
    powers = tt.xfun(2, dl)
    powers = tt.Toeplitz(powers, kind='L').T
    powers *= 1j * dh
    diag_blocks = tt.kron(kz, powers.tt)
    diag_blocks_np = np.exp(diag_blocks.full(asvector=True))
    diag_blocks = tt.multifuncrs([diag_blocks], np.exp, y0=tt.ones(diag_blocks.n), eps=1e-6, verb=0, nswp=20)
    print(np.linalg.norm(diag_blocks_np - diag_blocks.full(asvector=True)))

    powers_positive = tt.ones(2, dl) * (2 ** dl - 0.5) - tt.xfun(2, dl)
    powers_positive *= 1j * dh
    powers_negative = tt.ones(2, dl) * 0.5 + tt.xfun(2, dl)
    powers_negative *= 1j * dh
    diag_blocks_positive = tt.kron(kz, powers_positive)
    diag_blocks_negative = tt.kron(kz, powers_negative)
    diag_blocks_positive_np = np.exp(diag_blocks_positive.full(asvector=True))
    diag_blocks_negative_np = np.exp(diag_blocks_negative.full(asvector=True))
    diag_blocks_positive = tt.multifuncrs([diag_blocks_positive], np.exp, y0=tt.ones(diag_blocks_positive.n), eps=1e-6, verb=0, nswp=20)
    diag_blocks_negative = tt.multifuncrs([diag_blocks_negative], np.exp, y0=tt.ones(diag_blocks_negative.n), eps=1e-6, verb=0, nswp=20)
    print(np.linalg.norm(diag_blocks_positive_np - diag_blocks_positive.full(asvector=True)))
    print(np.linalg.norm(diag_blocks_negative_np - diag_blocks_negative.full(asvector=True)))

    # fourier coefficients and matrix V

