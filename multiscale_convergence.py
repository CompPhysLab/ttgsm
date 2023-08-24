from diffraction_solver import solve_diffraction
from incidence import Incidence
from grating import Grating


if __name__ == '__main__':
    smaller_scale = 6.3

    print('Input n')
    n = int(input())

    print('Input d')
    d = int(input())

    incidence = Incidence(angle=10)
    grating = Grating.multiscale_lamellar_random(period=n * smaller_scale, n=n,
                                                 thickness=0.5, max_permittivity=2.1, seed=1)

    x = solve_diffraction(d, d, grating, incidence, accuracy=1e-6, verb=True)
    amp = abs(x.full(asvector=True)[2 ** (d - 1)])
    print(n, d, amp)
