from diffraction_solver import solve_diffraction
from incidence import Incidence
from grating import Grating


if __name__ == '__main__':
    smaller_scale = 6.3
    n_range = [1, 10, 20, 30]
    for n in n_range:
        amps = []
        d_range = range(2, 17)

        incidence = Incidence(angle=10)
        grating = Grating.multiscale_lamellar_random(period=n * smaller_scale, n=n,
                                                     thickness=0.5, max_permittivity=2.1, seed=1)
        for d in d_range:
            x = solve_diffraction(d, d, grating, incidence, accuracy=1e-6)
            amp = abs(x.full(asvector=True)[2 ** (d - 1)])
            amps.append(amp)
            print(n, d, amp)
