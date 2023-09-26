from diffraction_solver import solve_diffraction
from incidence import Incidence
from grating import Grating
import logging


logging.basicConfig(filename='multiscale.log', filemode='w', format='%(message)s')

try:
    smaller_scale = 6.3
    n_range = [100]
    d_range = range(2, 4)

    for d in d_range:
        for n in n_range:

            incidence = Incidence(angle=10)
            grating = Grating.multiscale_lamellar_random(period=n * smaller_scale, n=n,
                                                         thickness=0.25, max_permittivity=2.1, seed=1)
            x = solve_diffraction(d, d, grating, incidence, accuracy=1e-5)
            amp = abs(x.full(asvector=True)[2 ** (d - 1)])
            # print(n, d, amp)
            logging.warning(f'{n} {d} {amp}')
except Exception as e:
    logging.warning(e)
