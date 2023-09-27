from diffraction_solver import solve_diffraction
from incidence import Incidence
from grating import Grating
import logging


logging.basicConfig(filename='multiscale.log', filemode='w', format='%(message)s')

try:
    smaller_scale = 6.3

    print('n range:', end='\t\t')
    n_range = list(map(int, input().split(',')))

    print('d range from:', end='\t')
    d_from = int(input())
    print('d range to:', end='\t')
    d_to = int(input())
    d_range = range(d_from, d_to)

    print('thickness:', end='\t')
    thickness = float(input())

    for d in d_range:
        for n in n_range:

            incidence = Incidence(angle=10)
            grating = Grating.multiscale_lamellar_random(period=n * smaller_scale, n=n,
                                                         thickness=thickness, max_permittivity=2.1, seed=1)
            x = solve_diffraction(d, d, grating, incidence, accuracy=1e-5)
            amp = abs(x.full(asvector=True)[2 ** (d - 1)])
            # print(n, d, amp)
            logging.warning(f'{n} {d} {amp}')
except Exception as e:
    logging.warning(e)
