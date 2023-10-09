from diffraction_solver import solve_diffraction
from incidence import Incidence
from grating import Grating
import logging
import sys


logging.basicConfig(filename='multiscale.log', filemode='w', format='%(message)s')

try:
    if __name__ == '__main__':
        print('n range:', end='\t')
        n_range_str = list(map(int, input().split(',')))
        print('d range from:', end='\t')
        d_from_str = int(input())
        print('d range to:', end='\t')
        d_to_str = int(input())
        print('thickness:', end='\t')
        thickness_str = float(input())
        print('accuracy:', end='\t')
        accuracy_str = float(input())
    else:
        n_range = list(map(int, sys.argv[1].split(',')))
        d_from = int(sys.argv[2])
        d_to = int(sys.argv[3])
        d_range = range(d_from, d_to)
        thickness = float(sys.argv[4])
        accuracy = float(sys.argv[5])

    smaller_scale = 6.3

    for d in d_range:
        for n in n_range:

            incidence = Incidence(angle=10)
            grating = Grating.multiscale_lamellar_random(period=n * smaller_scale, n=n,
                                                         thickness=thickness, max_permittivity=2.1, seed=1)
            x = solve_diffraction(d, d, grating, incidence, accuracy=accuracy)
            amp = abs(x.full(asvector=True)[2 ** (d - 1)])
            # print(n, d, amp)
            logging.warning(f'{n} {d} {amp}')
except Exception as e:
    logging.warning(e)
