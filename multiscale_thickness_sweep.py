from diffraction_solver import solve_diffraction
from incidence import Incidence
from grating import Grating

import numpy as np
import datetime


wavelength = 6.28

n = 1
eps = 2.1
d_range = range(2, 16)
thickness_range = np.linspace(0.1, 2, 10)
accuracy = 1e-6
nswp = 20

incidence = Incidence(angle=10)

smaller_scale = wavelength

with open('multiscale_thickness_sweep.log', 'a') as f:
    f.write('-' * 10)
    f.write('Starting new simulation\n')
    f.write(f'{datetime.datetime.now()}\n')
    f.write(f'accuracy={accuracy}, nswp={nswp}\n')
    f.write('Incidence:\n')
    f.write(f'\tangle\t{incidence.angle}\n')
    f.write('Grating:\n')
    f.write(f'\tn\t{1}\n')
    f.write(f'\teps\t{eps}\n')
    f.write('\n')
    f.write('D\tTHICKNESS\tAMP\n')

for thickness in thickness_range:
    for d in d_range:
        grating = Grating.multiscale_lamellar_random(
            period=n * smaller_scale,
            n=n,
            thickness=thickness,
            max_permittivity=eps,
            seed=1
        )

        x = solve_diffraction(
            do=d,
            dl=d,
            grating=grating,
            incidence=incidence,
            accuracy=accuracy,
            nswp=nswp
        )

        amp = abs(x.full(asvector=True)[2 ** (d - 1)])
        with open('multiscale_thickness_sweep.log', 'a') as f:
            f.write(f'{d}\t{thickness}\t{amp}')
