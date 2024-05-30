import numpy as np
import datetime

from diffraction_solver import solve_diffraction
from incidence import Incidence
from grating import Grating


gsm_a0 = 0.844796923956883

incidence = Incidence(angle=30)
grating = Grating.lamellar(period=6, thickness=6, relative_filling=0.75, max_permittivity=2.1)

with open('accuracy_control.log', 'a') as f:
    f.write('-' * 10)
    f.write('Starting new simulation\n')
    f.write(f'{datetime.datetime.now()}\n')
    f.write('Incidence:\n')
    f.write(f'\tangle\t{incidence.angle}\n')
    f.write('Grating:\n')
    f.write(f'\tperiod\t{grating.period}\n')
    f.write(f'\tthickness\t{grating.thickness}\n')
    f.write(f'\tepsilon\t2.1\n')
    f.write(f'GSM reference amplitude: {gsm_a0}\n')
    f.write('\n')
    f.write('D\tEPS\tERROR\n')

for d in range(2, 17):
    for n_digits in range(3, 11):
        x = solve_diffraction(d, d, grating, incidence, accuracy=10 ** -n_digits)
        error = abs(max(abs(x.full(asvector=True))) - gsm_a0)
        with open('accuracy_control.log', 'a') as f:
            f.write(f'{d}\t{n_digits}\t{error}')
