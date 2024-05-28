import numpy as np

from diffraction_solver import solve_diffraction
from incidence import Incidence
from grating import Grating


if __name__ == '__main__':
    gsm_a0 = 0.844796923956883

    incidence = Incidence(angle=30)
    grating = Grating.lamellar(period=6, thickness=6, relative_filling=0.75, max_permittivity=2.1)

    for d in range(2, 17):
        for n_digits in range(3, 11):
            x = solve_diffraction(d, d, grating, incidence, accuracy=10 ** -n_digits)
            error = abs(max(abs(x.full(asvector=True))) - gsm_a0)
            print(d, n_digits, error)
