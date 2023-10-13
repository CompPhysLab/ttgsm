import numpy as np
from time import time
from diffraction_solver import solve_diffraction
from incidence import Incidence
from grating import Grating


if __name__ == '__main__':
    do = 8
    dl = 8

    incidence = Incidence(angle=10)
    grating = Grating.lamellar(period=1.0, thickness=0.4, relative_filling=0.3, max_permittivity=2.1)

    start = time()

    x = solve_diffraction(do, dl, grating, incidence, accuracy=1e-6)
    print(x)

    print('Time:', time() - start, 's')
