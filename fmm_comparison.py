import numpy as np

from diffraction_solver import solve_diffraction
from incidence import Incidence
from grating import Grating

from fmm_reference import fmm_data


if __name__ == '__main__':
    for fmm_test in fmm_data[:1]:
        incidence = Incidence(fmm_test['angle'])
        grating = Grating.lamellar(period=fmm_test['period'],
                                   thickness=fmm_test['thickness'],
                                   relative_filling=fmm_test['relative_filling'],
                                   max_permittivity=fmm_test['permittivity'],
                                   relative_shift=0.)

        x = solve_diffraction(9, 9, grating, incidence, verb=0)
        x = abs(x.full(asvector=True)[:2 ** 9])

        print(np.linalg.norm(x - fmm_test['amps']))
