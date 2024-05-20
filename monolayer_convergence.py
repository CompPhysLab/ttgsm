import numpy as np
import matplotlib.pyplot as plt

from diffraction_solver import solve_diffraction
from incidence import Incidence
from grating import Grating


if __name__ == '__main__':
    n = 5
    d_range = range(2, 17)
    # d_range = [15] * 1
    thickness = 0.3
    accuracy = 1e-6
    n_sweeps = 20
    verb = 0

    smaller_scale = 2 * np.pi

    amps = []

    for d in d_range:
        incidence = Incidence(angle=10)
        grating = Grating.multiscale_lamellar_random(period=n * smaller_scale, n=n,
                                                     thickness=thickness, max_permittivity=2.1, seed=1)
        x = solve_diffraction(
            do=d,
            dl=2,
            grating=grating,
            incidence=incidence,
            accuracy=accuracy,
            nswp=n_sweeps,
            verb=verb
        )

        amp = abs(x.full(asvector=True)[2 ** (d - 1)])
        amp = 1 - amp
        print(d, amp)
        amps.append(amp)

    amps = np.array(amps)

    plt.plot(np.array(d_range)[1:], np.abs(amps[:-1] - amps[1:]), marker='o')
    # plt.plot(2 ** np.array(d_range)[1:], np.abs(amps[:-1] - amps[1:]), marker='o')
    plt.yscale('log')
    # plt.xscale('log')
    plt.grid()
    # plt.xlabel('Number of diffraction orders')
    plt.xlabel('d_O')
    plt.ylabel('Difference')
    plt.show()
