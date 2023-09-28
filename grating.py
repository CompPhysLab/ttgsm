import numpy as np
import random
import logging


class Grating:
    def __init__(self,
                 period, thickness, permittivity_fourier,
                 background_permittivity=1):
        self.period = period
        self.thickness = thickness
        self.permittivity_fourier = permittivity_fourier
        self.background_permittivity = background_permittivity

    @staticmethod
    def lamellar_permittivity_fourier(relative_filling,
                                      max_permittivity, min_permittivity=1.,
                                      background_permittivity=1,
                                      relative_shift=0):
        def func(mode_index):
            factor = (max_permittivity - min_permittivity) / background_permittivity
            with np.errstate(divide='ignore', invalid='ignore'):
                result = factor * np.sin(np.pi * mode_index * relative_filling) / (np.pi * mode_index)
            result = np.nan_to_num(result, nan=factor * relative_filling + min_permittivity / background_permittivity)
            result = result.astype(complex)
            result *= np.exp(-1j * mode_index * 2 * np.pi * relative_shift)
            return result

        return func

    @staticmethod
    def lamellar(period, thickness,
                 relative_filling,
                 max_permittivity, min_permittivity=1.,
                 background_permittivity=1,
                 relative_shift=0):
        return Grating(period,
                       thickness,
                       Grating.lamellar_permittivity_fourier(
                           relative_filling, max_permittivity, min_permittivity,
                           background_permittivity, relative_shift),
                       background_permittivity)

    @staticmethod
    def multiscale_lamellar_random(period, n, thickness,
                                   max_permittivity, min_permittivity=1.,
                                   max_width=0.7, min_width=0.2,
                                   background_permittivity=1,
                                   seed=None):

        logging.basicConfig(filename='multiscale_structure.log', format='%(message)s')

        if seed is not None:
            random.seed(seed)

        width_per_slab = 1 / n
        slab_functions = []
        for i in range(n):
            center_point = width_per_slab * (i - n / 2 + 0.5)
            width = width_per_slab * random.uniform(min_width, max_width)
            logging.warning(f'{center_point}, {width}')
            slab_functions.append(
                Grating.lamellar_permittivity_fourier(
                    relative_filling=width,
                    max_permittivity=max_permittivity - min_permittivity, min_permittivity=0,
                    background_permittivity=background_permittivity,
                    relative_shift=center_point
                )
            )
        slab_functions.append(
            Grating.lamellar_permittivity_fourier(
                relative_filling=1,
                max_permittivity=min_permittivity, min_permittivity=0,
                background_permittivity=background_permittivity
            )
        )

        def multiscale_permittivity_fourier(x):
            return sum([func(x) for func in slab_functions])

        return Grating(period=period, thickness=thickness,
                       permittivity_fourier=multiscale_permittivity_fourier,
                       background_permittivity=background_permittivity)
