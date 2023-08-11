import numpy as np


class Incidence:
    def __init__(self, angle):
        self.angle = angle
        self.kx0 = np.sin(angle * np.pi / 180)
