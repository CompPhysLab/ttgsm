import numpy as np
import tt
from tt.amen import amen_solve, amen_mv

from tt_utils import tt_tile, map_on_modes, qtt_exp, block_diagonal, insert_zeros


class SimulationVariables:
    def __init__(self, do, dl, grating, incidence, accuracy=1e-6, verb=0):
        self.do = do
        self.dl = dl
        self.accuracy = accuracy
        self.grating = grating
        self.incidence = incidence

        self.no = 2 ** do
        self.nl = 2 ** dl
        self.dh = grating.thickness / self.nl

        self.verb = verb

        self.kz_vector = map_on_modes(lambda x: k_zn(grating, incidence, x), do, accuracy, verb=verb)


def k_xn(grating, incidence, n):
    return incidence.kx0 + n * 2 * np.pi / grating.period


def k_zn(grating, incidence, n):
    return np.sqrt(grating.background_permittivity - k_xn(grating, incidence, n) ** 2, dtype=complex)


def calculate_r(simulation_variables):
    do, dl = simulation_variables.do, simulation_variables.dl
    dh = simulation_variables.dh
    kz = simulation_variables.kz_vector
    accuracy = simulation_variables.accuracy
    verb = simulation_variables.verb

    # exponents with layer indexes
    powers = tt.xfun(2, dl)
    powers = tt.Toeplitz(powers, kind='L').T
    powers *= 1j * dh

    # Kronecker product resulting in tensor of diagonal blocks
    diag_blocks = tt.kron(kz, powers.tt)

    # exp(x)
    diag_blocks = tt.multifuncrs([diag_blocks], np.exp, eps=accuracy, verb=verb)

    # in each block: subtract one from upper triangle and 1/2 from diagonal elements
    subtrahend = tt.Toeplitz(tt.ones(2, dl), kind='L') - 0.5 * tt.eye(2, dl)
    subtrahend = tt.kron(tt.ones(2, do), subtrahend.tt)
    diag_blocks -= subtrahend

    # block-diagonal matrix
    # the following code does the same as block_diagonal(),
    # but adds layers into first cores instead of last ones
    cores = tt.vector.to_list(diag_blocks)
    new_cores = []
    for i in range(do):
        a, b, c = cores[i].shape
        new_core = np.zeros((a, 4, c), dtype=np.complex128)
        new_core[:, 0, :] = cores[i][:, 0, :]
        new_core[:, 3, :] = cores[i][:, 1, :]
        new_cores.append(new_core)
    new_cores += cores[do:]
    r = tt.vector.from_list(new_cores)
    r = tt.matrix(r)

    # concatenate R+ and R- using Kronecker product
    upper_mask_cores = [np.zeros((1, 4, 1))]
    lower_mask_cores = [np.zeros((1, 4, 1))]
    upper_mask_cores[0][:, 0, :] = 1
    lower_mask_cores[0][:, 3, :] = 1
    upper_mask = tt.matrix(tt.vector.from_list(upper_mask_cores))
    lower_mask = tt.matrix(tt.vector.from_list(lower_mask_cores))
    r = tt.kron(r, upper_mask) + tt.kron(r.T, lower_mask)
    r = r.round(eps=accuracy)
    return r


def calculate_t(simulation_variables):
    do, dl = simulation_variables.do, simulation_variables.dl
    dh = simulation_variables.dh
    kz = simulation_variables.kz_vector
    accuracy = simulation_variables.accuracy
    verb = simulation_variables.verb

    # exponents i * \Delta z_p
    powers_positive = tt.ones(2, dl) * (2 ** dl - 0.5) - tt.xfun(2, dl)
    powers_positive *= 1j * dh

    powers_negative = tt.ones(2, dl) * 0.5 + tt.xfun(2, dl)
    powers_negative *= 1j * dh

    # Kronecker product resulting in tensor of diagonal blocks
    diag_blocks_positive = tt.kron(kz, powers_positive)
    diag_blocks_negative = tt.kron(kz, powers_negative)

    # exp(x)
    diag_blocks_positive = tt.multifuncrs([diag_blocks_positive], np.exp, eps=accuracy, verb=verb)
    diag_blocks_negative = tt.multifuncrs([diag_blocks_negative], np.exp, eps=accuracy, verb=verb)

    # block-diagonal matrix
    # the following code does the same as block_diagonal(),
    # but adds layers into first cores instead of last ones
    cores = tt.vector.to_list(diag_blocks_positive)
    new_cores = []
    for i in range(do):
        a, b, c = cores[i].shape
        new_core = np.zeros((a, 4, c), dtype=np.complex128)
        new_core[:, 0, :] = cores[i][:, 0, :]
        new_core[:, 3, :] = cores[i][:, 1, :]
        new_cores.append(new_core)
    new_cores += cores[do:]
    t_positive = tt.vector.from_list(new_cores)
    t_positive = tt.matrix(t_positive, n=[2] * do + [1] * dl, m=[2] * (do + dl))

    cores = tt.vector.to_list(diag_blocks_negative)
    new_cores = []
    for i in range(do):
        a, b, c = cores[i].shape
        new_core = np.zeros((a, 4, c), dtype=np.complex128)
        new_core[:, 0, :] = cores[i][:, 0, :]
        new_core[:, 3, :] = cores[i][:, 1, :]
        new_cores.append(new_core)
    new_cores += cores[do:]
    t_negative = tt.vector.from_list(new_cores)
    t_negative = tt.matrix(t_negative, n=[2] * do + [1] * dl, m=[2] * (do + dl))

    upper_mask_cores = [np.zeros((1, 4, 1))]
    lower_mask_cores = [np.zeros((1, 4, 1))]
    upper_mask_cores[0][:, 1, :] = 1
    lower_mask_cores[0][:, 2, :] = 1
    upper_mask = tt.matrix(tt.vector.from_list(upper_mask_cores))
    lower_mask = tt.matrix(tt.vector.from_list(lower_mask_cores))
    t = tt.kron(t_positive, upper_mask) + tt.kron(t_negative, lower_mask)
    t = t.round(eps=accuracy)
    return t


def calculate_p(simulation_variables):
    do, dl = simulation_variables.do, simulation_variables.dl
    accuracy = simulation_variables.accuracy
    verb = simulation_variables.verb

    def p_func(n):
        return 1 / k_zn(simulation_variables.grating, simulation_variables.incidence, n)

    p = map_on_modes(p_func, do, accuracy, verb)
    p = tt.diag(tt_tile(p, 2, dl + 1))
    return p


def calculate_q(simulation_variables):
    do, dl = simulation_variables.do, simulation_variables.dl
    d = do + dl

    identity = tt.eye(2, d)
    cores = tt.vector.to_list(identity.tt)
    cores.append(np.ones((1, 4, 1)))
    q = tt.matrix(tt.vector.from_list(cores), n=[2] * (d + 1), m=[2] * (d + 1))
    return q


def calculate_v(simulation_variables):
    do, dl = simulation_variables.do, simulation_variables.dl
    permittivity_fourier = simulation_variables.grating.permittivity_fourier
    dh = simulation_variables.dh
    background_permittivity = simulation_variables.grating.background_permittivity
    accuracy = simulation_variables.accuracy
    verb = simulation_variables.verb

    modes = (2 ** do) * tt.ones(2, do) - tt.xfun(2, do)
    fourier_nonzero_vector_positive = tt.multifuncrs([modes], permittivity_fourier, eps=accuracy, verb=verb)
    fourier_nonzero_vector_negative = tt.multifuncrs([-modes], permittivity_fourier, eps=accuracy, verb=verb)
    fourier_zero_mode = permittivity_fourier(0)

    v_block = tt.Toeplitz(fourier_nonzero_vector_negative, kind='U')
    v_block += tt.Toeplitz(fourier_nonzero_vector_positive, kind='U').T
    v_block += fourier_zero_mode * tt.eye(2, do)
    v_block -= tt.eye(2, do)
    v_block *= 1j * 0.5 * background_permittivity * dh
    v = block_diagonal(tt_tile(v_block.tt, 2, dl + 1), do)
    v = v.round(eps=accuracy)
    return v


def plane_wave_in_layers(simulation_variables):
    do, dl = simulation_variables.do, simulation_variables.dl
    kz0 = k_zn(simulation_variables.grating, simulation_variables.incidence, 0)
    dh = simulation_variables.dh

    phase_multiplier = 1j * kz0 * dh
    amps = qtt_exp(dl, alpha=-phase_multiplier, phase=(2 ** dl - 0.5) * phase_multiplier)

    # add zeros in all non-zero diffraction orders
    amps = insert_zeros(amps, do, non_zero_position=2 ** (do - 1))

    # double vector size: add zeros for all waves propagating in negative direction
    cores = tt.vector.to_list(amps)
    cores.append(np.array([[[0], [1]]]))
    amps = tt.vector.from_list(cores)

    return amps


def solve_diffraction(do, dl, grating, incidence, accuracy=1e-6, verb=0, nswp=20):
    # TODO: warnings are suppressed because of ttpy TT Cross implementation, make pull request to ttpy in the future
    # TODO: needs to be DISABLED while testing new features!!!
    np.seterr(divide='ignore', invalid='ignore')

    simulation_variables = SimulationVariables(do, dl, grating, incidence, accuracy=accuracy, verb=verb)

    r = calculate_r(simulation_variables)
    p = calculate_p(simulation_variables)
    v = calculate_v(simulation_variables)
    q = calculate_q(simulation_variables)
    t = calculate_t(simulation_variables)

    diffraction_matrix = p
    for matrix in [v, q]:
        diffraction_matrix = diffraction_matrix * matrix
        diffraction_matrix = diffraction_matrix.round(accuracy)

    a = r * diffraction_matrix
    a = a.round(accuracy)
    identity = tt.eye(2, do + dl + 1)
    a = identity - a
    # memory_to_print = len(a.tt.core)

    external = plane_wave_in_layers(simulation_variables)
    modes_in_layers = amen_solve(a, external, tt.ones(2, do + dl + 1), accuracy, verb=verb)
    modes_in_layers = modes_in_layers.round(accuracy)

    # propagate to substrate and superstrate
    a = t * diffraction_matrix
    a = a.round(accuracy)
    # TODO: amen_mv + initial guess (try y = Ax = [0 ... 0 1 0 ... 0 1 0 ... 0])
    # np.set_printoptions(threshold=np.inf)
    # np.set_printoptions(linewidth=np.inf)
    # print(modes_in_layers.full(asvector=True))
    # print(a.full())

    modes = amen_mv(a.real(), modes_in_layers.real(), accuracy, verb=verb, nswp=nswp)[0]

    # print(modes.full(asvector=True))
    # print(modes)
    # print(np.linalg.norm(modes.real().full(asvector=True) - np.dot(a.real().full(), modes_in_layers.real().full(asvector=True))))

    # initial_guess_modes = tt.ones(a.n)
    # modes = amen_mv(a, modes_in_layers, accuracy, y=initial_guess_modes, verb=1)[0]

    # modes = tt.matvec(a, modes_in_layers)
    # modes = modes.round(accuracy)

    # as matrix T was rectangular, modes have (do + dl) dimensions
    # but n(do+1, ..., do+dl) = 1
    # multiply cores with size 1 to make d = do+1
    modes_cores = tt.vector.to_list(modes)
    modes_cores_shortened = modes_cores[:do]
    shortened_core = 1
    for i in range(do, do + dl):
        shortened_core = np.dot(shortened_core, modes_cores[i][:, 0, :])
    last_core = np.zeros((shortened_core.shape[0], modes.n[-1], 1), dtype=complex)
    for i in range(modes.n[-1]):
        last_core_layer = np.dot(shortened_core, modes_cores[-1][:, i, :])
        last_core[:, i, :] = last_core_layer
    modes_cores_shortened.append(last_core)
    modes = tt.vector.from_list(modes_cores_shortened)

    external_one_layer = insert_zeros(
        tt.vector.from_list([np.array([[[1], [0]]], dtype=complex)]),
        do, 2 ** (do - 1))
    modes += external_one_layer * np.exp(1j * k_zn(grating, incidence, 0) * grating.thickness)

    return modes
