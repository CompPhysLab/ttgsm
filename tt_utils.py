import tt
import numpy as np


def map_on_modes(func, d, accuracy, verb=0):
    """
    Maps a function func on vector of mode indexes using TT-Cross.
    Returns func([-N, ..., N-1]) in TT-format, where N = 2 ** (d - 1)
    """
    x = tt.xfun(2, d)

    def func_shifted(arg):
        return func(arg - 2 ** (d - 1))
    y = tt.multifuncrs([x], func_shifted, eps=accuracy, y0=tt.ones(2, d), verb=verb)

    return y


# TODO: pull request into ttpy
def tt_tile(vector, n, d=None):
    # TODO: documentation
    if not isinstance(vector, tt.vector):
        raise TypeError(f'Expected tt.vector, not {type(vector).__name__}')
    if d is None:
        # TODO: raise error if n is not int
        n = np.array(n, dtype=int)
        d = n.size
    else:
        n = np.array([n] * d, dtype=int)
    cores = tt.vector.to_list(vector)
    for i in range(d):
        cores.append(np.ones((1, n[i], 1)))
    return tt.vector.from_list(cores)


# TODO: pull request to ttpy
def block_diagonal(x, d_block, n_block=None, m_block=None):
    # TODO: documentation
    dtype = x.core.dtype

    if n_block is not None:
        n_block = np.array(n_block, dtype=int)
    if m_block is not None:
        m_block = np.array(m_block, dtype=int)

    cores = tt.vector.to_list(x)
    new_cores = cores[:d_block]
    for i in range(d_block, x.d):
        s1, s2, s3 = cores[i].shape
        new_core = np.zeros((s1, s2 ** 2, s3), dtype=dtype)
        for j in range(s2):
            new_core[:, j * (s2 + 1), :] = cores[i][:, j, :]
        new_cores.append(new_core)
    result = tt.vector.from_list(new_cores)
    n_matrix, m_matrix = None, None
    if n_block is not None:
        n_matrix = np.concatenate((n_block, x.n[d_block:]))
    if m_block is not None:
        m_matrix = np.concatenate((m_block, x.n[d_block:]))
    result = tt.matrix(result, n=n_matrix, m=m_matrix)
    return result


def qtt_sin(d, alpha=1.0, phase=0.0):
    """ Create TT-vector for :math:`\\sin(\\alpha n + \\varphi)`."""
    dtype = type(alpha * phase * 1.0)

    cores = []
    first_core = np.zeros([1, 2, 2], dtype=dtype)
    first_core[0, 0, :] = [np.cos(phase), np.sin(phase)]
    first_core[0, 1, :] = [np.cos(alpha + phase), np.sin(alpha + phase)]
    cores.append(first_core)
    for i in range(1, d - 1):
        next_core = np.zeros([2, 2, 2], dtype=dtype)
        next_core[0, 0, :] = [1.0, 0.0]
        next_core[1, 0, :] = [0.0, 1.0]
        next_core[0, 1, :] = [np.cos(alpha * 2 ** i), np.sin(alpha * 2 ** i)]
        next_core[1, 1, :] = [-np.sin(alpha * 2 ** i), np.cos(alpha * 2 ** i)]
        cores.append(next_core)
    last_core = np.zeros([2, 2, 1], dtype=dtype)
    last_core[0, :, 0] = [0.0, np.sin(alpha * 2 ** (d - 1))]
    last_core[1, :, 0] = [1.0, np.cos(alpha * 2 ** (d - 1))]
    cores.append(last_core)
    return tt.vector.from_list(cores)


def qtt_cos(d, alpha=1.0, phase=0.0):
    """ Create TT-vector for :math:`\\cos(\\alpha n + \\varphi)`."""
    return qtt_sin(d, alpha, phase + np.pi * 0.5)


# TODO: pull request into ttpy
def qtt_exp(d, alpha=1.0, phase=0.0):
    """ Create TT-vector for :math:`\\exp(\\alpha n + \\varphi)`."""
    dtype = type(alpha * phase * 1.0)
    args = -1j * alpha, -1j * phase
    result = qtt_cos(d, *args) + 1j * qtt_sin(d, *args)
    if dtype is float:
        result = result.real()
    return result


def insert_zeros(x, d, non_zero_position=0):
    """
    Returns TT-vector containing len(x) blocks.
    Each block has a size of 2 ** d.
    Block k has only one non-zero element x_k
    positioned in non_zero_position within a block.
    """
    if non_zero_position >= 2 ** d:
        raise ValueError('non_zero_position must be less than 2 ** d')
    cores = []
    for i in range(d):
        if non_zero_position % 2 == 0:
            cores.append(np.array([[[1], [0]]]))
        else:
            cores.append(np.array([[[0], [1]]]))
        non_zero_position = non_zero_position // 2
    cores += tt.vector.to_list(x)
    return tt.vector.from_list(cores)
