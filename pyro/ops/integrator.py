from __future__ import absolute_import, division, print_function

import torch
from torch.autograd import grad


def velocity_verlet(z, r, potential_fn, inverse_mass_matrix, step_size, num_steps=1, z_grads=None):
    r"""
    Second order symplectic integrator that uses the velocity verlet algorithm.

    :param dict z: dictionary of sample site names and their current values
        (type :class:`~torch.Tensor`).
    :param dict r: dictionary of sample site names and corresponding momenta
        (type :class:`~torch.Tensor`).
    :param callable potential_fn: function that returns potential energy given z
        for each sample site. The negative gradient of the function with respect
        to ``z`` determines the rate of change of the corresponding sites'
        momenta ``r``.
    :param torch.Tensor inverse_mass_matrix: a tensor :math:`M^{-1}` which is used
        to calculate kinetic energy: :math:`E_{kinetic} = \frac{1}{2}z^T M^{-1} z`.
        Here :math:`M` can be a 1D tensor (diagonal matrix) or a 2D tensor (dense matrix).
    :param float step_size: step size for each time step iteration.
    :param int num_steps: number of discrete time steps over which to integrate.
    :param torch.Tensor z_grads: optional gradients of potential energy at current ``z``.
    :return tuple (z_next, r_next, z_grads, potential_energy): next position and momenta,
        together with the potential energy and its gradient w.r.t. ``z_next``.
    """
    z_next = z.copy()
    r_next = r.copy()
    for _ in range(num_steps):
        z_next, r_next, z_grads, potential_energy = _single_step_verlet(z_next,
                                                                        r_next,
                                                                        potential_fn,
                                                                        inverse_mass_matrix,
                                                                        step_size,
                                                                        z_grads)
    return z_next, r_next, z_grads, potential_energy


def _single_step_verlet(z, r, potential_fn, inverse_mass_matrix, step_size, z_grads=None):
    r"""
    Single step velocity verlet that modifies the `z`, `r` dicts in place.
    """

    z_grads = _potential_grad(potential_fn, z)[0] if z_grads is None else z_grads

    for site_name in r:
        r[site_name] = r[site_name] + 0.5 * step_size * (-z_grads[site_name])  # r(n+1/2)

    r_grads = _kinetic_grad(inverse_mass_matrix, r)
    for site_name in z:
        z[site_name] = z[site_name] + step_size * r_grads[site_name]  # z(n+1)

    z_grads, potential_energy = _potential_grad(potential_fn, z)
    for site_name in r:
        r[site_name] = r[site_name] + 0.5 * step_size * (-z_grads[site_name])  # r(n+1)

    return z, r, z_grads, potential_energy


def _potential_grad(potential_fn, z):
    z_keys, z_nodes = zip(*z.items())
    for node in z_nodes:
        node.requires_grad_(True)
    potential_energy = potential_fn(z)
    grads = grad(potential_energy, z_nodes)
    for node in z_nodes:
        node.requires_grad_(False)
    return dict(zip(z_keys, grads)), potential_energy.detach()


def _kinetic_grad(inverse_mass_matrix, r):
    # XXX consider using list/OrderDict to store z and r
    # so we don't have to sort the keys
    r_flat = torch.cat([r[site_name].reshape(-1) for site_name in sorted(r)])
    if inverse_mass_matrix.dim() == 1:
        grads_flat = inverse_mass_matrix * r_flat
    else:
        grads_flat = inverse_mass_matrix.matmul(r_flat)

    # unpacking
    grads = {}
    pos = 0
    for site_name in sorted(r):
        next_pos = pos + r[site_name].numel()
        grads[site_name] = grads_flat[pos:next_pos].reshape(r[site_name].shape)
        pos = next_pos
    assert pos == grads_flat.size(0)
    return grads
