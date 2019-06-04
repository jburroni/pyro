from __future__ import absolute_import, division, print_function

import torch

import pyro
import pyro.optim as optim

from pyro.distributions import Normal
from pyro.infer import SVI, TraceMeanField_ELBO

torch.set_default_tensor_type('torch.FloatTensor')
pyro.enable_validation(True)
pyro.util.set_rng_seed(0)

N_data = 50
layer1_width = 3
layer2_width = 4
N_particles = 7


def modelguide(data, mode="no_particles", include_obs=True, prior_mean=0.0):
    layer1 = pyro.plate("layer1_outputs", layer1_width)
    layer2 = pyro.plate("layer2_outputs", layer2_width)

    with pyro.plate("data", N_data):
        with layer1:
            w1 = pyro.sample("w1", Normal(prior_mean, 1.0))
            if mode == "particles":
                print("w1", w1.shape, "mode", mode)
                assert w1.shape == (N_particles, layer1_width, N_data)
            else:
                print("w1", w1.shape, "mode", mode)
                assert w1.shape == (layer1_width, N_data)

        with layer2:
            # note w1.mean has a N_particles dimension in particles mode
            # so no expansion will happen here (the w1 samples do the expanding)
            w2 = pyro.sample("w2", Normal(w1.mean(-2, keepdim=True), 1.0))
            if mode == "particles":
                print("w2", w2.shape, "mode", mode)
                assert w2.shape == (N_particles, layer2_width, N_data)
            else:
                assert w2.shape == (layer2_width, N_data)
                print("w2", w2.shape, "mode", mode)

        if include_obs:
              pyro.sample("obs", Normal(w2.mean(-2, keepdim=True), 1.0), obs=data)


def model(data, mode="no_particles"):
    return modelguide(data, mode, include_obs=True, prior_mean=0.0)

def guide(data, mode="no_particles"):
    return modelguide(data, mode, include_obs=False, prior_mean=0.123)


def main():
    data = torch.randn(N_data)
    opt = optim.Adam({"lr": 0.001})

    svi = SVI(model, guide, opt, loss=TraceMeanField_ELBO())
    svi_particles = SVI(model, guide, opt,
                        loss=TraceMeanField_ELBO(num_particles=N_particles, vectorize_particles=True,
                                                 max_plate_nesting=2))

    for k in range(2):
        loss = svi.step(data, mode="no_particles")
        print("loss", loss)
        loss_particles = svi_particles.step(data, mode="particles")
        print("loss_particles", loss_particles)


if __name__ == '__main__':
    main()
