from __future__ import absolute_import, division, print_function

import torch

import pyro
import pyro.optim as optim

from pyro.distributions import Normal
from pyro.infer import SVI, TraceMeanField_ELBO

torch.set_default_tensor_type('torch.FloatTensor')
pyro.enable_validation(True)
pyro.util.set_rng_seed(0)


def model(data):
    return pyro.sample("w", Normal(0.0, 1.0))

def guide(data):
    return pyro.sample("w", Normal(0.0, 1.0))


def main():

    data = torch.randn(5,5)

    opt = optim.Adam({"lr": 0.001})

    svi = SVI(model, guide, opt, loss=TraceMeanField_ELBO())

    #svi_eval = SVI(sparse_gamma_def.model, guide, opt,
    #               loss=TraceMeanField_ELBO(num_particles=args.eval_particles, vectorize_particles=True))

    for k in range(3):
        loss = svi.step(data)
        print("loss", loss)


if __name__ == '__main__':
    main()
