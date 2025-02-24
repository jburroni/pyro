from __future__ import absolute_import, division, print_function

import math

import torch
import torch.nn as nn
from torch.distributions import constraints

from pyro.distributions.torch_transform import TransformModule
from pyro.distributions.util import copy_docs_from


@copy_docs_from(TransformModule)
class PolynomialFlow(TransformModule):
    """
    An autoregressive normalizing flow as described in Jaini et al. (2019) using the element-wise transformation

        :math:`y_n = c_n + \\int^{x_n}_0\\sum^K_{k=1}\\left(\\sum^R_{r=0}a^{(n)}_{r,k}u^r\\right)du`

    where :math:`x_n` is the :math:`n`th input, :math:`y_n` is the :math:`n`th output, and :math:`c_n\\in\\mathbb{R}`,
    :math:`\\left{a^{(n)}_{r,k}\\in\\mathbb{R}\\right}` are learnable parameters that are the output of an
    autoregressive NN inputting :math:`x_{\\prec n}={x_1,x_2,\\ldots,x_{n-1}}`.

    Together with `TransformedDistribution` this provides a way to create richer variational approximations.

    Example usage:

    >>> from pyro.nn import AutoRegressiveNN
    >>> input_dim = 10
    >>> count_degree = 4
    >>> count_sum = 3
    >>> base_dist = dist.Normal(torch.zeros(input_dim), torch.ones(input_dim))
    >>> arn = AutoRegressiveNN(input_dim, [input_dim*10], param_dims=[(count_degree + 1)*count_sum])
    >>> flow = PolynomialFlow(arn, input_dim=input_dim, count_degree=count_degree, count_sum=count_sum)
    >>> pyro.module("my_flow", flow)  # doctest: +SKIP
    >>> flow_dist = dist.TransformedDistribution(base_dist, [flow])
    >>> flow_dist.sample()  # doctest: +SKIP
        tensor([-0.4071, -0.5030,  0.7924, -0.2366, -0.2387, -0.1417,  0.0868,
                0.1389, -0.4629,  0.0986])

    The inverse of this transform does not possess an analytical solution and is left unimplemented. However,
    the inverse is cached when the forward operation is called during sampling, and so samples drawn using
    polynomial flow can be scored.

    :param autoregressive_nn: an autoregressive neural network whose forward call returns a tensor of real-valued
        numbers of size (batch_size, (count_degree+1)*count_sum, input_dim)
    :type autoregressive_nn: nn.Module
    :param count_degree: The degree of the polynomial to use for each element-wise transformation.
    :type count_degree: int
    :param count_sum: The number of polynomials to sum in each element-wise transformation.
    :type count_sum: int

    References:

    Sum-of-squares polynomial flow. [arXiv:1905.02325]
    Priyank Jaini, Kira A. Shelby, Yaoliang Yu
    """

    domain = constraints.real
    codomain = constraints.real
    bijective = True
    event_dim = 1
    autoregressive = True

    def __init__(self, autoregressive_nn, input_dim, count_degree, count_sum):
        super(PolynomialFlow, self).__init__(cache_size=1)

        self.arn = autoregressive_nn
        self.input_dim = input_dim
        self.count_degree = count_degree
        self.count_sum = count_sum
        self._cached_logDetJ = None

        self.c = nn.Parameter(torch.Tensor(input_dim))
        self.reset_parameters()

        # Vector of powers of input dimension
        powers = torch.arange(1, count_degree + 2, dtype=torch.get_default_dtype())
        self.register_buffer('powers', powers)

        # Build mask of constants
        mask = self.powers + torch.arange(count_degree+1).unsqueeze(-1).type_as(powers)
        power_mask = mask
        mask = mask.reciprocal()

        self.register_buffer('power_mask', power_mask)
        self.register_buffer('mask', mask)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.c.size(0))
        self.c.data.uniform_(-stdv, stdv)

    def _call(self, x):
        """
        :param x: the input into the bijection
        :type x: torch.Tensor

        Invokes the bijection x=>y; in the prototypical context of a TransformedDistribution `x` is a
        sample from the base distribution (or the output of a previous flow)
        """
        # Calculate the polynomial coefficients
        # ~ (batch_size, count_sum, count_degree+1, input_dim)
        A = self.arn(x).view(-1, self.count_sum, self.count_degree + 1, self.input_dim)

        # Take cross product of coefficients across degree dim
        # ~ (batch_size, count_sum, count_degree+1, count_degree+1, input_dim)
        coefs = A.unsqueeze(-2) * A.unsqueeze(-3)

        # Calculate output as sum-of-squares polynomial
        x_view = x.view(-1, 1, 1, self.input_dim)
        x_pow_matrix = x_view.pow(self.power_mask.unsqueeze(-1)).unsqueeze(-4)

        # Eq (8) from the paper, expanding the squared term and integrating
        # NOTE: The view_as is necessary because the batch dimensions were collapsed previously
        y = self.c + (coefs * x_pow_matrix * self.mask.unsqueeze(-1)).sum((1, 2, 3)).view_as(x)

        # log(|det(J)|) is calculated by the fundamental theorem of calculus, i.e. remove the constant
        # term and the integral from eq (8) (the equation for this isn't given in the paper)
        x_pow_matrix = x_view.pow(self.power_mask.unsqueeze(-1) - 1).unsqueeze(-4)
        self._cached_logDetJ = torch.log((coefs * x_pow_matrix).sum((1, 2, 3)).view_as(x) + 1e-8).sum(-1)

        return y

    def _inverse(self, y):
        """
        :param y: the output of the bijection
        :type y: torch.Tensor

        Inverts y => x. As noted above, this implementation is incapable of inverting arbitrary values
        `y`; rather it assumes `y` is the result of a previously computed application of the bijector
        to some `x` (which was cached on the forward call)
        """

        raise KeyError("PolynomialFlow expected to find key in intermediates cache but didn't")

    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log Jacobian
        """
        return self._cached_logDetJ
