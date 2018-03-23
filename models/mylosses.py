# -*- coding: utf-8 -*-
import numpy as np
from torch.nn.modules import loss
from torch.nn import functional as F
import torch
from torch.autograd import Variable

class RelMAELoss(loss._Loss):
    r"""Creates a criterion that measures the mean squared error between
    `n` elements in the input `x` and target `y`.

    The loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \left( x_n - y_n \right)^2,

    where :math:`N` is the batch size. If reduce is ``True``, then:

    .. math::
        \ell(x, y) = \begin{cases}
            \operatorname{mean}(L), & \text{if}\; \text{size_average} = \text{True},\\
            \operatorname{sum}(L),  & \text{if}\; \text{size_average} = \text{False}.
        \end{cases}

    `x` and `y` arbitrary shapes with a total of `n` elements each.

    The sum operation still operates over all the elements, and divides by `n`.

    The division by `n` can be avoided if one sets the internal variable
    `size_average` to ``False``.

    To get a batch of losses, a loss per batch element, set `reduce` to
    ``False``. These losses are not averaged and are not affected by
    `size_average`.

    Args:
        size_average (bool, optional): By default, the losses are averaged
           over observations for each minibatch. However, if the field
           size_average is set to ``False``, the losses are instead summed for
           each minibatch. Only applies when reduce is ``True``. Default: ``True``
        reduce (bool, optional): By default, the losses are averaged
           over observations for each minibatch, or summed, depending on
           size_average. When reduce is ``False``, returns a loss per batch
           element instead and ignores size_average. Default: ``True``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Target: :math:`(N, *)`, same shape as the input

    Examples::

        >>> loss = nn.MSELoss()
        >>> input = autograd.Variable(torch.randn(3, 5), requires_grad=True)
        >>> target = autograd.Variable(torch.randn(3, 5))
        >>> output = loss(input, target)
        >>> output.backward()
    """
    def __init__(self, size_average=True, reduce=True):
        super(RelMAELoss, self).__init__(size_average)
        self.reduce = reduce

    def forward(self, input, target):
        input = (input + 1) / 2.0 * 4095.0
        target = (target + 1) / 2.0 * 4095.0
        loss._assert_no_grad(target)
        abs_diff = torch.abs(target - input)
        relative_abs_diff = abs_diff / (target + np.finfo(float).eps)
        rel_mae = torch.mean(relative_abs_diff)

        #from eval:
        # compute MRAE
        # diff = gt - rc
        # abs_diff = np.abs(diff)
        # relative_abs_diff = np.divide(abs_diff, gt + np.finfo(float).eps)  # added epsilon to avoid division by zero.
        # MRAEs[f] = np.mean(relative_abs_diff)
        return rel_mae


class ZeroGanLoss(loss._Loss):
    r"""Creates a criterion that measures the mean squared error between
    `n` elements in the input `x` and target `y`.

    The loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \left( x_n - y_n \right)^2,

    where :math:`N` is the batch size. If reduce is ``True``, then:

    .. math::
        \ell(x, y) = \begin{cases}
            \operatorname{mean}(L), & \text{if}\; \text{size_average} = \text{True},\\
            \operatorname{sum}(L),  & \text{if}\; \text{size_average} = \text{False}.
        \end{cases}

    `x` and `y` arbitrary shapes with a total of `n` elements each.

    The sum operation still operates over all the elements, and divides by `n`.

    The division by `n` can be avoided if one sets the internal variable
    `size_average` to ``False``.

    To get a batch of losses, a loss per batch element, set `reduce` to
    ``False``. These losses are not averaged and are not affected by
    `size_average`.

    Args:
        size_average (bool, optional): By default, the losses are averaged
           over observations for each minibatch. However, if the field
           size_average is set to ``False``, the losses are instead summed for
           each minibatch. Only applies when reduce is ``True``. Default: ``True``
        reduce (bool, optional): By default, the losses are averaged
           over observations for each minibatch, or summed, depending on
           size_average. When reduce is ``False``, returns a loss per batch
           element instead and ignores size_average. Default: ``True``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Target: :math:`(N, *)`, same shape as the input

    Examples::

        >>> loss = nn.MSELoss()
        >>> input = autograd.Variable(torch.randn(3, 5), requires_grad=True)
        >>> target = autograd.Variable(torch.randn(3, 5))
        >>> output = loss(input, target)
        >>> output.backward()
    """

    def __init__(self, size_average=True, reduce=True):
        super(ZeroGanLoss, self).__init__(size_average)
        self.reduce = reduce

    def forward(self, input, target):
        # zero = Variable(torch.Tensor([0]).double())
        zeros = input * 0.
        return torch.sum(zeros)
