#
# CEBRA: Consistent EmBeddings of high-dimensional Recordings using Auxiliary variables
# © Mackenzie W. Mathis & Steffen Schneider (v0.4.0+)
# Source code:
# https://github.com/AdaptiveMotorControlLab/CEBRA
#
# Please see LICENSE.md for the full license document:
# https://github.com/AdaptiveMotorControlLab/CEBRA/blob/main/LICENSE.md
#
# This file contains the PyTorch implementation of Jacobian regularization described in [1].
#   Judy Hoffman, Daniel A. Roberts, and Sho Yaida,
#   "Robust Learning with Jacobian Regularization," 2019.
#   [arxiv:1908.02729](https://arxiv.org/abs/1908.02729)
#
# Adapted from https://github.com/facebookresearch/jacobian_regularizer/blob/main/jacobian/jacobian.py
# licensed under the following MIT License:
#
#   MIT License
#
#   Copyright (c) Facebook, Inc. and its affiliates.
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all
#   copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#   SOFTWARE.
#

from __future__ import division

import torch
import torch.nn as nn


class JacobianReg(nn.Module):
    '''
    Loss criterion that computes the trace of the square of the Jacobian.

    Arguments:
        n (int, optional): determines the number of random projections.
            If n=-1, then it is set to the dimension of the output
            space and projection is non-random and orthonormal, yielding
            the exact result.  For any reasonable batch size, the default
            (n=1) should be sufficient.
    '''

    def __init__(self, n=1):
        assert n == -1 or n > 0
        self.n = n
        super(JacobianReg, self).__init__()

    def forward(self, x, y):
        '''
        computes (1/2) tr |dy/dx|^2
        '''
        B, C = y.shape
        if self.n == -1:
            num_proj = C
        else:
            num_proj = self.n
        J2 = 0
        for ii in range(num_proj):
            if self.n == -1:
                # orthonormal vector, sequentially spanned
                v = torch.zeros(B, C)
                v[:, ii] = 1
            else:
                # random properly-normalized vector for each sample
                v = self._random_vector(C=C, B=B)
            if x.is_cuda:
                v = v.cuda()
            Jv = self._jacobian_vector_product(y, x, v, create_graph=True)
            J2 += C * torch.norm(Jv)**2 / (num_proj * B)
        R = (1 / 2) * J2
        return R

    def _random_vector(self, C, B):
        '''
        creates a random vector of dimension C with a norm of C^(1/2)
        (as needed for the projection formula to work)
        '''
        if C == 1:
            return torch.ones(B)
        v = torch.randn(B, C)
        arxilirary_zero = torch.zeros(B, C)
        vnorm = torch.norm(v, 2, 1, True)
        v = torch.addcdiv(arxilirary_zero, 1.0, v, vnorm)
        return v

    def _jacobian_vector_product(self, y, x, v, create_graph=False):
        '''
        Produce jacobian-vector product dy/dx dot v.

        Note that if you want to differentiate it,
        you need to make create_graph=True
        '''
        flat_y = y.reshape(-1)
        flat_v = v.reshape(-1)
        grad_x, = torch.autograd.grad(flat_y,
                                      x,
                                      flat_v,
                                      retain_graph=True,
                                      create_graph=create_graph)
        return grad_x
