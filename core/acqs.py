from botorch.models import FixedNoiseGP, ModelListGP, SingleTaskGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
import os, sys
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound, qExpectedImprovement
from  botorch.acquisition.analytic import LogConstrainedExpectedImprovement, _scaled_improvement, _ei_helper
from botorch.optim import optimize_acqf
from gpytorch.kernels import ScaleKernel, RBFKernel
import numpy as np
from torch import Tensor

from utils.parser import nparray_to_params_dict


class EI(ExpectedImprovement):
    def forward(self, X: Tensor) -> Tensor:
        mean, sigma = self._mean_and_sigma(X)
        u = _scaled_improvement(mean, sigma, self.best_f, self.maximize)
        return sigma * _ei_helper(u)
