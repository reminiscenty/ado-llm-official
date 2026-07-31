import warnings
from typing import Any

import numpy as np
import torch
from botorch.acquisition import (
    ExpectedImprovement,
    LogExpectedImprovement,
    UpperConfidenceBound,
    qExpectedImprovement,
    qLogExpectedImprovement,
    qUpperConfidenceBound,
)
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

from utils.parser import get_param_bounds, nparray_to_params_dict
from utils.utils import get_xy

warnings.filterwarnings("ignore")

ACQF_1 = {
    "EI": ExpectedImprovement,
    "UCB": UpperConfidenceBound,
    "LogEI": LogExpectedImprovement,
}
ACQF_q = {
    "EI": qExpectedImprovement,
    "UCB": qUpperConfidenceBound,
    "LogEI": qLogExpectedImprovement,
}


class GPBO:
    """Gaussian-process Bayesian optimizer over normalized circuit parameters."""

    def __init__(
        self,
        data_init: dict[str, list[Any]],
        params_list: list[str],
        n_proposal: int = 10,
        ranges: dict[str, list[float]] | None = None,
        acq_func: str = "EI",
    ):
        """Initialize the GP and acquisition function.

        Args:
            data_init: Evaluated initial designs.
            params_list: Ordered parameter names.
            n_proposal: Number of candidates returned per iteration.
            ranges: Physical bounds keyed by ``w``, ``l``, ``r``, and ``c``.
            acq_func: One of ``EI``, ``LogEI``, or ``UCB``.
        """
        if ranges is None:
            raise ValueError("Parameter ranges are required for BO.")
        if acq_func not in ACQF_1:
            raise ValueError(f"Unsupported acquisition function: {acq_func}")

        self.NUM_RESTARTS = 100
        self.RAW_SAMPLES = 512
        self.params_list = params_list
        self.n_proposal = n_proposal
        self.n_params = len(self.params_list)
        self.ranges = ranges
        self.xbounds = self.get_x_bounds()

        print("Initializing BO GP with data_init...")
        _, lb_y = get_xy(data_init, self.normalize_x)
        self.initialize_gp(data_init)
        self.standard_xbounds = torch.tensor(
            [[0.0] * self.n_params, [1.0] * self.n_params],
            dtype=torch.double,
        )

        if self.n_proposal == 1:
            self.acqf_func = ACQF_1[acq_func]
        else:
            self.acqf_func = ACQF_q[acq_func]
        self.acqf = self.acqf_func(self.gp, lb_y.max())

    def get_x_bounds(self) -> torch.Tensor:
        """Return physical parameter bounds as a ``2 x d`` tensor."""
        bounds = torch.zeros(2, self.n_params, dtype=torch.double)
        for j, param in enumerate(self.params_list):
            lower, upper = get_param_bounds(param, self.ranges)
            bounds[0, j] = lower
            bounds[1, j] = upper
        return bounds

    def normalize_x(self, x: torch.Tensor) -> torch.Tensor:
        """Map physical parameter values to ``[0, 1]``."""
        normalized = (x - self.xbounds[0]) / (
            self.xbounds[1] - self.xbounds[0]
        )
        return torch.clamp(normalized, 0, 1)

    def denormalize_x(self, normalized_x: torch.Tensor) -> torch.Tensor:
        """Map normalized values back to physical parameter units."""
        return normalized_x * (
            self.xbounds[1] - self.xbounds[0]
        ) + self.xbounds[0]

    def initialize_gp(
        self,
        data_collected: dict[str, list[Any]],
        state_dict: dict[str, Any] | None = None,
        ulb_x: torch.Tensor | None = None,
        ulb_y: torch.Tensor | None = None,
    ) -> None:
        """Build the GP model from evaluated and optional unlabeled data."""
        lb_x, lb_y = get_xy(data_collected, self.normalize_x)
        lb_y = lb_y.view(-1, 1)
        if ulb_x is None:
            train_x = lb_x
        else:
            train_x = torch.cat([lb_x, ulb_x], dim=0)
        if ulb_y is None:
            train_obj = lb_y
        else:
            train_obj = torch.cat([lb_y, ulb_y], dim=0)
        gp = SingleTaskGP(train_x, train_obj)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        if state_dict is not None:
            gp.load_state_dict(state_dict)
        self.gp = gp
        self.mll = mll

    def propose_params(
        self,
        data_collected: dict[str, list[Any]],
        ulb_x: torch.Tensor | None = None,
        ulb_y: torch.Tensor | None = None,
    ) -> list[dict[str, str]]:
        """Fit the GP and optimize the acquisition function."""
        if self.n_proposal == 0:
            return []
        lb_x, lb_y = get_xy(data_collected, self.normalize_x)
        self.initialize_gp(
            data_collected,
            state_dict=self.gp.state_dict(),
            ulb_x=ulb_x,
            ulb_y=ulb_y,
        )
        state_dict = self.gp.state_dict()
        try:
            fit_gpytorch_mll(self.mll)
        except Exception as e:
            print("Error in fitting GP model: ", e)
            self.gp.load_state_dict(state_dict)

        self.acqf = self.acqf_func(self.gp, lb_y.max())
        candidates, _ = optimize_acqf(
            self.acqf,
            bounds=self.standard_xbounds,
            q=self.n_proposal,
            num_restarts=self.NUM_RESTARTS,
            raw_samples=self.RAW_SAMPLES,
            options={"batch_limit": 100, "maxiter": 500},
        )

        params_proposed_numpy = self.denormalize_x(candidates).numpy()
        params_proposed_numpy = np.clip(
            params_proposed_numpy,
            self.xbounds[0].numpy(),
            self.xbounds[1].numpy(),
        )

        for candidate in candidates:
            candidate = candidate.reshape(1, -1)
            print("Candidate: ", candidate)

        params_proposed = []
        for candidate in params_proposed_numpy:
            params_query = nparray_to_params_dict(candidate, self.params_list)
            params_proposed.append(params_query)

        return params_proposed
