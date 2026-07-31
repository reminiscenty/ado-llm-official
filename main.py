import argparse
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from backend.llm import gpt
from core import proposer, sampler, task
from core.bo import GPBO
from utils import set_seed


METHOD_NAMES = {0: "BO", 1: "LLM", 2: "ADO-LLM"}
CIRCUIT_NAMES = {0: "amp2", 1: "comp"}
ACQUISITION_FUNCTIONS = {0: "EI", 1: "LogEI", 2: "UCB"}
GPT_VERSIONS = {
    0: "gpt-3.5-turbo",
    1: "gpt-4-turbo",
    2: "gpt-4o-mini",
    3: "gpt-5.6-luna",
}


class Pipeline:
    """Coordinate initialization, LLM proposals, BO proposals, and simulation."""

    def __init__(
        self,
        path_task_setting: str,
        n_init_data: int = 1,
        init_method: str = "zeroshot",
        n_sample: int = 10,
        sample_method: str = "all",
        shuffle_sample: bool = False,
        n_proposal_llm: int = 1,
        n_proposal_bo: int = 1,
        n_itr: int = 10,
        gpt_version: str = "4o-mini",
        openai_api_seed: int = 42,
        path_checkpoints: str = "./checkpoints",
        log_info: bool = False,
        acq_func: str = "EI",
        debug_mode: bool = False,
    ):
        """Initialize an ADO-LLM optimization pipeline.

        Args:
            path_task_setting: Path to a task JSON configuration.
            n_init_data: Number of initial evaluated designs.
            init_method: One of ``zeroshot``, ``random``, or ``fixed``.
            n_sample: Number of demonstrations supplied to the LLM.
            sample_method: One of ``topk``, ``random``, ``all``, or ``mixed``.
            shuffle_sample: Whether to shuffle sampled demonstrations.
            n_proposal_llm: LLM proposals per iteration; zero disables them.
            n_proposal_bo: BO proposals per iteration; zero disables them.
            n_itr: Number of optimization iterations.
            gpt_version: OpenAI model identifier.
            openai_api_seed: Seed forwarded to the OpenAI API.
            path_checkpoints: Base output path.
            log_info: Whether to print proposal details.
            acq_func: BO acquisition function name.
            debug_mode: Whether to print full LLM prompts and responses.
        """
        if init_method not in {"zeroshot", "random", "fixed"}:
            raise ValueError(f"Unsupported initialization method: {init_method}")
        if sample_method not in {"topk", "random", "all", "mixed"}:
            raise ValueError(f"Unsupported sampling method: {sample_method}")

        self.gpt_version = gpt_version
        self.openai_api_seed = openai_api_seed
        self.backend = gpt.GPT(model=gpt_version, seed=openai_api_seed, debug_mode=debug_mode)

        self.data_dict_keys = ["params", "metrics", "targets", "aux_info", "params_numpy"]
        self.task = task.Task(path_task_setting, self.data_dict_keys, backend=self.backend)
        self.backend.setSystemMessage(self.task.task_context)
        self.log_info = log_info

        self.name_task = self.task.name_task

        self.params_list = self.task.params_list
        self.n_params = self.task.n_params
        self.n_metrics = self.task.n_metrics
        self.ranges = self.task.ranges

        self.n_init_data = n_init_data
        self.init_method = init_method
        self.n_sample = n_sample
        self.sample_method = sample_method
        self.shuffle_sample = shuffle_sample

        self.n_proposal_llm = n_proposal_llm
        self.n_proposal_bo = n_proposal_bo

        self.n_itr = n_itr

        self.path_checkpoints = (
            path_checkpoints + f"_{self.name_task}_{gpt_version}"
        )
        os.makedirs(self.path_checkpoints, exist_ok=True)

        self.checkpoints: dict[str, Any] = {
            "gpt_version": self.gpt_version,
            "openai_api_seed": openai_api_seed,
        }

        (
            self.data_collected,
            self.data_collected_llm,
            self.data_collected_bo,
        ) = self.task.initialize(
            self.n_init_data, self.init_method, log_info=True
        )

        self.llm_proposer = proposer.LLMProposer(
            params_list=self.params_list,
            task_context=self.task.task_context,
            backend=self.backend,
            n_proposal=self.n_proposal_llm,
            ranges=self.ranges,
            example_keys=["params", "metrics", "aux_info"],
            metrics_unit=self.task.metrics_unit,
            debug_mode=debug_mode,
        )

        self.sampler = sampler.Sampler(
            n_sample=self.n_sample,
            method=self.sample_method,
            shuffle=self.shuffle_sample,
        )
        self.bo_proposer = GPBO(
            data_init=self.data_collected,
            params_list=self.params_list,
            n_proposal=self.n_proposal_bo,
            ranges=self.ranges,
            acq_func=acq_func,
        )

        self.target_best = max(self.data_collected["targets"])
        print(f"Initial best target value: {self.target_best:.2f}")

    def optimize(self) -> None:
        """Run the configured optimization loop and print the best design."""
        for itr in range(self.n_itr):
            print(f"Current iteration: {itr + 1}.")

            data_pro = self.sampler.sample(self.data_collected)

            params_proposed_llm = self.llm_proposer.propose_params(data_pro)
            for params_query_llm in params_proposed_llm:
                data_new_llm = self.task.evaluate(params_query_llm, log_info=True)
                self.update_data(self.data_collected_llm, data_new_llm)
                self.update_data(self.data_collected, data_new_llm)

            params_proposed_bo = self.bo_proposer.propose_params(self.data_collected)
            for params_query_bo in params_proposed_bo:
                data_new_bo = self.task.evaluate(params_query_bo, log_info=False)
                self.update_data(self.data_collected_bo, data_new_bo)
                self.update_data(self.data_collected, data_new_bo)

            self.target_best = max(self.data_collected["targets"])
            idx_best = np.argmax(self.data_collected["targets"])
            print(f"At iteration {itr + 1}, the best target value is: {self.target_best:.6f}")
            print(f"metrics: {self.data_collected['metrics'][idx_best]}")

            if self.log_info:
                print(f"Proposed parameters by BO: {params_proposed_bo}")
                print(f"LLM demonstrations: {data_pro}")
                print(f"Proposed parameters by LLM: {params_proposed_llm}")
            self.save_checkpoints(itr)

        idx_best = np.argmax(self.data_collected["targets"])
        for k, v in self.data_collected.items():
            print(f"{k}: {v[idx_best]}")

    @staticmethod
    def update_data(
        data_collected: dict[str, list[Any]], data_new: dict[str, Any]
    ) -> None:
        """Append one evaluated design to a collection."""
        for k, v in data_new.items():
            data_collected[k].append(v)

    def save_checkpoints(self, itr: int) -> None:
        """Persist the latest collected data and iteration number."""
        self.checkpoints["data_collected"] = self.data_collected
        self.checkpoints["itr"] = itr + 1

        checkpoint_path = Path(self.path_checkpoints) / "checkpoints.pkl"
        with checkpoint_path.open("wb") as f:
            pickle.dump(self.checkpoints, f)

    def load_checkpoints(self) -> dict[str, Any]:
        """Load collected data from this pipeline's checkpoint directory."""
        checkpoint_path = Path(self.path_checkpoints) / "checkpoints.pkl"
        with checkpoint_path.open("rb") as f:
            checkpoints = pickle.load(f)
        self.data_collected = checkpoints["data_collected"]
        print(f"Loaded checkpoints from iteration {checkpoints['itr']}")
        return checkpoints


def load_checkpoints(path_checkpoints: str) -> dict[str, Any]:
    """Load a checkpoint dictionary from an output directory."""
    checkpoint_path = Path(path_checkpoints) / "checkpoints.pkl"
    with checkpoint_path.open("rb") as f:
        checkpoints = pickle.load(f)
    return checkpoints


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run BO, LLM-only, or ADO-LLM analog design optimization."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--method", type=int, choices=METHOD_NAMES, default=0)
    parser.add_argument("--ckt", type=int, choices=CIRCUIT_NAMES, default=0)
    parser.add_argument("--acqf", type=int, choices=ACQUISITION_FUNCTIONS, default=0)
    parser.add_argument("--gpt", type=int, choices=GPT_VERSIONS, default=0)
    parser.add_argument("--debug_mode", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Configure and run an optimization experiment."""
    log_info = False
    sample_method = "topk"
    n_init_data = 5
    n_simulations = 100
    args = parse_args()

    seed = args.seed
    method_name = METHOD_NAMES[args.method]
    ckt_name = CIRCUIT_NAMES[args.ckt]
    gpt_version = GPT_VERSIONS[args.gpt]
    set_seed(seed)
    openai_api_seed = seed
    acqf_func = ACQUISITION_FUNCTIONS[args.acqf]
    print(
        f"method: {method_name}, circuit: {ckt_name}, "
        f"acquisition function: {acqf_func}"
    )

    if method_name == "BO":
        n_proposal_llm = 0
        n_proposal_bo = 5
        init_method = "random"
    elif method_name == "LLM":
        n_proposal_llm = 5
        n_proposal_bo = 0
        init_method = "zeroshot"
    else:
        n_proposal_llm = 1
        n_proposal_bo = 4
        init_method = "zeroshot"

    path_checkpoints = f"./results/{method_name}_{acqf_func}/seed{seed}"
    proposals_per_iteration = n_proposal_llm + n_proposal_bo
    n_iter = n_simulations // proposals_per_iteration
    task_str = f"tasks/{ckt_name}/{ckt_name}.json"

    llmbo = Pipeline(
        task_str,
        openai_api_seed=openai_api_seed,
        init_method=init_method,
        gpt_version=gpt_version,
        n_init_data=n_init_data,
        n_itr=n_iter,
        sample_method=sample_method,
        n_sample=5,
        n_proposal_llm=n_proposal_llm,
        n_proposal_bo=n_proposal_bo,
        path_checkpoints=path_checkpoints,
        log_info=log_info,
        acq_func=acqf_func,
        debug_mode=args.debug_mode,
    )
    llmbo.optimize()


if __name__ == "__main__":
    main()
