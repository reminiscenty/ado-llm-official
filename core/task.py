import json
from pathlib import Path
from typing import Any

import numpy as np

import utils
from backend.hspice.interface import hspice_eval_f
from core.zeroshot_agent import ZeroShotAgent


class Task:
    """Circuit optimization task backed by an HSPICE evaluation."""

    def __init__(
        self,
        path_task_setting: str,
        data_dict_keys: list[str],
        backend: Any,
    ):
        with Path(path_task_setting).open(encoding="utf-8") as file:
            self.task_setting = json.load(file)
        self.data_dict_keys = data_dict_keys

        self.name_task = self.task_setting["ckt_name"]

        self.params_list = self.task_setting["params_list"]
        self.metrics_list = self.task_setting["metrics_list"]
        self.metrics_unit = self.task_setting["metrics_unit"]

        self.n_params = len(self.params_list)
        self.n_metrics = len(self.metrics_list)

        description_path = Path(self.task_setting["path_description"])
        self.task_context = description_path.read_text(encoding="utf-8")

        self.ranges = {
            "w": self.task_setting["width_range"],
            "l": self.task_setting["length_range"],
            "r": self.task_setting["resistance_range"],
            "c": self.task_setting["capacitance_range"],
        }

        self.zeroshot_agent = ZeroShotAgent(
            params_list=self.params_list,
            task_context=self.task_context,
            backend=backend,
        )

    def generate_params_init(
        self, n_init_data: int, init_method: str
    ) -> list[dict[str, str]]:
        """Generate initial parameter dictionaries."""
        if init_method == "zeroshot":
            params_init = self.zeroshot_agent.propose_params(n_init_data)
        elif init_method == "random":
            params_init = self.random_initialize(n_init_data)
        elif init_method == "fixed":
            params_init = self.fixed_initialize(n_init_data)
        else:
            raise ValueError(f"Unsupported initialization method: {init_method}")

        return params_init

    def initialize(
        self, n_init_data: int, init_method: str, log_info: bool = True
    ) -> tuple[dict[str, list[Any]], dict[str, list[Any]], dict[str, list[Any]]]:
        """Evaluate initial designs and create the optimization data stores."""
        data_collected = utils.create_empty_data_dict(self.data_dict_keys)
        data_collected_llm = utils.create_empty_data_dict(self.data_dict_keys)
        data_collected_bo = utils.create_empty_data_dict(self.data_dict_keys)

        params_init = self.generate_params_init(n_init_data, init_method)

        for params_init_i in params_init:
            data_init_i = self.evaluate(params_init_i, log_info=log_info)
            for k, v in data_init_i.items():
                data_collected[k].append(v)
        return data_collected, data_collected_llm, data_collected_bo

    def random_initialize(self, n_init_data: int) -> list[dict[str, str]]:
        """Generate uniformly random designs inside the configured bounds.

        Args:
            n_init_data: Number of initial designs to generate.
        """
        params_init = []
        for _ in range(n_init_data):
            params_numpy_i = np.zeros(self.n_params)

            for j, param in enumerate(self.params_list):
                low, high = utils.get_param_bounds(param, self.ranges)
                params_numpy_i[j] = np.random.uniform(low, high)

            param_dict_i = utils.nparray_to_params_dict(params_numpy_i, self.params_list)
            params_init.append(param_dict_i)

        return params_init

    def fixed_initialize(self, n_init_data: int) -> list[dict[str, str]]:
        """Return designs from a fixed initialization pool."""
        params_init_pool = [
            {
                "w1": "25u",
                "l1": "1u",
                "w2": "20u",
                "l2": "1u",
                "w3": "25u",
                "l3": "1u",
                "w4": "20u",
                "l4": "1u",
                "w5": "10u",
                "l5": "1u",
                "w6": "25u",
                "l6": "1u",
                "r1": "20k",
                "c1": "0.2p",
            }
        ]
        if n_init_data > len(params_init_pool):
            raise ValueError(
                f"Fixed initialization contains {len(params_init_pool)} design(s), "
                f"but {n_init_data} were requested."
            )
        return params_init_pool[:n_init_data]

    def evaluate(
        self, params_query: dict[str, str], log_info: bool = False
    ) -> dict[str, Any]:
        """Clip, simulate, and package one circuit design."""
        params_query, clipped_params = utils.clip_params_dict(
            params_query, self.params_list, self.ranges
        )
        if clipped_params and log_info:
            print(f"Clipped out-of-range parameters: {clipped_params}")

        params_inc = ".param " + "".join(
            f"{param}={value} " for param, value in params_query.items()
        )

        _, f_eval_raw = hspice_eval_f(params_inc, self.task_setting)

        params_numpy = utils.params_dict_to_nparray(params_query)
        metrics = self.gather_metrics(f_eval_raw)
        target = f_eval_raw["fom"]
        aux_info = f_eval_raw["aux_info"]

        data_new = {
            "params": params_query,
            "metrics": metrics,
            "targets": target,
            "aux_info": aux_info,
            "params_numpy": params_numpy,
        }

        if log_info:
            metrics = " " + "".join(f"{k}={v:.3f} " for k, v in metrics.items())
            print(f"params: {params_inc}")
            print(f"metrics: {metrics}")
            print(f"targets: {target:.3f}")
            print(f"aux_info: {aux_info}")

        return data_new

    def gather_metrics(self, fvals: dict[str, Any]) -> dict[str, float]:
        """Select configured metrics from raw simulator results."""
        return {metric: fvals[metric] for metric in self.metrics_list}
