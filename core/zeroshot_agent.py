from typing import Any

from utils.parser import extract_param_sets_from_response


class ZeroShotAgent:
    """Generate initial designs from the task context without demonstrations."""

    def __init__(
        self,
        params_list: list[str],
        task_context: str,
        backend: Any,
        max_request_attempt: int = 3,
        debug_mode: bool = False,
    ):
        self.task_context = task_context
        self.params_list = params_list
        self.params_units = ["f", "m", "n", "p", "u", "k", "G", "M"]
        self.n_params = len(self.params_list)

        self.backend = backend
        self.max_request_attempt = max_request_attempt
        self.debug_mode = debug_mode

    def generate_prompt(self, n_init_data: int) -> str:
        """Generate a prompt for zero-shot warm start.

        Args:
            n_init_data: Number of initial designs requested.
        """
        prompt = self.task_context
        prompt += (
            "\nBased on the task description, propose parameter values likely to "
            "achieve a high objective value. "
        )
        prompt += (
            f"Conclude with {n_init_data} recommended parameter sets using the "
            "format shown below.\n"
        )
        prompt += "**Examples**\n"
        prompt += ".param " + " ".join(
            f"{param}=[]" for param in self.params_list
        )
        prompt += (
            f"\nPropose **{n_init_data}** initial parameter sets within the "
            "**Design Space** that satisfy the **Performance Specifications**.\n"
        )

        return prompt

    def extract_params(self, response: str) -> list[dict[str, str]]:
        """Extract complete parameter sets from one response."""
        return extract_param_sets_from_response(
            response,
            self.params_list,
            params_units=self.params_units,
        )

    def parse_llm_responses(
        self, responses: list[str] | None
    ) -> list[dict[str, str]]:
        """Parse all returned model generations."""
        params_parsed = []
        if responses is None:
            print("API call failed.")
            return params_parsed

        for response in responses:
            params_parsed.extend(self.extract_params(response))
        return params_parsed

    def propose_params(self, n_init_data: int) -> list[dict[str, str]]:
        """Request enough valid designs to initialize optimization."""
        params_init = []

        for _ in range(self.max_request_attempt):
            prompt = self.generate_prompt(n_init_data)

            responses = self.backend.request(prompt)
            if not responses or not responses[0]:
                print("responses: <empty>")
                continue
            params_init.extend(self.parse_llm_responses(responses))

            if self.debug_mode:
                print(f"***Prompt***\n\n{prompt}")
                print(f"***Response***\n\n{responses[0]}")

            if len(params_init) >= n_init_data:
                break

        if len(params_init) < n_init_data:
            raise ValueError(
                "LLM failed to propose enough valid parameter sets."
            )
        return params_init[:n_init_data]