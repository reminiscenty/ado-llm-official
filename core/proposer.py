from typing import Any

from core.fewshot_agent import FewShotAgent
from utils.parser import extract_param_sets_from_response


class LLMProposer(FewShotAgent):
    """Generate circuit parameter candidates from in-context demonstrations."""

    def __init__(
        self,
        params_list: list[str],
        task_context: str,
        backend: Any,
        example_keys: list[str],
        metrics_unit: dict[str, str],
        n_proposal: int = 5,
        ranges: dict[str, list[float]] | None = None,
        max_request_attempt: int = 3,
        debug_mode: bool = False,
    ):
        super().__init__(example_keys=example_keys, metrics_unit=metrics_unit)
        self.task_context = task_context
        self.params_list = params_list
        self.params_units = ["f", "m", "n", "p", "u", "k", "G", "M"]
        self.n_params = len(self.params_list)

        self.backend = backend
        self.max_request_attempt = max_request_attempt

        self.ranges = ranges
        self.n_proposal = n_proposal

        self.debug_mode = debug_mode

    def propose_params(
        self, data_collected: dict[str, list[Any]]
    ) -> list[dict[str, str]]:
        """Request and parse the configured number of LLM proposals."""
        if self.n_proposal == 0:
            return []

        examples = self.generate_examples(data_collected)
        params_proposed = []
        for _ in range(self.max_request_attempt):
            prompt = self.generate_prompt(examples, None)
            responses = self.backend.request(prompt)
            params_proposed.extend(self.parse_llm_responses(responses))

            if self.debug_mode:
                print(f"***Prompt***\n\n{prompt}")
                print(f"***Response***\n\n{responses[0] if responses else '<empty>'}")

            if len(params_proposed) >= self.n_proposal:
                break

        if len(params_proposed) < self.n_proposal:
            raise ValueError(
                "LLM failed to propose enough valid parameter sets."
            )

        return params_proposed[: self.n_proposal]

    def generate_prefix(self, prompt_input: Any) -> str:
        """Build the instruction before the few-shot examples."""
        prefix = (
            "Optimize these circuit parameters "
            f"{self.params_list} to satisfy the **Performance Specifications**. "
        )
        prefix += "**Examples**\n"
        return prefix

    def generate_suffix(self, prompt_input: Any) -> str:
        """Build the instruction after the few-shot examples."""
        suffix = (
            "Your designs should be neither too far from nor identical to the "
            "examples.\n"
            f"Propose **{self.n_proposal}** parameter sets within the "
            "**Design Space** that satisfy the **Performance Specifications**.\n"
            "Use the same parameter format as the examples."
        )
        return suffix

    def extract_params(self, response: str) -> list[dict[str, str]]:
        """Extract complete parameter sets from one model response."""
        return extract_param_sets_from_response(
            response,
            self.params_list,
            params_units=self.params_units,
            max_sets=self.n_proposal,
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
