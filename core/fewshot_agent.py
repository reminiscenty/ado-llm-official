from typing import Any

from langchain.prompts import PromptTemplate, FewShotPromptTemplate

from utils.parser import format_float


class FewShotAgent:
    """Base class for constructing few-shot circuit-design prompts."""

    def __init__(
        self,
        example_keys: list[str],
        metrics_unit: dict[str, str],
    ):
        self.example_keys = example_keys
        self.metrics_unit = metrics_unit

        example_template = """Example: {number}
.param: {params}
Metrics: {metrics}
Transistor operating region: {aux_info}
"""

        self.example_prompt = PromptTemplate(
            input_variables=self.example_keys,
            template=example_template
        )

    def generate_prefix(self, prompt_input: Any) -> str:
        raise NotImplementedError

    def generate_suffix(self, prompt_input: Any) -> str:
        raise NotImplementedError

    def generate_prompt(
        self, examples: list[dict[str, Any]], prompt_input: Any
    ) -> str:
        """Render the complete few-shot prompt."""
        prefix = self.generate_prefix(prompt_input)
        suffix = self.generate_suffix(prompt_input)

        few_shot_prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=self.example_prompt,
            prefix=prefix,
            suffix=suffix,
            input_variables=["prompt_input"],
            example_separator="\n"
        )

        prompt = few_shot_prompt.format(prompt_input=f"{prompt_input}")

        return prompt

    def generate_examples(
        self, data_collected: dict[str, list[Any]]
    ) -> list[dict[str, Any]]:
        """Convert evaluated designs into prompt examples."""
        n_data_collected = len(data_collected["params"])
        examples = []

        for i in range(n_data_collected):
            example = {"number": i}
            for example_key in self.example_keys:
                example_str = ""
                if example_key == "params":
                    for k, v in data_collected["params"][i].items():
                        example_str += f"{k}={v} "
                elif example_key == "metrics":
                    for k, v in data_collected["metrics"][i].items():
                        example_str += f"{k}={format_float(v)} {self.metrics_unit[k]}, "
                elif example_key == "targets":
                    example_str = f"FOM={data_collected['targets'][i]:.2f}"
                elif example_key == "aux_info":
                    example_str = data_collected["aux_info"][i]
                else:
                    raise ValueError(f"Invalid example key: {example_key}")
                example[example_key] = example_str
            examples.append(example)

        return examples

    def parse_llm_responses(self, responses: list[str] | None) -> Any:
        raise NotImplementedError
