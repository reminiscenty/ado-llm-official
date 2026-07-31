from typing import Any

from utils.dataset import get_subset


class Sampler:
    """Select evaluated designs to use as LLM demonstrations."""

    def __init__(
        self, n_sample: int, method: str = "topk", shuffle: bool = False
    ):
        """Configure demonstration sampling.

        Args:
            n_sample: Maximum number of examples.
            method: One of ``topk``, ``random``, ``mixed``, or ``all``.
            shuffle: Reserved for compatibility with earlier experiments.
        """
        self.n_sample = n_sample
        self.method = method
        self.shuffle = shuffle

    def sample(
        self, data_collected: dict[str, list[Any]]
    ) -> dict[str, list[Any]]:
        """Return a subset of collected designs."""
        n_topk = 0
        n_random = 0
        n_data = len(data_collected["targets"])
        n_max = min(self.n_sample, n_data)
        if self.method == "topk":
            n_topk = n_max
            n_random = 0
        elif self.method == "random":
            n_random = n_max
            n_topk = 0
        elif self.method == "all":
            return data_collected
        elif self.method == "mixed":
            n_topk = n_max // 2
            n_random = n_max - n_topk
        return get_subset(data_collected, n_topk, n_random)


