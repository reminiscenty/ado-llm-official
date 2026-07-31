from pathlib import Path
import shutil
from typing import Any

from backend.hspice.comp.interface import hspice_eval_f_comp
from backend.hspice.amp2.interface import hspice_eval_f_amp2


def validate_hspice_setup(args: dict[str, Any]) -> Path:
    """Validate external HSPICE and netlist prerequisites.

    HSPICE itself, circuit netlists, and technology/model files are not
    distributed with this repository.
    """
    if shutil.which("hspice") is None:
        raise RuntimeError(
            "The `hspice` executable was not found on PATH. Install or load a "
            "licensed Synopsys HSPICE environment before running experiments."
        )

    ckt_dir = Path(args["ckt_dir"])
    netlist_path = ckt_dir / f"{args['ckt_name']}.sp"
    if not netlist_path.is_file():
        raise FileNotFoundError(
            f"Missing circuit netlist: {netlist_path}. Netlists and technology "
            "model files are not included in this repository; provide your own "
            "compatible files under the configured `ckt_dir`."
        )
    return netlist_path


def hspice_eval_f(
    point_to_evaluate: str, args: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    """Evaluate one parameter set with the configured circuit backend."""
    validate_hspice_setup(args)
    if args["ckt_name"] == "comp":
        return hspice_eval_f_comp(point_to_evaluate, args)
    if args["ckt_name"] == "amp2":
        return hspice_eval_f_amp2(point_to_evaluate, args)
    raise ValueError(f"Unsupported circuit backend: {args['ckt_name']}")