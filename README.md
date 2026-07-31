# ADO-LLM

Official implementation of:

> **ADO-LLM: Analog Design Bayesian Optimization with In-Context Learning of Large Language Models**
>
> Yuxuan Yin, Yu Wang, Boxun Xu, and Peng Li
>
> IEEE/ACM International Conference on Computer-Aided Design (ICCAD), 2024
> [ACM Digital Library](https://dl.acm.org/doi/10.1145/3676536.3676816) · [arXiv](https://arxiv.org/abs/2406.18770)

ADO-LLM combines large-language-model proposals with Gaussian-process Bayesian
optimization (BO) for analog circuit sizing. The LLM contributes circuit-design
knowledge and quickly suggests promising regions; BO contributes systematic
exploration, exploitation, and diverse evaluated examples. In each iteration,
new designs from both sources are evaluated by a circuit simulator and added to
the shared optimization history.

## Important: external circuit and simulator assets are not included

This repository is **not runnable out of the box**.

The following assets are intentionally not distributed:

- circuit netlists (`*.sp`);
- foundry or technology model decks and related include files;
- the Synopsys HSPICE executable, license, or installation files;
- generated HSPICE outputs (`*.lis`, `*.ma*`, `*.mt*`, `*.dp*`);
- the original experiment result data.

The repository contains the ADO-LLM optimization implementation, prompt and task
templates, HSPICE integration/parsing code, and objective functions. To run an
experiment, you must supply a licensed HSPICE installation and compatible
netlists/model files. You are responsible for ensuring that you have permission
to use and distribute those external assets.

The `ckts/` directory is ignored by Git. Create it locally as described below.
The task descriptions intentionally contain `TODO` placeholders where
netlist-specific information must be supplied.

## Method overview

ADO-LLM maintains one shared collection of evaluated designs:

1. Initialize the collection with random designs (BO) or zero-shot LLM designs
   (LLM and ADO-LLM).
2. Select high-quality evaluated designs as in-context demonstrations.
3. Ask the LLM for new parameter sets.
4. Fit a Gaussian process to all evaluated parameter/figure-of-merit pairs.
5. Optimize an acquisition function to obtain new BO candidates.
6. Clip every proposed parameter to the configured physical design bounds.
7. Write the candidate to `param.inc`, run HSPICE, parse its outputs, and compute
   the scalar figure of merit.
8. Add the evaluated designs to the history and repeat.

The supplied entry point compares three modes:

- `BO`: five BO candidates per iteration; random initialization.
- `LLM`: five LLM candidates per iteration; zero-shot LLM initialization.
- `ADO-LLM`: one LLM candidate plus four BO candidates per iteration; zero-shot
  LLM initialization.

The default budget is five initial designs followed by 100 iterative
simulations (20 iterations × 5 proposals).

## Repository layout

```text
.
├── main.py                         # Experiment entry point and optimization loop
├── core/
│   ├── task.py                     # Task loading, bounds, simulation evaluation
│   ├── proposer.py                 # In-context LLM proposal generation
│   ├── zeroshot_agent.py           # LLM warm-start generation
│   ├── fewshot_agent.py            # Prompt example formatting
│   ├── bo.py                       # Active GP/BoTorch optimizer
│   └── sampler.py                  # Top-k/random demonstration selection
├── backend/
│   ├── llm/gpt.py                  # OpenAI Chat Completions client
│   └── hspice/
│       ├── interface.py            # Circuit dispatch and prerequisite checks
│       ├── amp2/                   # Amplifier parser and objective
│       └── comp/                   # Comparator parser and objective
├── tasks/
│   ├── amp2/                       # Amplifier task JSON and prompt template
│   └── comp/                       # Comparator task JSON and prompt template
├── utils/                          # Data and engineering-unit conversion helpers
├── environment.yml                # Reproducible Conda environment
└── requirements.txt               # Pip package snapshot
```

Some legacy or exploratory modules remain in `core/`; `main.py` uses
`core/bo.py` as its BO implementation.

## Requirements

- Linux (the exported environment was created on Linux x86-64)
- Conda or Miniconda
- Python 3.10
- A valid OpenAI API key and access to the selected model
- A licensed Synopsys HSPICE installation available as `hspice` on `PATH`
- Circuit netlists and all technology/model include files referenced by them

HSPICE is commercial software and cannot be installed through
`environment.yml`.

## Installation

Clone the repository and create the exported environment:

```bash
git clone <repository-url>
cd ado-llm-official
conda env create -f environment.yml
conda activate ado-llm
```

Alternatively, create a Python 3.10 environment and install the pip snapshot:

```bash
conda create -n ado-llm python=3.10 -y
conda activate ado-llm
python -m pip install -r requirements.txt
```

The full Conda environment is recommended because PyTorch, BoTorch, and
GPyTorch compatibility is version-sensitive.

Configure OpenAI:

```bash
export OPENAI_API_KEY="your-api-key"
```

The current pipeline constructs the OpenAI client for every method, including
pure BO, so `OPENAI_API_KEY` must currently be set for all runs.

Load your institution's or organization's HSPICE environment and verify it:

```bash
which hspice
hspice -v
```

Exact setup commands vary by HSPICE installation and license server.

## Supplying the missing circuit assets

Create this local structure:

```text
ckts/
├── amp2/
│   ├── amp2.sp
│   └── ... technology/model files required by amp2.sp
└── comp/
    ├── comp.sp
    └── ... technology/model files required by comp.sp
```

Do not commit proprietary netlists or model decks unless their licenses permit
it.

Each netlist must:

- include `param.inc` from its own circuit directory;
- use parameter names matching the corresponding task JSON;
- define the analyses and measurements expected by the parser;
- produce HSPICE output files in the positions/formats expected by the
  corresponding `backend/hspice/<circuit>/interface.py`;
- make every technology/model include path resolvable in your environment.

During evaluation the code overwrites:

```text
ckts/<circuit>/param.inc
```

Do not run two experiments concurrently against the same circuit directory,
because they would race on `param.inc` and the simulator output files.

### Amplifier backend expectations

The amplifier task is configured in `tasks/amp2/amp2.json`. Its backend expects:

- netlist: `ckts/amp2/amp2.sp`;
- parameters: `w1`–`w6`, `l1`–`l6`, `r1`, and `c1`;
- metrics: gain, CMRR, gain-bandwidth product, phase margin, and power;
- output files including `.ma0`, `.ma1`, and `.dp0`.

The parser converts GBW to MHz and power to µW before objective calculation.

### Comparator backend expectations

The comparator task is configured in `tasks/comp/comp.json`. Its backend
expects:

- netlist: `ckts/comp/comp.sp`;
- parameters: `w1`–`w6` and `l1`–`l6`;
- metrics: gain, unity-gain frequency, hysteresis error, offset, and power;
- output files including `.ma0`, `.mt0`, and `.dp0`.

The parser converts unity-gain frequency to MHz, hysteresis and offset to mV,
and power to µW.

The existing parsers use fixed line positions in HSPICE output files. If your
HSPICE version or netlist emits a different layout, update the appropriate
interface parser before running an optimization.

## Completing the task descriptions

Before using either supplied task, replace the `TODO` sections in:

```text
tasks/amp2/amp2_description.txt
tasks/comp/comp_description.txt
```

Provide:

- the relevant core circuit topology or a permitted textual abstraction;
- a short explanation of the circuit's functional blocks;
- the optimization parameters and their circuit roles;
- design principles that connect sizing changes to operating regions and
  performance metrics.

These files become the LLM system context. Their technical accuracy directly
affects zero-shot initialization and iterative proposals. Do not put secrets,
licensed model contents, or API credentials in them because they are sent to
the configured LLM provider.

## Running experiments

Show all options:

```bash
python main.py --help
```

The CLI uses integer selectors:

- `--method`: `0` BO, `1` LLM, `2` ADO-LLM
- `--ckt`: `0` two-stage amplifier, `1` hysteresis comparator
- `--acqf`: `0` EI, `1` LogEI, `2` UCB
- `--gpt`: `0` GPT-3.5 Turbo, `1` GPT-4 Turbo, `2` GPT-4o mini,
  `3` GPT-5.6 Luna
- `--seed`: random seed used by NumPy, PyTorch, and the API request
- `--debug_mode`: print complete LLM prompts and responses

Model availability changes over time. Choose a model enabled for your OpenAI
project; older model identifiers may be retired.

Examples:

```bash
# ADO-LLM on the amplifier with LogEI and GPT-4o mini
python main.py --method 2 --ckt 0 --acqf 1 --gpt 2 --seed 0

# Pure BO on the comparator with expected improvement
python main.py --method 0 --ckt 1 --acqf 0 --seed 0

# LLM-only search with verbose prompts and responses
python main.py --method 1 --ckt 0 --gpt 2 --debug_mode
```

## Outputs and checkpoints

Each iteration writes:

```text
results/<METHOD>_<ACQUISITION>/seed<SEED>_<CIRCUIT>_<MODEL>/checkpoints.pkl
```

The checkpoint contains:

- the selected model identifier;
- the API seed;
- the complete evaluated data collection;
- the last completed iteration number.

`results/` is ignored by Git. Checkpoints use Python pickle and must only be
loaded from trusted sources.

The evaluated collection stores:

- `params`: engineering-unit parameter dictionaries;
- `metrics`: simulator metrics;
- `targets`: scalar figures of merit maximized by the optimizer;
- `aux_info`: transistor operating-region feedback;
- `params_numpy`: numeric parameter vectors in SI units.

## Task configuration

Each `tasks/<name>/<name>.json` file defines:

- the task-description path and circuit directory;
- ordered parameter and metric names;
- display units used in LLM examples;
- physical design bounds;
- normalization ranges;
- metric weights;
- performance specifications;
- fallback values for failed measurements.

Parameter order must be consistent across the JSON configuration, netlist,
LLM output, and simulator integration.

All LLM and BO candidates are clipped to the physical bounds before simulation.
The stored parameter dictionary is the clipped design, so the GP is trained on
the design that was actually simulated.

## Adding another circuit

To add a circuit named `myckt`:

1. Create `tasks/myckt/myckt.json` and a task-description text file.
2. Define ordered parameters, units, bounds, metrics, specifications, failure
   values, and objective weights.
3. Implement `backend/hspice/myckt/interface.py` to run HSPICE and parse metrics.
4. Implement the circuit objective and operating-region parser.
5. Add dispatch for `myckt` in `backend/hspice/interface.py`.
6. Add a CLI circuit mapping in `main.py`.
7. Place your local netlist at `ckts/myckt/myckt.sp`.
8. Test one manual simulator evaluation before starting an optimization.

## Troubleshooting

### `OPENAI_API_KEY` is missing

Export the key in the shell that launches `main.py`. Verify that the selected
model is available to that OpenAI project and that the project has quota.

### Empty LLM responses

Reasoning models may consume much of their completion budget internally. The
backend allocates a larger budget for GPT-5 and `o`-series models and retries
empty generations. If failures continue, inspect output with `--debug_mode`,
try another enabled model, and confirm API quota.

### `hspice` executable not found

Load your local HSPICE module/environment and ensure `which hspice` succeeds.
The repository cannot provide the simulator or its license.

### Missing `ckts/<name>/<name>.sp`

Supply your own compatible netlist. The original circuit netlists are not
included.

### Missing `.ma0`, `.ma1`, `.mt0`, or `.dp0`

Inspect the HSPICE `.lis` output first. Typical causes are a failed simulation,
missing model include, different measurement names, or an output layout that
does not match the parser.

### LLM proposals outside the design space

The pipeline clips every parameter to the configured bounds before simulation.
When logging is enabled, clipped parameter names are reported.

### GP fitting warnings

BO fitting may be sensitive to repeated designs or nearly constant objectives.
The implementation keeps the previous GP state if fitting fails. Check failed
simulations and duplicate points before increasing BO optimizer settings.

## Authors, affiliation, and contact

The authors are affiliated with the **Department of Electrical and Computer
Engineering, University of California, Santa Barbara (ECE/UCSB)**:

- Yuxuan Yin
- Yu Wang
- Boxun Xu
- Prof. Peng Li

For questions about ADO-LLM or research collaboration, contact Prof. Peng Li at
[lip@ucsb.edu](mailto:lip@ucsb.edu).

Copyright © 2024 Yuxuan Yin, Yu Wang, Boxun Xu, and Peng Li. All authors are
copyright holders of this implementation.

## Citation

If this repository or the ADO-LLM method is useful in your work, cite:

```bibtex
@inproceedings{yin2024adollm,
  title     = {ADO-LLM: Analog Design Bayesian Optimization with In-Context
               Learning of Large Language Models},
  author    = {Yin, Yuxuan and Wang, Yu and Xu, Boxun and Li, Peng},
  booktitle = {Proceedings of the 43rd IEEE/ACM International Conference on
               Computer-Aided Design},
  year      = {2024},
  pages     = {1--9},
  doi       = {10.1145/3676536.3676816}
}
```

For the paper and authoritative publication metadata, see
[DOI 10.1145/3676536.3676816](https://doi.org/10.1145/3676536.3676816).
