# Exploiting More Than Symmetry in Variational Quantum Machine Learning

[![arXiv](https://img.shields.io/badge/arXiv-2606.20316-b31b1b.svg)](https://arxiv.org/abs/2606.20316)
[![Accepted at QCE26](https://img.shields.io/badge/QCE26-accepted-00629B.svg)](https://qce.quantum.ieee.org/2026/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Complete reproducibility package for the paper *Exploiting More Than Symmetry
in Variational Quantum Machine Learning*.

**Publication status:** accepted as a workshop paper at the
**4th International Workshop on Quantum Machine Learning: From Research to
Practice**, part of the 2026 IEEE International Conference on Quantum Computing
and Engineering (QCE26 / IEEE Quantum Week 2026).

| Resource | Link |
|---|---|
| Paper | [arXiv:2606.20316](https://arxiv.org/abs/2606.20316) |
| Venue | [IEEE Quantum Week 2026 (QCE26)](https://qce.quantum.ieee.org/2026/) |
| Identified manuscript | [`paper/main.pdf`](paper/main.pdf) |
| Reproduction script | [`scripts/reproduce_paper_from_outputs.sh`](scripts/reproduce_paper_from_outputs.sh) |
| Citation metadata | [`CITATION.cff`](CITATION.cff) |

## Result

The paper separates two design choices in symmetry-aware variational quantum
circuits:

1. how much of the data symmetry to impose through parameter sharing; and
2. which symmetry-preserving interactions to make trainable.

The controlled Tic-Tac-Toe benchmark shows that symmetry improves
generalization, but the larger gain comes from interactions aligned with the
task's winning-line motifs. Random-sharing and parameter-count controls rule
out simpler explanations based only on reducing the number of parameters.

## Package contents

- `paper/` — LaTeX source, identified and anonymous PDFs, and figure-generation
  code.
- `results/csv/` — checked CSV and JSON outputs used by the paper.
- `src/` — datasets, symmetry groups, circuit definitions, training, and
  validation code.
- `experiments/` — the final experiment matrix and package checks.
- `scripts/` — the artifact-based reproduction and optional full-rerun entry
  points.

## Reproduce from checked outputs

Use Python 3.11 or a compatible recent Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Run the reviewer-oriented reproduction path:

```bash
scripts/reproduce_paper_from_outputs.sh
```

The script:

1. runs the code and data sanity checks;
2. validates the committed experiment tables;
3. regenerates all paper figures; and
4. rebuilds the identified and anonymous manuscript PDFs.

A local LaTeX installation with `latexmk` is required for the final build.

## Checked evidence

The final evidence under `results/csv/` contains:

- `360` edge-ansatz subgroup rows;
- `120` edge-plus-lines subgroup rows;
- `160` ablation rows;
- `110` random-sharing control rows;
- the training-budget audit and generated summary table.

The generated paper figures are:

- `paper/fig1_4panel_standalone.pdf`
- `paper/gfx/fig2_main_evidence.pdf`
- `paper/gfx/fig3_controls.pdf`

## Optional full rerun

The checked outputs are sufficient to validate and rebuild the paper. To rerun
the final experiment matrix from scratch:

```bash
EPOCHS=100 SHARDS=5 scripts/run_consistent_paper_experiments.sh
```

Set `RUN_AUDIT=1` to rerun the training-budget audit before the final matrix.

## Fixed protocol

Each legal board is encoded on nine qubits with `RX(2*pi*g_i/3)`. Final runs
use `L=3`, `p=2`, Adam with learning rate `0.01`, batch size `15`, `30`
minibatch updates per epoch, `100` epochs, and a fixed test size of `600`.

The baseline `edge` ansatz uses orbit-shared single-qubit gates and directed
`CRY` edge interactions. The `edge+lines` ansatz adds orbit-shared `ZZZ` and
`CCRZ` interactions on the Tic-Tac-Toe winning triples.

## Scope

This is a deliberately small, exactly enumerable benchmark for isolating
architectural effects. The package supports the paper's claims about symmetry,
task-aligned interactions, and the documented controls. It does not claim
quantum advantage or establish transfer to large real-world tasks.

## Citation and license

Please cite the paper and repository using [`CITATION.cff`](CITATION.cff).
The software is released under the [MIT License](LICENSE). The manuscript and
publication PDFs remain subject to the applicable IEEE publication terms.
