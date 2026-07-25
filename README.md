# Symmetry Alone Is Not an Ansatz: Task-Aligned Interactions in Equivariant Quantum Circuits

[![arXiv](https://img.shields.io/badge/arXiv-2606.20316-b31b1b.svg)](https://arxiv.org/abs/2606.20316)
[![Accepted at QCE26](https://img.shields.io/badge/QCE26-accepted-00629B.svg)](https://qce.quantum.ieee.org/2026/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Complete reproducibility package for the paper *Symmetry Alone Is Not an
Ansatz: Task-Aligned Interactions in Equivariant Quantum Circuits*.

**Publication status:** accepted as a workshop paper at the
**4th International Workshop on Quantum Machine Learning: From Research to
Practice**, part of the 2026 IEEE International Conference on Quantum Computing
and Engineering (QCE26 / IEEE Quantum Week 2026).

| Resource | Link |
|---|---|
| Paper | [arXiv:2606.20316](https://arxiv.org/abs/2606.20316) |
| Venue | [IEEE Quantum Week 2026 (QCE26)](https://qce.quantum.ieee.org/2026/) |
| Identified manuscript (arXiv) | [`paper/main.pdf`](paper/main.pdf) |
| IEEE camera-ready manuscript | [`paper/main_ieee.pdf`](paper/main_ieee.pdf) |
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

The paper's evidence lives under `results/csv/leak_free_d4_v1/` and contains:

- `360` edge-ansatz subgroup rows;
- `120` edge-plus-lines subgroup rows;
- `160` ablation rows;
- `110` random-sharing control rows;
- the deduplicated per-run table, generated summary table, and
  `split_and_run_manifest.json` with full split and provenance records.

The top-level `results/csv/results_paper_consistent_*` tables are the
as-accepted evidence, retained for provenance; the exact relation between the
two generations is recorded in
`results/csv/leak_free_d4_v1/table_legacy_vs_leak_free.csv`.

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

## Evaluation protocol (camera-ready)

The camera-ready evaluation uses a `D4`-orbit-disjoint holdout (protocol
`d4_orbit_holdout_v1`), whose outputs live under
`results/csv/leak_free_d4_v1/`:

1. the test set is built from complete `D4` orbits, so train and test orbits
   are disjoint by construction (zero exact and zero orbit overlap);
2. the test size is `348`, the maximum feasible balanced orbit-complete
   holdout, fixed and shared across every run; and
3. no overlap fallback is permitted.

The earlier as-accepted split allowed train/test overlap through an overlap
fallback and through test boards whose `D4` orbit intersected the training
set; because the models are `D4`-equivariant, orbit overlap acts as test-set
leakage. Everything else — model matrix, train sizes, seeds, optimizer, and
the 100-epoch budget — is identical across the two generations, with no
tuning inside the re-evaluation. Headline quantities at train size `600`:

| Quantity | As accepted | Camera-ready |
|---|---|---|
| edge/none test accuracy | 0.627 | 0.647 |
| edge/`D4` test accuracy | 0.687 | 0.678 |
| edge+lines/`D4` test accuracy | 0.785 | 0.761 |
| `D4` gain over none | 6.1 pp | 3.2 pp |
| Line-interaction gain over edge/`D4` | 9.8 pp | 8.3 pp |
| Orbit sharing vs random sharing | 10.0 pp | 6.9 pp |

The paper's ordering of effects is unchanged: task-aligned line interactions
remain the dominant gain. Full per-run provenance — split indices and
digests, source and environment fingerprints, and the cross-generation
comparison tables — is recorded in
`results/csv/leak_free_d4_v1/split_and_run_manifest.json`.

The manuscripts, figures, and appendix are built from these tables. To rerun
the matrix from scratch:

```bash
SHARDS=5 scripts/run_leak_free_paper_experiments.sh
```

To validate the committed evidence tables:

```bash
python3 -m experiments.leak_free_paper_experiments --stage validate
```

The validate stage recomputes source and environment fingerprints, so it
passes bit-exactly only on the original source tree and package versions
recorded in the manifest.

## Fixed protocol

Each legal board is encoded on nine qubits with `RX(2*pi*g_i/3)`. Final runs
use `L=3`, `p=2`, Adam with learning rate `0.01`, batch size `15`, `30`
minibatch updates per epoch, `100` epochs, and a fixed 348-board test set
held out as complete `D4` orbits.

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
