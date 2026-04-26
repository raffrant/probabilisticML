# Weighted Graph States, Sector Lengths, and ML-Guided Cluster Extraction

This project studies **weighted graph states** under local measurements and asks two related questions:

1. Can we extract high-quality **cluster-like output states** by measuring some qubits?
2. Can **machine learning** predict which measurement protocols will give the best output states, using both **cluster fidelity** and **sector lengths** as targets?

The code combines quantum-state simulation, post-measurement analysis, sector-length computation, and machine learning models for protocol ranking. Sector lengths are used as structural descriptors of multipartite correlations, following the viewpoint developed in *Characterizing quantum states via sector lengths*. [web:151]

---

## Project idea

We start from a noisy or imperfect **weighted graph state**, typically on a `2x3` lattice. Each edge has a phase \( \phi \in [0,\pi] \), where \( \phi = \pi \) corresponds to the ideal CZ edge of a cluster state.

For each sampled weighted graph:

- build the graph state,
- choose a subset of qubits to measure,
- choose single-qubit measurement bases,
- compute the post-measurement state of the surviving qubits,
- evaluate how good that output state is.

We evaluate outputs in two complementary ways:

- **Cluster fidelity:** how close the surviving state is to an ideal cluster-chain target.
- **Sector lengths:** how the Pauli correlations are distributed across correlation sectors \(A_1, A_2, \dots, A_n\). These quantities summarize multipartite correlation structure in a basis-independent way. [web:151][file:150]

---

## Main files

### `graph_state_ml2.py`
Original weighted graph-state pipeline with:
- weighted CZ graph-state generation,
- projective measurement on selected qubits,
- fidelity-based optimization toward extracted cluster states,
- dataset generation,
- classical ML models for fidelity and subset prediction. [file:117]

### `sectorlengths.py`
Exploratory code for:
- Pauli decomposition,
- calculation of sector lengths from density matrices,
- comparison of correlation sectors across different quantum states. [file:150]

### `graph_sector_ml.py`
New combined pipeline for:
- generating weighted graph states,
- sampling measurement protocols,
- computing post-measurement sector lengths,
- computing cluster fidelity,
- training ML models to predict sector lengths and output quality,
- ranking measurement protocols to find the best output states quickly.

### `graph_sector_notebook.ipynb`
Notebook for:
- generating datasets,
- checking sector-length distributions,
- visualizing output-state quality,
- evaluating ML models,
- inspecting ML-ranked best protocols.

---

## Scientific motivation

Weighted graph states interpolate between weakly entangled and ideal graph/cluster states by varying edge phases. Measuring some qubits can concentrate or reshape entanglement in the remaining subsystem, which makes them a natural testbed for **entanglement extraction** and **measurement-based state engineering**. [web:186][file:117]

Sector lengths provide a useful second lens beyond fidelity. Instead of asking only “is this state close to one exact target?”, they also ask “does this state have the right multipartite correlation structure?”. That makes them especially useful when several output states may be locally different but structurally similar in their correlation content. [web:151][file:150]

---

## What the pipeline does

### 1. Generate weighted graph states
For a lattice such as `2x3`, each edge gets a random phase in `[0, π]`, and the corresponding weighted-CZ graph state is constructed. [file:117]

### 2. Apply measurement protocols
A protocol is defined by:
- which qubits are measured,
- the measurement basis angles on those qubits.

The code computes the post-measurement state and the success probability of that outcome. [file:117]

### 3. Compute output-state descriptors
For the surviving subsystem, the code stores:
- cluster fidelity,
- success probability,
- sector lengths \(A_1, A_2, A_3\) for 3-qubit outputs,
- distance between the measured sector-length vector and the ideal cluster sector-length vector. [file:150][web:151]

### 4. Train machine learning models
The ML pipeline learns:
- **multi-output regression** for sector lengths,
- regression for cluster fidelity,
- classification of whether a protocol gives a “good” output state. Multi-output regression is a standard way to model several target variables such as \(A_1, A_2, A_3\) jointly. [web:44]

### 5. Rank protocols
The trained models assign each protocol an ML score based on predicted fidelity and predicted probability of being a high-quality output. This score is then used to rank protocols and identify promising candidates quickly. [web:44]

---

## Typical workflow

### A. Build the dataset
Generate many rows, where each row corresponds to one:

- graph sample,
- measured subset,
- measurement basis choice.

Each row stores both input features and output-state quality metrics.

### B. Train models
Train models to predict:
- sector lengths,
- fidelity to cluster target,
- whether the output is in the top fraction of states.

### C. Use ML to find the best outputs
Instead of exhaustively optimizing everything, use ML to:
- score all candidate protocols,
- keep only the top-ranked ones,
- optionally refine those with a slower quantum optimization step.

This makes the search for good extracted states much faster in practice. [file:117]

---

## Example outputs

The code can produce:

- `graph_sector_dataset.csv` — protocol-level dataset.
- `sector_regression_metrics.csv` — ML accuracy for sector-length prediction.
- `fidelity_regression_metrics.csv` — ML accuracy for fidelity prediction.
- `good_output_classification_metrics.csv` — classification metrics for identifying good outputs.
- `graph_sector_scored_protocols.csv` — all protocols with ML scores.
- `graph_sector_best_protocols_per_graph.csv` — best ML-selected protocol for each graph sample.

The notebook also visualizes:
- fidelity distributions,
- sector-length distributions,
- actual vs predicted sector lengths,
- top ML-ranked protocols,
- gap between true best and ML-selected protocols.

---

## Installation

Use Python 3.10+ or 3.11+.

Install dependencies:

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn joblib tqdm jupyter ipykernel
```

---

## How to run

### Run the Python pipeline
After saving `graph_sector_ml.py`:

```bash
python graph_sector_ml.py
```

This will:
- generate a dataset,
- train ML models,
- rank protocols,
- save metrics and figures to `output/`.

### Run the notebook
Start Jupyter:

```bash
jupyter notebook
```

Then open:

```text
graph_sector_notebook.ipynb
```

---

## Suggested first settings

For a first practical run:

- `rows=2`, `cols=3`
- `measured_k=3`
- `n_graph_samples=100`
- `protocols_per_subset=3`

This gives a moderate dataset while keeping sector-length calculations cheap, because the surviving post-measurement state has only 3 qubits, so its full Pauli expansion uses only \(4^3 = 64\) Pauli strings. [file:150]

For a quicker debug run:

- `n_graph_samples=20`
- `protocols_per_subset=2`

---

## Interpretation of sector lengths

For an \(n\)-qubit state, sector lengths \(A_k\) summarize the total squared Pauli-correlation weight carried by \(k\)-body nontrivial correlations. In this project, they are computed from the Pauli expansion of the reduced density matrix after measurement, following the grouping logic already present in `sectorlengths.py`. [file:150][web:151]

This lets you compare output states not only by overlap with an ideal target, but also by their correlation structure.

---

## Research directions

This repository is a good starting point for several extensions:

- **Better protocol search:** use ML to shortlist protocols, then run local optimization only on the best few.
- **Larger lattices:** move from `2x3` to `3x3` or `4x3`.
- **Alternative targets:** use GHZ, Bell-chain, or custom graph-state targets.
- **Better labels:** define “good output” using both fidelity and sector-distance thresholds.
- **Different models:** compare Random Forests with Gradient Boosting, XGBoost, or neural networks.
- **Hybrid objectives:** optimize for both high cluster fidelity and sector-length closeness simultaneously. [file:117][file:150]

---

## Notes

- `sectorlengths.py` is partly exploratory and mixes reusable functions with plotting scripts. It is best treated as a reference for the sector-length logic rather than a finished module. [file:150]
- The original graph-state extraction code is fidelity-driven; the newer sector-length pipeline adds a more structural entanglement descriptor. [file:117][file:150]
- Multi-output regression is used because sector lengths are naturally vector-valued targets rather than a single scalar. [web:44]

---

## References

- Maciążek, Wiesniak, and collaborators, **“Characterizing quantum states via sector lengths”**, arXiv:1905.06928. [web:151]
- Scikit-learn documentation for **MultiOutputRegressor**, used for multi-target prediction. [web:44]
- Related weighted graph-state entanglement concentration ideas also appear in work on extracting perfect GHZ states from imperfect weighted graph states. [web:186]
