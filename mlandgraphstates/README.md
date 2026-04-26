Graph State Measurement Concentration

This repository studies entanglement concentration from random weighted graph states on grid graphs. It generates synthetic datasets, simulates single-qubit measurements on up to three qubits, optimizes local single-qubit rotations on the remaining qubits, and trains a simple machine-learning baseline to predict the best achievable fidelity to a target perfect graph state.
What it does

    Builds weighted graph states starting from ∣+⟩⊗n∣+⟩⊗n.

    Samples random controlled-phase weights on grid edges.

    Measures up to 3 qubits in arbitrary single-qubit bases.

    Optimizes local single-qubit rotations on the post-measurement state.

    Compares the result to a perfect graph-state target.

    Exports a dataset for downstream pattern discovery and machine learning.

Repository layout

text
graph-state-ml/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── pyproject.toml
├── src/
│   └── graph_state_ml/
│       ├── __init__.py
│       ├── quantum.py
│       ├── dataset.py
│       ├── ml.py
│       └── cli.py
├── scripts/
│   └── run_experiment.py
└── tests/
    └── test_quantum.py

Installation

bash
git clone <your-repo-url>
cd graph-state-ml
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

Usage

Run a small end-to-end experiment:

bash
python scripts/run_experiment.py --rows 2 --cols 3 --samples 120 --max-measured 3 --seed 7

This generates:

    weighted_graph_dataset.csv

    weighted_graph_ml_metrics.csv

Notes

The current target after measurement is the perfect graph state on the induced subgraph of the surviving qubits. This is a clean baseline, but not the most general graph-state update rule under measurements, so you may later want to replace it with a more physically exact target construction.
Citation

If you use this repository in academic work, cite your related paper or preprint here.
