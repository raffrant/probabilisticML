"""
Weighted graph-state sector-length ML pipeline.

Goal
----
Generate many weighted graph states, measure selected qubits, compute the
post-measurement sector lengths and cluster fidelity of the remaining register,
and use machine learning to predict which protocols produce the best outputs.

Main outputs
------------
1. A protocol-level dataset: one row per (graph sample, measurement subset, basis).
2. Multi-output regression for sector lengths A_k.
3. Regression for cluster fidelity.
4. Classification for whether a protocol yields a "good" output state.
5. ML-based ranking of protocols to identify the best output states quickly.
"""

from __future__ import annotations

import functools as ft
import warnings
from itertools import combinations, product
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

_BLUE = "#3B6DB3"
_ORANGE = "#E07B39"
_GREEN = "#3A9E6A"
_RED = "#C94040"
_GRAY = "#7A7A7A"
MIN_PROB = 1e-12


def _apply_style() -> None:
    sns.set_theme(style="whitegrid", font_scale=1.05)
    mpl.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "legend.frameon": False,
        }
    )


_apply_style()

X_PAULI = np.array([[0, 1], [1, 0]], dtype=complex)
Y_PAULI = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z_PAULI = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)


# ---------------------------------------------------------------------
# Quantum state utilities
# ---------------------------------------------------------------------

def kron_all(ops: Sequence[np.ndarray]) -> np.ndarray:
    out = np.array([1.0 + 0.0j])
    for op in ops:
        out = np.kron(out, op)
    return out


def get_initial_state(n: int) -> np.ndarray:
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    state = plus.copy()
    for _ in range(n - 1):
        state = np.kron(state, plus)
    return state


def apply_weighted_cz(state: np.ndarray, n: int, q1: int, q2: int, phi: float) -> np.ndarray:
    out = state.copy()
    dim = 2**n
    mask1 = 1 << (n - 1 - q1)
    mask2 = 1 << (n - 1 - q2)
    indices = np.arange(dim)
    selector = ((indices & mask1) > 0) & ((indices & mask2) > 0)
    out[selector] *= np.exp(1j * phi)
    return out


def grid_edges(rows: int, cols: int) -> List[Tuple[int, int]]:
    horizontal = [(i, i + 1) for i in range(rows * cols - 1) if (i + 1) % cols != 0]
    vertical = [(i, i + cols) for i in range(rows * cols - cols)]
    return horizontal + vertical


def build_weighted_graph_state(rows: int, cols: int, phases: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    n = rows * cols
    edges = grid_edges(rows, cols)
    psi = get_initial_state(n)
    for edge_idx, (q1, q2) in enumerate(edges):
        psi = apply_weighted_cz(psi, n, q1, q2, phases[edge_idx])
    return psi, edges


def cluster_chain(n: int) -> np.ndarray:
    psi = get_initial_state(n)
    for i in range(n - 1):
        psi = apply_weighted_cz(psi, n, i, i + 1, np.pi)
    return psi


def fidelity(psi: np.ndarray, phi: np.ndarray) -> float:
    return float(np.abs(np.vdot(psi, phi)) ** 2)


def measure_projection(
    state: np.ndarray,
    n: int,
    subset: Sequence[int],
    bases: np.ndarray,
) -> tuple[np.ndarray, float]:
    rem_indices = [i for i in range(n) if i not in subset]
    m = len(subset)

    bras = []
    for i in range(m):
        theta, phi = bases[i]
        bra = np.array(
            [
                np.cos(theta / 2.0),
                np.exp(-1j * phi) * np.sin(theta / 2.0),
            ],
            dtype=complex,
        )
        bras.append(bra)

    bra_total = bras[0]
    for b in bras[1:]:
        bra_total = np.kron(bra_total, b)

    perm = list(subset) + rem_indices
    state_tensor = state.reshape([2] * n)
    state_moved = np.transpose(state_tensor, perm).reshape(2**m, 2 ** (n - m))

    projected = bra_total @ state_moved
    norm = np.linalg.norm(projected)
    if norm < MIN_PROB:
        return projected, 0.0
    return projected / norm, norm**2


# ---------------------------------------------------------------------
# Sector lengths
# ---------------------------------------------------------------------

def density_matrix_from_state(psi: np.ndarray) -> np.ndarray:
    return np.outer(psi, psi.conjugate())


def sector_lengths_from_rho(rho: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    paulis = [I2, X_PAULI, Y_PAULI, Z_PAULI]
    labels = list(product(["i", "x", "y", "z"], repeat=n))
    ops = list(product(paulis, repeat=n))

    coeffs = np.zeros(4**n)
    A = np.zeros(n + 1)

    for idx, (label, op_tuple) in enumerate(zip(labels, ops)):
        P = ft.reduce(np.kron, op_tuple)
        c = np.real(np.trace(P @ rho)) / (2**n)
        coeffs[idx] = c

        weight = n - label.count("i")
        if weight > 0:
            A[weight] += (4**n) * (c**2)

    return A[1:], coeffs


def sector_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


# ---------------------------------------------------------------------
# Features and sampling
# ---------------------------------------------------------------------

def subset_to_label(subset: Sequence[int]) -> str:
    return "-".join(map(str, subset))


def edge_feature_dict(
    phases: np.ndarray,
    edges: Sequence[Tuple[int, int]],
    n_qubits: int,
) -> Dict[str, float]:
    d: Dict[str, float] = {}
    deviations = np.abs(phases - np.pi)

    for i, phase in enumerate(phases):
        d[f"phase_{i}"] = float(phase)
        d[f"phase_dev_{i}"] = float(deviations[i])
        d[f"phase_sin_{i}"] = float(np.sin(phase))
        d[f"phase_cos_{i}"] = float(np.cos(phase))

    d["phase_mean"] = float(np.mean(phases))
    d["phase_std"] = float(np.std(phases))
    d["phase_min"] = float(np.min(phases))
    d["phase_max"] = float(np.max(phases))
    d["phase_dev_mean"] = float(np.mean(deviations))
    d["phase_dev_std"] = float(np.std(deviations))
    d["phase_dev_max"] = float(np.max(deviations))

    node_load = np.zeros(n_qubits)
    for edge_idx, (u, v) in enumerate(edges):
        node_load[u] += deviations[edge_idx]
        node_load[v] += deviations[edge_idx]

    for q in range(n_qubits):
        d[f"node_load_{q}"] = float(node_load[q])

    return d


def measurement_feature_dict(n: int, subset: Sequence[int], bases: np.ndarray) -> Dict[str, float]:
    d: Dict[str, float] = {}
    subset = list(subset)
    measured_flags = np.zeros(n, dtype=int)
    for q in subset:
        measured_flags[q] = 1

    for q in range(n):
        d[f"is_measured_{q}"] = int(measured_flags[q])

    for i, q in enumerate(subset):
        d[f"measured_qubit_{i}"] = int(q)
        d[f"meas_theta_{i}"] = float(bases[i, 0])
        d[f"meas_phi_{i}"] = float(bases[i, 1])

    d["subset_label"] = subset_to_label(subset)
    return d


def sample_random_bases(m: int, rng: np.random.Generator) -> np.ndarray:
    bases = np.zeros((m, 2))
    bases[:, 0] = rng.uniform(0.0, np.pi, m)
    bases[:, 1] = rng.uniform(0.0, 2 * np.pi, m)
    return bases


# ---------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------

def protocol_row(
    sample_id: int,
    phases: np.ndarray,
    edges: list[tuple[int, int]],
    psi: np.ndarray,
    subset: Sequence[int],
    bases: np.ndarray,
    target_cluster: np.ndarray,
    target_sector: np.ndarray,
    n: int,
) -> Dict:
    reduced, prob = measure_projection(psi, n, subset, bases)
    if prob < MIN_PROB:
        return {}

    rho = density_matrix_from_state(reduced)
    surv_n = int(np.log2(len(reduced)))
    sector_vec, _ = sector_lengths_from_rho(rho, surv_n)
    fid_cluster = fidelity(reduced, target_cluster)
    sec_dist = sector_distance(sector_vec, target_sector)

    row = {
        "sample_id": sample_id,
        "subset_label": subset_to_label(subset),
        "success_probability": float(prob),
        "cluster_fidelity": float(fid_cluster),
        "sector_distance_to_cluster": float(sec_dist),
    }

    for k, val in enumerate(sector_vec, start=1):
        row[f"A_{k}"] = float(val)

    row.update(edge_feature_dict(phases, edges, n))
    row.update(measurement_feature_dict(n, subset, bases))
    return row


def generate_protocol_dataset(
    rows: int = 2,
    cols: int = 3,
    n_graph_samples: int = 200,
    measured_k: int = 3,
    protocols_per_subset: int = 4,
    seed: int = 42,
    n_jobs: int = -1,
) -> pd.DataFrame:
    n = rows * cols
    edges = grid_edges(rows, cols)
    all_subsets = list(combinations(range(n), measured_k))
    target_cluster = cluster_chain(n - measured_k)
    target_sector, _ = sector_lengths_from_rho(
        density_matrix_from_state(target_cluster),
        n - measured_k,
    )

    def one_graph(sample_id: int) -> list[Dict]:
        rng = np.random.default_rng(seed + sample_id)
        phases = rng.uniform(0.0, np.pi, len(edges))
        psi, _ = build_weighted_graph_state(rows, cols, phases)

        rows_out = []
        for subset in all_subsets:
            for _ in range(protocols_per_subset):
                bases = sample_random_bases(measured_k, rng)
                row = protocol_row(
                    sample_id=sample_id,
                    phases=phases,
                    edges=edges,
                    psi=psi,
                    subset=subset,
                    bases=bases,
                    target_cluster=target_cluster,
                    target_sector=target_sector,
                    n=n,
                )
                if row:
                    rows_out.append(row)
        return rows_out

    nested = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(one_graph)(sample_id)
        for sample_id in tqdm(range(n_graph_samples), desc="Generating graph samples", unit="graph")
    )

    flat = [row for rows_ in nested for row in rows_]
    df = pd.DataFrame(flat)

    if len(df) == 0:
        raise RuntimeError("Dataset is empty. Increase protocols_per_subset or check measurement settings.")

    q = df["cluster_fidelity"].quantile(0.90)
    df["good_output"] = (df["cluster_fidelity"] >= q).astype(int)
    return df


# ---------------------------------------------------------------------
# ML
# ---------------------------------------------------------------------

def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    exclude = {
        "sample_id",
        "subset_label",
        "cluster_fidelity",
        "sector_distance_to_cluster",
        "good_output",
        "A_1",
        "A_2",
        "A_3",
    }
    feature_cols = [c for c in df.columns if c not in exclude]
    return df[feature_cols].copy(), feature_cols


def train_sector_and_output_models(df: pd.DataFrame) -> Dict:
    X, feature_cols = prepare_features(df)

    y_sector_cols = [c for c in df.columns if c.startswith("A_")]
    y_sector = df[y_sector_cols].copy()
    y_fid = df["cluster_fidelity"].copy()
    y_good = df["good_output"].copy()

    X_tr, X_te, y_sector_tr, y_sector_te, y_fid_tr, y_fid_te, y_good_tr, y_good_te = train_test_split(
        X, y_sector, y_fid, y_good, test_size=0.25, random_state=42
    )

    sector_model = RandomForestRegressor(
        n_estimators=400,
        max_depth=18,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    sector_model.fit(X_tr, y_sector_tr)

    fid_model = RandomForestRegressor(
        n_estimators=400,
        max_depth=18,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    fid_model.fit(X_tr, y_fid_tr)

    good_model = RandomForestClassifier(
        n_estimators=400,
        max_depth=18,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    good_model.fit(X_tr, y_good_tr)

    pred_sector = sector_model.predict(X_te)
    pred_fid = fid_model.predict(X_te)
    pred_good = good_model.predict(X_te)
    pred_good_proba = good_model.predict_proba(X_te)[:, 1]

    sector_metrics = {}
    for idx, col in enumerate(y_sector_cols):
        sector_metrics[col] = {
            "mae": float(mean_absolute_error(y_sector_te[col], pred_sector[:, idx])),
            "r2": float(r2_score(y_sector_te[col], pred_sector[:, idx])),
        }

    fid_metrics = {
        "mae": float(mean_absolute_error(y_fid_te, pred_fid)),
        "r2": float(r2_score(y_fid_te, pred_fid)),
    }

    good_metrics = {
        "accuracy": float(accuracy_score(y_good_te, pred_good)),
        "roc_auc": float(roc_auc_score(y_good_te, pred_good_proba)),
    }

    return {
        "feature_cols": feature_cols,
        "y_sector_cols": y_sector_cols,
        "sector_model": sector_model,
        "fid_model": fid_model,
        "good_model": good_model,
        "X_tr": X_tr,
        "X_te": X_te,
        "y_sector_te": y_sector_te,
        "y_fid_te": y_fid_te,
        "y_good_te": y_good_te,
        "pred_sector": pred_sector,
        "pred_fid": pred_fid,
        "pred_good": pred_good,
        "pred_good_proba": pred_good_proba,
        "sector_metrics": sector_metrics,
        "fid_metrics": fid_metrics,
        "good_metrics": good_metrics,
    }


# ---------------------------------------------------------------------
# ML ranking of best protocols
# ---------------------------------------------------------------------

def rank_protocols_with_ml(df: pd.DataFrame, models: Dict) -> pd.DataFrame:
    X, _ = prepare_features(df)
    scored = df.copy()

    scored["pred_cluster_fidelity"] = models["fid_model"].predict(X)
    scored["pred_good_output_proba"] = models["good_model"].predict_proba(X)[:, 1]
    pred_sector = models["sector_model"].predict(X)

    for idx, col in enumerate(models["y_sector_cols"]):
        scored[f"pred_{col}"] = pred_sector[:, idx]

    scored["ml_score"] = (
        0.65 * scored["pred_cluster_fidelity"]
        + 0.35 * scored["pred_good_output_proba"]
    )

    return scored.sort_values("ml_score", ascending=False).reset_index(drop=True)


def best_protocols_per_graph(scored_df: pd.DataFrame) -> pd.DataFrame:
    idx = scored_df.groupby("sample_id")["ml_score"].idxmax()
    return scored_df.loc[idx].sort_values("sample_id").reset_index(drop=True)


# ---------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------

def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_fidelity_hist(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df["cluster_fidelity"], bins=30, kde=True, color=_BLUE, alpha=0.6, ax=ax)
    ax.set_title("Cluster-fidelity distribution")
    ax.set_xlabel("Cluster fidelity")
    _save(fig, out_path)


def fig_sector_scatter(y_true: pd.DataFrame, y_pred: np.ndarray, sector_cols: list[str], out_path: Path) -> None:
    n = len(sector_cols)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for i, col in enumerate(sector_cols):
        ax = axes[i]
        ax.scatter(y_true[col], y_pred[:, i], s=16, alpha=0.4, color=_BLUE)
        lo = min(y_true[col].min(), y_pred[:, i].min())
        hi = max(y_true[col].max(), y_pred[:, i].max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_title(col)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")

    fig.suptitle("Sector-length prediction")
    _save(fig, out_path)


def fig_top_protocols(scored_df: pd.DataFrame, out_path: Path, top_n: int = 15) -> None:
    top = scored_df.head(top_n).copy()
    top = top.iloc[::-1]

    labels = [f"s{sid}:{sub}" for sid, sub in zip(top["sample_id"], top["subset_label"])]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(labels, top["ml_score"], color=_GREEN)
    ax.set_title("Top protocols by ML score")
    ax.set_xlabel("ML score")
    _save(fig, out_path)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main(
    rows: int = 2,
    cols: int = 3,
    n_graph_samples: int = 200,
    measured_k: int = 3,
    protocols_per_subset: int = 4,
    seed: int = 42,
    n_jobs: int = -1,
) -> None:
    out = Path("output")
    out.mkdir(exist_ok=True)

    print("\n--- Step 1/4: Generate sector-length dataset ---")
    df = generate_protocol_dataset(
        rows=rows,
        cols=cols,
        n_graph_samples=n_graph_samples,
        measured_k=measured_k,
        protocols_per_subset=protocols_per_subset,
        seed=seed,
        n_jobs=n_jobs,
    )
    df.to_csv(out / "graph_sector_dataset.csv", index=False)
    print(f"Dataset shape: {df.shape}")

    print("\n--- Step 2/4: Train ML models ---")
    M = train_sector_and_output_models(df)

    sector_metrics_df = pd.DataFrame(M["sector_metrics"]).T.reset_index().rename(columns={"index": "sector"})
    fid_metrics_df = pd.DataFrame([M["fid_metrics"]])
    good_metrics_df = pd.DataFrame([M["good_metrics"]])

    sector_metrics_df.to_csv(out / "sector_regression_metrics.csv", index=False)
    fid_metrics_df.to_csv(out / "fidelity_regression_metrics.csv", index=False)
    good_metrics_df.to_csv(out / "good_output_classification_metrics.csv", index=False)

    print("\nSector metrics")
    print(sector_metrics_df.to_string(index=False))
    print("\nFidelity metrics")
    print(fid_metrics_df.to_string(index=False))
    print("\nGood-output metrics")
    print(good_metrics_df.to_string(index=False))

    print("\n--- Step 3/4: Rank protocols with ML ---")
    scored = rank_protocols_with_ml(df, M)
    scored.to_csv(out / "graph_sector_scored_protocols.csv", index=False)

    best_df = best_protocols_per_graph(scored)
    best_df.to_csv(out / "graph_sector_best_protocols_per_graph.csv", index=False)
    print(best_df[[
        "sample_id",
        "subset_label",
        "cluster_fidelity",
        "pred_cluster_fidelity",
        "pred_good_output_proba",
        "ml_score",
    ]].head(12).to_string(index=False))

    print("\n--- Step 4/4: Save figures ---")
    fig_fidelity_hist(df, out / "fig_cluster_fidelity_hist.png")
    fig_sector_scatter(M["y_sector_te"], M["pred_sector"], M["y_sector_cols"], out / "fig_sector_prediction.png")
    fig_top_protocols(scored, out / "fig_top_protocols.png")

    print(f"\nAll outputs written to: {out.resolve()}")


if __name__ == "__main__":
    main()