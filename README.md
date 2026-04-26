# Machine Learning for Graph-State Extraction from Weighted Graph States

This repository explores **general strategies for extracting useful graph states from imperfect weighted graph states using machine learning**.

The core idea is simple:

- start from a weighted graph state,
- apply local measurements on some qubits,
- look at the remaining post-measurement state,
- decide whether that output is a good graph-state resource,
- and use **machine learning** to predict which measurement protocols are best.

Instead of relying only on brute-force search, the goal is to learn patterns that tell us **which weighted graphs and which measurements are likely to produce the best output states**.

---

## General research question

Given an imperfect or noisy **weighted graph state**, can we use local measurements to extract a smaller state that is:

- close to an ideal **cluster state** or another target graph state,
- strongly multipartite entangled,
- structurally useful for measurement-based quantum information tasks,
- and predictable using machine learning?

This repository is organized around several possible ways to define what a “good output state” means.

---

## Main idea: ML-guided extraction

A measurement protocol is defined by:

- the initial weighted graph state,
- the subset of qubits to measure,
- the local basis angles used for those measurements.

For each protocol, we compute the remaining post-measurement state and assign it a quality score. Machine learning is then used to approximate that score and rank protocols quickly.

In practice, the ML task is:

- **input:** graph features + measurement features,
- **output:** a score or descriptor of the post-measurement state.

The best protocols are then the ones with the highest predicted output quality.

---

# Approaches to output quality

## 1. Sector lengths approach

This is the **first and most structural idea** in the project.

Sector lengths \(A_k\) quantify how much \(k\)-body correlation is present in a quantum state in a basis-independent way. They come from the Pauli expansion of the density matrix and provide a compact description of multipartite correlation structure. This makes them a natural tool for studying post-measurement graph-state extraction. [web:151]

### Why use sector lengths?

A fidelity score only tells us how close the output is to one specific target. Sector lengths answer a broader question:

> Does the output state have the right kind of multipartite correlation structure?

This is useful because two states can differ by local basis changes or small deformations while still sharing a similar correlation profile. Sector lengths have also been studied specifically for graph states and sector-length distributions of graph states, which makes them especially relevant here. [web:190][web:151]

### ML idea with sector lengths

Use machine learning to predict the sector-length vector of the post-measurement state, for example:

- \(A_1\)
- \(A_2\)
- \(A_3\)

for a 3-qubit extracted state.

This becomes a **multi-output regression** problem:

- **input:** weighted graph phases, measured subset, measurement basis parameters,
- **target:** sector-length vector of the output state.

Then define “best output” in one of these ways:

- smallest distance to the sector-length vector of an ideal cluster state,
- largest desired higher-body sector,
- smallest undesired lower-body sector,
- or a custom combination of sector features.

### Why this is promising

This approach does not force the output to match one exact state. Instead, it learns to detect **good correlation structure**, which may be more robust and physically informative than fidelity alone. It is also closely connected to entanglement detection and multipartite correlation constraints. [web:151]

---

## 2. Fidelity to a target graph state

This is the most direct idea.

Choose a target output state, such as:

- a 3-qubit cluster chain,
- a GHZ state,
- a Bell pair,
- or another graph state.

For each measurement protocol, compute the fidelity of the post-measurement state to that target. Then train an ML model to predict that fidelity from the input features.

### ML idea with fidelity

- **input:** weighted graph phases + measurement protocol,
- **target:** fidelity of the output state to the chosen graph-state target.

This gives a scalar regression problem, which is simple and practical.

### Why this matters

This is the natural approach when the goal is explicit **state extraction**. It is closely related to entanglement concentration ideas where imperfect weighted graph states are locally processed to produce perfect or near-perfect target states. Weighted graph states are already known to support concentration protocols that extract ideal GHZ states from imperfect resources. [web:186]
