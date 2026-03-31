# Classical Simulation of Quantum Dynamics via Pauli Propagation

[`Classical Simulation of Noiseless Quantum Dynamics without Randomness`](https://arxiv.org/abs/2601.15770).
Jue Xu, Chu Zhao, Xiangran Zhang, Shuchen Zhu, and Qi Zhao, 2025.
[![arXiv](https://img.shields.io/badge/paper%20%28v1%29-arXiv%3A2601.15770-B31B1B)](https://arxiv.org/abs/2601.15770)


## Overview

This repository implements the **Low-weight Pauli Dynamics (LPD)** algorithm for classically simulating quantum Hamiltonian evolution. LPD approximates local expectation values $\mathrm{Tr}(e^{iHt} O\, e^{-iHt} \rho)$ by propagating observables in the Heisenberg picture and truncating high-weight Pauli strings.

**Key insight:** Entanglement in the input state *suppresses* truncation error, making classical simulation easier — not harder — for entangled states.


## Problem Setting

Given a $k_h$-local Hamiltonian $H$, a $k_o$-local observable $O$, and an input state $\rho$:
1. **Trotterize** the time evolution $e^{-iHt}$ into a sequence of Pauli rotations.
2. **Propagate** the observable $O$ backward through the circuit in the Heisenberg picture. Each Pauli rotation either leaves a Pauli string unchanged (commuting) or branches it into two terms with a $\sin(\delta t)$ damping factor (anticommuting).
3. **Truncate** all Pauli strings with weight above a threshold $w^*$.
4. **Evaluate** $\mathrm{Tr}(\rho \cdot O_{\mathrm{truncated}})$.


## Key Results

- **Entanglement helps:** The Pauli 2-norm (which bounds truncation error) is suppressed by the subsystem entanglement entropy of $\rho$.
- **Polynomial runtime:** For short-time dynamics at constant precision, LPD runs in $O(n^{w^*+c})$ time, polynomial in system size.
- **Hybrid MPS + LPD:** For product initial states, forward-evolve $\rho$ with MPS (Schrödinger picture) to build entanglement, then backward-evolve $O$ with LPD (Heisenberg picture) — extending the reachable simulation time beyond either method alone.


## Numerical Results

<!-- - [Pauli propagation (LPD)](./pauli_propagation.ipynb) — Core LPD algorithm demonstration -->
- [LPD](./PF2.ipynb) — Low-weight Pauli with the second-order product formula
<!-- - [MPS simulation](./MPS.ipynb) — Matrix product state evolution -->
- [Hybrid LPD + MPS](./hybrid.ipynb) — Combined Heisenberg + Schrödinger picture simulation
<!-- - [Magic and entanglement](./Magic_Entangle.ipynb) — Operator magic and entanglement entropy analysis
- [OTOC](./otoc.ipynb) — Out-of-time-order correlator calculations
- [Energy expectation](./Energy.ipynb) — Energy expectation value computation
- [2D Ising model](./Ising2D.ipynb) — Simulations on 2D lattice geometries
- [Trotter error analysis](./trotter_error.ipynb) — Error bounds and convergence
- [Dynamics examples](./DynamicsExample.ipynb) — Illustrative dynamics simulations -->


## Code Structure

```
spd/                          # Sparse Pauli Dynamics package
├── SparsePauliDynamics.py    # Full SPD simulation (threshold-based filtering)
├── LightPauliDynamics.py     # LPD with weight-based truncation
├── PauliRepresentation.py    # Bit-packed Pauli string data structure
├── OperatorSequence.py       # Pauli rotation gate sequences
├── utils.py                  # Numba-accelerated bit operations
└── extras/
    └── HeavyHexUtils.py      # IBM Heavy Hex topology utilities

pauli.py                      # Pauli rotation, weight decomposition, expectation values
Hamiltonians.py               # Local Hamiltonian construction (XYZ interactions, QMFI)
input_states.py               # Clifford 2/3-design state generation
quantity.py                   # Entanglement entropy, operator magic, fidelity
trotter_bounds.py             # Analytical and empirical Trotter error bounds
mps.py                        # MPS utilities

julia/                        # Julia implementation for performance comparison
```


## Main References

- Aharonov, Gao, Landau, Liu, and Vazirani. [A Polynomial-Time Classical Algorithm for Noisy Random Circuit Sampling](https://dl.acm.org/doi/10.1145/3564246.3585234). In Proceedings of STOC 2023, 945–957 (2023).
- Begušić and Chan. [Fast and converged classical simulations of evidence for the utility of quantum computing before fault tolerance](https://www.science.org/doi/10.1126/sciadv.adk4321). Science Advances 10, eadk4321 (2024).
- Shao, Wei, Cheng, and Liu. [Simulating noisy variational quantum algorithms: A polynomial approach](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.133.120603). Physical Review Letters 133, 120603 (2024).
- Begušić and Chan. [Real-Time Operator Evolution in Two and Three Dimensions via Sparse Pauli Dynamics](https://link.aps.org/doi/10.1103/PRXQuantum.6.020302). PRX Quantum 6, 020302 (2025).
- Rudolph, Jones, Teng, Angrisani, and Holmes. [Pauli Propagation: A Computational Framework for Simulating Quantum Systems](https://arxiv.org/abs/2505.21606). arXiv:2505.21606 (2025).
- Schuster, Yin, Gao, and Yao. [A Polynomial-Time Classical Algorithm for Noisy Quantum Circuits](https://link.aps.org/doi/10.1103/xct1-7kf2). Physical Review X 15, 041018 (2025).


## Usage

```bash
# Create python environment
conda create --name myenv python=3.10
# Install requirements
pip install quantum_simulation_recipe
```
