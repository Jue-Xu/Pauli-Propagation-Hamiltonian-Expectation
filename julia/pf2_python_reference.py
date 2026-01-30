"""
Standalone Python script to reproduce PF2.ipynb numerical results.
This generates the reference expectation values for comparison with Julia implementations.

Parameters:
- n = 10 qubits
- MFI Hamiltonian: H = 0.8 Σ Xᵢ + 0.9 Σ Yᵢ + 1.0 Σ XᵢXᵢ₊₁
- t = 5, r = 50, dt = 0.1
- Initial state: |1010101010⟩
- Observable: Z on qubit 0 (rightmost)
- Weight threshold: w* = 5
"""

import sys
import os
import numpy as np
from functools import partial
from numpy.linalg import matrix_power
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qiskit.quantum_info import SparsePauliOp, Statevector
from spd.LightPauliDynamics import LowWeightPauliPropagation
from pauli import exp_val_0101_state_pauli_rep, expect_value
from quantum_simulation_recipe.spin import Nearest_Neighbour_1d
from quantum_simulation_recipe.trotter import pf, expH

def run_pf2_simulation():
    """Run the full PF2 simulation and return results."""

    print("=" * 70)
    print("PF2 Python Reference Simulation")
    print("=" * 70)

    # Parameters
    n = 10
    t = 5.0
    r = 50
    dt = t / r
    Jx, hx, hy = 1.0, 0.8, 0.9
    w_threshold = 5

    print(f"Parameters: n={n}, t={t}, r={r}, dt={dt}")
    print(f"Hamiltonian: MFI with Jx={Jx}, hx={hx}, hy={hy}")
    print(f"Weight threshold: w*={w_threshold}")
    print(f"Initial state: |1010101010⟩")
    print(f"Observable: Z on qubit 0")
    print("=" * 70)

    # Build Hamiltonian using quantum_simulation_recipe
    print("\n1. Building Hamiltonian...")
    qimf = Nearest_Neighbour_1d(n, hx=hx, hy=hy, Jx=Jx, pbc=False)

    # Extract terms for Trotter decomposition
    xx_even = SparsePauliOp.from_sparse_list([*qimf.xx_tuples[::2]], num_qubits=n).simplify()
    xx_odd = SparsePauliOp.from_sparse_list([*qimf.xx_tuples[1::2]], num_qubits=n).simplify()
    x_terms = SparsePauliOp.from_sparse_list([*qimf.x_tuples], num_qubits=n).simplify()

    H_list = [x_terms, qimf.y_terms, xx_even, xx_odd]
    H_pf2_list = [x_terms, qimf.y_terms, xx_even, xx_odd, xx_odd, xx_even, qimf.y_terms, x_terms]
    H_ordered = sum(H_list)
    H_pf2_ordered = sum(H_pf2_list)

    # Initial state |1010101010⟩
    init_state_str = '10' * (n // 2)
    init_state = Statevector.from_label(init_state_str).data

    # Observable: Z on qubit 0 (rightmost)
    z1 = SparsePauliOp('I' * (n - 1) + 'Z', 1)

    # =========================================================================
    # 2. IDEAL (EXACT) EVOLUTION
    # =========================================================================
    print("\n2. Computing Ideal (Exact) Evolution...")
    t_num_dense = 50
    U_dt_ideal_dense = expH(sum(H_ordered), t / t_num_dense, use_jax=False)

    ideal_expvals = []
    for i in range(t_num_dense + 1):
        state_t = matrix_power(U_dt_ideal_dense, i) @ init_state
        expval = expect_value(z1.to_matrix(), state_t)
        ideal_expvals.append(expval)
        if i % 10 == 0:
            print(f"   t={i * t / t_num_dense:.1f}: ⟨Z₁⟩ = {expval:.6f}")

    ideal_expvals = np.array(ideal_expvals)
    print(f"   Computed {len(ideal_expvals)} time points")

    # =========================================================================
    # 3. TROTTER (PF2) EVOLUTION
    # =========================================================================
    print("\n3. Computing Trotter (PF2) Evolution...")
    U_dt_appro = pf(H_ordered, dt, 2, 1)

    trott_expvals = []
    for i in range(r + 1):
        state_t = matrix_power(U_dt_appro, i) @ init_state
        expval = expect_value(z1.to_matrix(), state_t)
        trott_expvals.append(expval)
        if i % 10 == 0:
            print(f"   step {i} (t={i * dt:.1f}): ⟨Z₁⟩ = {expval:.6f}")

    trott_expvals = np.array(trott_expvals)
    print(f"   Computed {len(trott_expvals)} time points")

    # =========================================================================
    # 4. LPD (Light Pauli Dynamics) with w*=5
    # =========================================================================
    print(f"\n4. Computing LPD with w*={w_threshold}...")

    # Build operator sequence for LPD
    ops = dt / 2 * sum(H_pf2_ordered)
    obs = z1

    # Expectation value function for |0101...01⟩ state
    exp_val_func = partial(exp_val_0101_state_pauli_rep, n_qubits=n)

    # Run LPD simulation
    sim = LowWeightPauliPropagation.from_pauli_list(obs, ops, threshold=w_threshold, nprocs=4)
    lpd_expvals = sim.run_dynamics(r, process=exp_val_func, process_every=1, verbose=True)
    lpd_expvals = np.array(lpd_expvals).real

    print(f"   Computed {len(lpd_expvals)} time points")

    # =========================================================================
    # 5. Save results
    # =========================================================================
    print("\n5. Saving results...")

    t_dense_list = np.array(range(len(ideal_expvals))) * t / t_num_dense
    t_list = np.array(range(len(trott_expvals))) * dt

    results = {
        'parameters': {
            'n': n,
            't_total': t,
            'r': r,
            'dt': dt,
            'Jx': Jx,
            'hx': hx,
            'hy': hy,
            'w_threshold': w_threshold,
            'init_state': init_state_str
        },
        't_dense_list': t_dense_list.tolist(),
        't_list': t_list.tolist(),
        'ideal_expvals': ideal_expvals.tolist(),
        'trott_expvals': trott_expvals.tolist(),
        'lpd_expvals': lpd_expvals.tolist()
    }

    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pf2_python_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"   Results saved to: {output_file}")

    # =========================================================================
    # 6. Print comparison table
    # =========================================================================
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    print(f"{'t':>6} | {'Ideal':>12} | {'Trotter':>12} | {'LPD':>12}")
    print("-" * 50)

    for t_check in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]:
        idx = int(t_check / dt)
        print(f"{t_check:>6.1f} | {ideal_expvals[idx]:>12.6f} | {trott_expvals[idx]:>12.6f} | {lpd_expvals[idx]:>12.6f}")

    print("-" * 50)
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = run_pf2_simulation()
