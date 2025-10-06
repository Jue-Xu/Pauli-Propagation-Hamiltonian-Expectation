import math
import numpy as np
from qiskit.quantum_info import SparsePauliOp, Operator, commutator, DensityMatrix, partial_trace, entropy, Statevector


def entangle_entropy(state, sub_n):
    rdm = partial_trace(DensityMatrix(Statevector(state)), list(range(0, sub_n)))
    return entropy(rdm)

def operator_magic(op):
    """
    Compute the operator magic of a given operator.
    Operator magic is defined as:
    M(O) = log2(||O||_F^2 / (1/2^n * ||O||_1^2))
    where ||O||_F is the Frobenius norm and ||O||_1 is the 1-norm (sum of absolute values of coefficients in Pauli basis).
    
    Args:
        op: A qiskit Operator or SparsePauliOp  object. 
            If SparsePauliOp, it will be converted to Operator.
    """
    magic = -sum([abs(c)**2 * math.log(abs(c)**2) for c in op.coeffs])
    return magic

# def state_magic(state):
    
def compute_entanglement_entropy(statevector, subsystem_qubits, n):
    """
    Compute the von Neumann entanglement entropy for a subsystem
    
    Args:
        statevector: The quantum state
        subsystem_qubits: List of qubit indices for the subsystem
        n: Total number of qubits
    
    Returns:
        Entanglement entropy
    """
    # Convert statevector to density matrix
    dm = DensityMatrix(statevector)
    
    # Determine which qubits to trace out
    all_qubits = list(range(n))
    qubits_to_trace = [q for q in all_qubits if q not in subsystem_qubits]
    
    # Partial trace to get reduced density matrix
    rho_reduced = partial_trace(dm, qubits_to_trace)
    
    # Compute von Neumann entropy
    # S = -Tr(ρ log ρ)
    eigenvalues = np.linalg.eigvalsh(rho_reduced.data)
    # Filter out numerical zeros
    eigenvalues = eigenvalues[eigenvalues > 1e-15]
    entropy_val = -np.sum(eigenvalues * np.log2(eigenvalues))
    
    return entropy_val

def compute_mps_fidelity(state_mps, state_exact):
    """
    Compute fidelity between MPS approximation and exact state
    """
    fidelity = np.abs(np.vdot(state_exact.data, state_mps.data))**2
    return fidelity


def compute_observables(state, n):
    """
    Compute expectation values of various observables
    """
    observables = {}
    
    # Magnetization in X direction
    mag_x = 0
    for j in range(n):
        pauli_str = ['I'] * n
        pauli_str[j] = 'X'
        op = SparsePauliOp.from_list([(''.join(pauli_str[::-1]), 1.0)])
        mag_x += state.expectation_value(op).real
    observables['mag_x'] = mag_x / n
    
    # Magnetization in Y direction
    mag_y = 0
    for j in range(n):
        pauli_str = ['I'] * n
        pauli_str[j] = 'Y'
        op = SparsePauliOp.from_list([(''.join(pauli_str[::-1]), 1.0)])
        mag_y += state.expectation_value(op).real
    observables['mag_y'] = mag_y / n
    
    # Magnetization in Z direction
    mag_z = 0
    for j in range(n):
        pauli_str = ['I'] * n
        pauli_str[j] = 'Z'
        op = SparsePauliOp.from_list([(''.join(pauli_str[::-1]), 1.0)])
        mag_z += state.expectation_value(op).real
    observables['mag_z'] = mag_z / n
    
    return observables



# def exact_evolution(n, J_x, h_x, h_y, time, periodic=True):
#     """
#     Compute exact time evolution using matrix exponentiation
#     (for comparison with MPS approximation)
#     """
#     from scipy.linalg import expm
    
#     # Build Hamiltonian
#     H = build_qmfi_hamiltonian(n, J_x, h_x, h_y, periodic)
#     H_matrix = H.to_matrix()
    
#     # Time evolution operator
#     U = expm(-1j * H_matrix * time)
    
#     # Initial state |00...00⟩
#     initial_state = np.zeros(2**n, dtype=complex)
#     initial_state[0] = 1.0
    
#     # Evolved state
#     evolved_state = U @ initial_state
    
#     return Statevector(evolved_state)
