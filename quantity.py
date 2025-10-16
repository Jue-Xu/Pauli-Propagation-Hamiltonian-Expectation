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

    """
    magic = -sum([abs(c)**2 * math.log(abs(c)**2) for c in op.coeffs])
    return magic

from qiskit.quantum_info import Pauli
from itertools import product
    # def magic(state,pauligroup):#state:Statevector, pauligroup:string (from Pauli_group(n)) 输入Pauli减少运算次数
def state_linear_magic(state):
    d = len(state)
    m = []
    pauligroup = [''.join(ops) for ops in product(['I','X','Y','Z'], repeat=int(np.log2(d)))]
    for j, paulistr in enumerate(pauligroup):
        a = state.evolve(Pauli(paulistr))
        m1 = state.inner(a)
        m1 = np.sqrt(np.real(np.conj(m1)*m1))
        m.append(m1)
    magica = 1-d*np.average(np.power(m,4))
    return magica     

# #n-qubit Pauli group(return a string)
# def Pauli_group(n):
#     String = ['I','X','Y','Z']
#     if n == 1:
#         return String
#     else:
#         a = Pauli_group(n-1)
#         j = 0
#         b = []
#         while j < 4:
#             c = String[j]
#             i = 0
#             while i < np.power(4,n-1):
#                 b.append(c+a[i])
#                 i += 1
#             j += 1
#         return b
    
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
