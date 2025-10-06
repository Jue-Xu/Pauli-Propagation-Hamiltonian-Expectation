import numpy as np
from qiskit.quantum_info import SparsePauliOp

from qiskit import QuantumCircuit
from qiskit.quantum_info import Clifford, Statevector, random_clifford
from qiskit.circuit.library import PauliGate
import matplotlib.pyplot as plt


# Method 2: Using Clifford Circuits (Exact Unitary 3-Design)
def generate_clifford_states(num_qubits, num_states):
    """
    Generate states using random Clifford circuits.
    Clifford group forms a unitary 3-design (hence also a 2-design).
    """
    from qiskit.quantum_info import random_clifford
    
    states = []
    for _ in range(num_states):
        # Generate random Clifford operator
        cliff = random_clifford(num_qubits)
        
        # Apply to |0⟩ state
        zero_state = Statevector.from_label('0' * num_qubits)
        state = zero_state.evolve(cliff)
        states.append(state)
    
    return states

def generate_clifford_2_design(n_qubits, n_states):
    """
    Generate a 2-design using random Clifford states.
    
    Args:
        n_qubits: Number of qubits
        n_states: Number of states to generate
        
    Returns:
        List of Statevector objects
    """
    states = []
    
    for _ in range(n_states):
        # Generate a random Clifford operator
        cliff = random_clifford(n_qubits)
        
        # Create a circuit and apply the Clifford
        qc = QuantumCircuit(n_qubits)
        qc.append(cliff.to_circuit(), range(n_qubits))
        
        # Get the state vector (applied to |0...0⟩)
        state = Statevector.from_instruction(qc)
        states.append(state)
    
    return states

def generate_random_circuit_2_design(n_qubits, n_states, depth=None):
    """
    Generate approximate 2-design using random quantum circuits.
    
    Args:
        n_qubits: Number of qubits
        n_states: Number of states to generate
        depth: Circuit depth (default: O(n_qubits²) for good approximation)
        
    Returns:
        List of Statevector objects
    """
    if depth is None:
        depth = n_qubits * n_qubits
    
    states = []
    
    # Define gate set for random circuits
    single_qubit_gates = ['rx', 'ry', 'rz']
    
    for _ in range(n_states):
        qc = QuantumCircuit(n_qubits)
        
        for d in range(depth):
            # Random single-qubit gates
            for qubit in range(n_qubits):
                gate = np.random.choice(single_qubit_gates)
                angle = np.random.uniform(0, 2*np.pi)
                getattr(qc, gate)(angle, qubit)
            
            # Entangling gates (CZ gates on neighboring qubits)
            if d % 2 == 0:
                for qubit in range(0, n_qubits-1, 2):
                    qc.cz(qubit, qubit+1)
            else:
                for qubit in range(1, n_qubits-1, 2):
                    qc.cz(qubit, qubit+1)
        
        state = Statevector.from_instruction(qc)
        states.append(state)
    
    return states