import math
import numpy as np
from qiskit.quantum_info import SparsePauliOp, Operator, commutator
from spd.OperatorSequence import *
from spd.SparsePauliDynamics import *
from spd.LightPauliDynamics import *
from spd.utils import unpackbits
def pauli_rotation_evo(G, theta, P, coeffs=None):
    if coeffs is None:
        result = math.cos(theta*2)*P + math.sin(theta*2)*1j*G@P
    elif len(coeffs) == 2:
        result = coeffs[0] * P + coeffs[1] * 1j * G@P
    return result


# given a SparsePauliOp, return a list of the sum of square of coefficients of every weight of Pauli operators
def pauli_weight_norm(spo: SparsePauliOp) -> dict:
    """
    Calculate the norm of each weight of Pauli operators in a SparsePauliOp.
    """
    weight_norm = {}
    for pauli, coeff in zip(spo.paulis, spo.coeffs):
        # print(pauli, coeff)
        weight = sum([1 for c in pauli if str(c) != 'I'])
        # print(weight)
        if weight not in weight_norm:
            weight_norm[weight] = 0
        weight_norm[weight] += abs(coeff)**2
    return weight_norm


def low_weight_pauli(spo: SparsePauliOp, max_weight: int) -> SparsePauliOp:
    """Return a SparsePauliOp with only low-weight Pauli operators."""
    mask = [sum(1 for c in str(p) if c != 'I') <= max_weight for p in spo.paulis]
    return SparsePauliOp(spo.paulis[mask], coeffs=spo.coeffs[mask])

def pauli_weight(pauli):
    """
    Calculate the weight of a Pauli operator.
    """
    return sum(1 for c in str(pauli) if c != 'I')

def truncate_high_weight_pauli(ob, w: int):
    """
    Truncate the observable by removing high-weight (>w) Pauli terms.
    Input: observable (SparsePauliOp), weight threshold (int)
    Output: truncated observable (SparsePauliOp)
    """
    temp = []
    for item in list(ob):
        # print(item, item.paulis)
        if pauli_weight(item.paulis[0]) <= w:
            temp.append(item)
    return sum(temp)
    # return sum(temp).simplify()

def w_weight_pauli(spo: SparsePauliOp, max_weight: int) -> SparsePauliOp:
    """Return a SparsePauliOp with only low-weight Pauli operators."""
    mask = [sum(1 for c in str(p) if c != 'I') == max_weight for p in spo.paulis]
    return SparsePauliOp(spo.paulis[mask], coeffs=spo.coeffs[mask])


def expect_value(ob, state, tol=1e-7):
    # if isinstance(state, Statevector):
    #     return np.abs(state.conj().T @ ob @ state)
    # elif isinstance(state, DensityMatrix):
    #     return np.trace(ob @ state)
    # else:
    #     raise ValueError('invalid state type')
    expval = state.conj().T @ ob @ state
    if expval.imag > tol:
        print('imaginary part of expectation value is', expval.imag)
        raise ValueError('Expectation value is not real')
    return expval.real
    # return np.abs(state.conj().T @ ob @ state)



def exp_val_0101_state_pauli_rep(observable, n_qubits=None, pattern=None):
    """
    Compute the expectation value of an observable (PauliRepresentation) with respect to an 0101 state.
    Default pattern is |0101...01⟩ for even number of qubits.
    
    Args:
        observable: PauliRepresentation object
        n_qubits: Number of qubits (uses observable.nq if not specified)  
        pattern: Optional custom pattern string (e.g., "0101" or "010101")
    
    Returns:
        The expectation value Tr(ρ O)
    """
    
    if n_qubits is None:
        n_qubits = observable.nq
    
    # Determine the pattern
    if pattern is None:
        if n_qubits % 2 != 0:
            raise ValueError("Default pattern requires even number of qubits")
        # Create alternating pattern "0101...01"
        pattern = '01' * (n_qubits // 2)
    else:
        if len(pattern) != n_qubits:
            raise ValueError(f"Pattern length {len(pattern)} doesn't match number of qubits {n_qubits}")
    
    # Find Z-type terms (only I and Z operators)
    z_mask = observable.ztype()
    
    # Initialize expectation value
    expectation = 0.0
    
    # Iterate through Z-type terms only
    z_indices = np.where(z_mask)[0]
    
    for idx in z_indices:
        # Get the coefficient
        coeff = observable.coeffs[idx]
        
        # Convert the bit representation to get Z positions
        # For PauliRepresentation, we need to unpack the bits to see where Z operators are
        z_bits = unpackbits(observable.bits[idx:idx+1, :observable.nq], n_qubits)[0]
        
        # Count Z operators at positions where pattern has '1'
        sign_exponent = 0
        for i, bit in enumerate(pattern):
            if z_bits[i] and bit == '1':  # Z operator at position where pattern has '1'
                sign_exponent += 1
        
        # This term contributes with appropriate sign
        phase_factor = (-1j) ** observable.phase[idx]  # Apply the Pauli phase
        contribution = coeff * phase_factor * ((-1) ** sign_exponent)
        expectation += contribution
    
    return expectation


def exp_val_all_zeros_pauli_rep(observable):
    """
    Compute the expectation value of an observable (PauliRepresentation) with respect to the all-zeros state |0^n⟩.
    
    The algorithm uses the fact that the Pauli decomposition of ρ_0 = |0^n⟩⟨0^n| 
    contains only Pauli strings with I and Z operators, each with coefficient 1/2^n.
    
    Args:
        observable: PauliRepresentation object
    
    Returns:
        The expectation value Tr(ρ_0 O)
    """
    
    # Find Z-type terms (only I and Z operators)
    z_mask = observable.ztype()
    
    # Initialize expectation value
    expectation = 0.0
    
    # Iterate through Z-type terms only
    z_indices = np.where(z_mask)[0]
    
    for idx in z_indices:
        # Get the coefficient
        coeff = observable.coeffs[idx]
        
        # Apply the Pauli phase
        phase_factor = (-1j) ** observable.phase[idx]
        
        # For all-zeros state, all Z-type terms contribute with their full coefficient
        # (no sign changes based on Z positions since |0⟩ is eigenstate of Z with eigenvalue +1)
        contribution = coeff * phase_factor
        expectation += contribution
    
    return expectation


def decompose_by_weight_pauli_rep(observable):
    """
    Decompose a PauliRepresentation observable into a dictionary organized by weight.
    
    Args:
        observable: PauliRepresentation object
    
    Returns:
        Dictionary where:
        - key: integer weight (number of non-identity operators)
        - value: PauliRepresentation containing only terms of that weight
    """
    
    weight_dict = {}
    
    # Get weights for all Pauli terms
    weights = observable.weights
    
    # Convert coeffs to numpy array if it's a list
    if isinstance(observable.coeffs, list):
        coeffs_array = np.array(observable.coeffs, dtype=np.complex128)
    else:
        coeffs_array = observable.coeffs
    
    # Group terms by weight
    for weight in np.unique(weights):
        # Find indices of terms with this weight
        weight_mask = (weights == weight)
        weight_indices = np.where(weight_mask)[0]
        
        if len(weight_indices) > 0:
            # Extract bits, phase, and coefficients for this weight
            weight_bits = observable.bits[weight_indices]
            weight_phase = observable.phase[weight_indices]
            weight_coeffs = coeffs_array[weight_indices]
            
            # Create PauliRepresentation for this weight
            weight_pauli_rep = PauliRepresentation(
                bits=weight_bits,
                phase=weight_phase,
                nq=observable.nq,
                coeffs=weight_coeffs
            )
            
            weight_dict[int(weight)] = weight_pauli_rep
    
    return weight_dict

def weight_2norm_distr(ob, n):
    dist = []
    for w in range(1, n+1):
        ob_w = decompose_by_weight_pauli_rep(ob)
        # print(ob_w)
        if w in ob_w:
            dist.append(ob_w[w].p2norm()**2)
        else:
            # print(f'No weight {w} Paulis')
            dist.append(0)

    return dist
