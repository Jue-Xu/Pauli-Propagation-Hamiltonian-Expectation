import math
from qiskit.quantum_info import SparsePauliOp, Operator, commutator

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


def w_weight_pauli(spo: SparsePauliOp, max_weight: int) -> SparsePauliOp:
    """Return a SparsePauliOp with only low-weight Pauli operators."""
    mask = [sum(1 for c in str(p) if c != 'I') == max_weight for p in spo.paulis]
    return SparsePauliOp(spo.paulis[mask], coeffs=spo.coeffs[mask])


def expect_value(ob, state):
    # if isinstance(state, Statevector):
    #     return np.abs(state.conj().T @ ob @ state)
    # elif isinstance(state, DensityMatrix):
    #     return np.trace(ob @ state)
    # else:
    #     raise ValueError('invalid state type')
    expval = state.conj().T @ ob @ state
    if expval.imag > 1e-7:
        print('imaginary part of expectation value is', expval.imag)
        raise ValueError('Expectation value is not real')
    return expval.real
    # return np.abs(state.conj().T @ ob @ state)
