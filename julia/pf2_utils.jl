"""
Shared utilities for PF2 numerical simulations.

This module provides common functions for building Hamiltonians, computing
expectation values, and defining the initial state for the MFI model simulations.
"""
module PF2Utils

using LinearAlgebra
using SparseArrays

export n, Jx, hx, hy, t_final, r, dt, w_threshold, t_num_dense
export initial_state_pattern, Z1_qubit
export pauli_I, pauli_X, pauli_Y, pauli_Z
export build_mfi_hamiltonian_terms, build_full_hamiltonian
export single_qubit_op, two_qubit_op, kron_n
export initial_state_vector, expect_alternating_state_dense

# Global parameters matching Python notebook
const n = 10                    # Number of qubits
const Jx = 1.0                  # XX coupling strength
const hx = 0.8                  # X field strength
const hy = 0.9                  # Y field strength
const t_final = 5.0             # Total evolution time
const r = 50                    # Number of Trotter steps
const dt = t_final / r          # Time step (0.1)
const w_threshold = 5           # Weight threshold for LPD (w*=5)
const t_num_dense = 50          # Number of dense time points for ideal evolution

# Qubit conventions:
# Python (Qiskit): Rightmost qubit is index 0. 'IIIIIIIIIZ' = Z on qubit 0
# Julia: We use 1-indexing. To match Python's observable on rightmost qubit,
#        we apply Z on qubit 1 (which in Julia corresponds to rightmost position)
# Initial state: Python '1010101010' means (from right to left):
#   qubit 0: |1⟩, qubit 1: |0⟩, qubit 2: |1⟩, ...
# In Julia with 1-indexing (from right to left, matching Qiskit bit ordering):
#   qubit 1: |1⟩, qubit 2: |0⟩, qubit 3: |1⟩, qubit 4: |0⟩, ...
# So odd qubits (1,3,5,7,9) are in |1⟩, even qubits (2,4,6,8,10) are in |0⟩

const initial_state_pattern = [isodd(i) ? 1 : 0 for i in 1:n]  # [1,0,1,0,1,0,1,0,1,0]
const Z1_qubit = 1  # Z on qubit 1 (rightmost, matching Qiskit's 'IIIIIIIIIZ')

# Pauli matrices
const pauli_I = sparse(ComplexF64[1 0; 0 1])
const pauli_X = sparse(ComplexF64[0 1; 1 0])
const pauli_Y = sparse(ComplexF64[0 -im; im 0])
const pauli_Z = sparse(ComplexF64[1 0; 0 -1])

"""
    kron_n(ops::Vector{SparseMatrixCSC{ComplexF64, Int64}})

Compute the tensor product of a list of operators.
"""
function kron_n(ops)
    result = ops[1]
    for i in 2:length(ops)
        result = kron(result, ops[i])
    end
    return result
end

"""
    single_qubit_op(op, qubit, n_qubits)

Create an n-qubit operator with `op` acting on `qubit` (1-indexed) and I elsewhere.
Qubit 1 is the rightmost (least significant) position.
"""
function single_qubit_op(op, qubit, n_qubits)
    ops = [pauli_I for _ in 1:n_qubits]
    ops[n_qubits - qubit + 1] = op  # Reverse index for consistency with Qiskit
    return kron_n(ops)
end

"""
    two_qubit_op(op1, qubit1, op2, qubit2, n_qubits)

Create an n-qubit operator with `op1` on `qubit1`, `op2` on `qubit2`, and I elsewhere.
"""
function two_qubit_op(op1, qubit1, op2, qubit2, n_qubits)
    ops = [pauli_I for _ in 1:n_qubits]
    ops[n_qubits - qubit1 + 1] = op1
    ops[n_qubits - qubit2 + 1] = op2
    return kron_n(ops)
end

"""
    build_mfi_hamiltonian_terms(n_qubits, hx, hy, Jx)

Build the separate Hamiltonian terms for the Mixed-Field Ising (MFI) model.
Returns a dictionary with keys: :H_x, :H_y, :H_xx_even, :H_xx_odd
"""
function build_mfi_hamiltonian_terms(n_qubits, hx_val, hy_val, Jx_val)
    dim = 2^n_qubits

    # X field terms: hx * Σ Xᵢ
    H_x = spzeros(ComplexF64, dim, dim)
    for i in 1:n_qubits
        H_x += hx_val * single_qubit_op(pauli_X, i, n_qubits)
    end

    # Y field terms: hy * Σ Yᵢ
    H_y = spzeros(ComplexF64, dim, dim)
    for i in 1:n_qubits
        H_y += hy_val * single_qubit_op(pauli_Y, i, n_qubits)
    end

    # XX coupling terms (even bonds): Jx * Σ XᵢXᵢ₊₁ for i=1,3,5,...
    H_xx_even = spzeros(ComplexF64, dim, dim)
    for i in 1:2:(n_qubits-1)
        H_xx_even += Jx_val * two_qubit_op(pauli_X, i, pauli_X, i+1, n_qubits)
    end

    # XX coupling terms (odd bonds): Jx * Σ XᵢXᵢ₊₁ for i=2,4,6,...
    H_xx_odd = spzeros(ComplexF64, dim, dim)
    for i in 2:2:(n_qubits-1)
        H_xx_odd += Jx_val * two_qubit_op(pauli_X, i, pauli_X, i+1, n_qubits)
    end

    return Dict(
        :H_x => H_x,
        :H_y => H_y,
        :H_xx_even => H_xx_even,
        :H_xx_odd => H_xx_odd
    )
end

"""
    build_full_hamiltonian(n_qubits, hx, hy, Jx)

Build the full MFI Hamiltonian: H = hx Σ Xᵢ + hy Σ Yᵢ + Jx Σ XᵢXᵢ₊₁
"""
function build_full_hamiltonian(n_qubits, hx_val, hy_val, Jx_val)
    terms = build_mfi_hamiltonian_terms(n_qubits, hx_val, hy_val, Jx_val)
    return terms[:H_x] + terms[:H_y] + terms[:H_xx_even] + terms[:H_xx_odd]
end

"""
    initial_state_vector(pattern, n_qubits)

Create the initial state vector |pattern⟩ where pattern[i]=1 means qubit i is in |1⟩.
Qubit 1 is rightmost (least significant bit).
"""
function initial_state_vector(pattern, n_qubits)
    # Compute basis index: |b_n ... b_2 b_1⟩ has index Σ b_i * 2^(i-1)
    idx = sum(pattern[i] * 2^(i-1) for i in 1:n_qubits)
    state = zeros(ComplexF64, 2^n_qubits)
    state[idx + 1] = 1.0  # Julia is 1-indexed
    return state
end

"""
    expect_alternating_state_dense(operator, n_qubits)

Compute ⟨ψ|O|ψ⟩ where |ψ⟩ = |1010101010⟩ (alternating pattern).
Works for dense/sparse matrix operators.
"""
function expect_alternating_state_dense(operator, n_qubits)
    state = initial_state_vector(initial_state_pattern, n_qubits)
    return real(dot(state, operator * state))
end

end # module
