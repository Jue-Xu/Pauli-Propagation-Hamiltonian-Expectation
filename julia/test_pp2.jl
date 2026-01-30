using PauliPropagation

n = 10

# Alternating state |1010101010⟩
# Qiskit: qubit 0 (rightmost) = |1⟩, qubit 1 = |0⟩, etc.
# In PauliPropagation: site 1 = qubit 0 = |1⟩, site 2 = qubit 1 = |0⟩
# State integer: bit_i = 1 if isodd(i), 0 otherwise
# = 1*2^0 + 0*2^1 + 1*2^2 + 0*2^3 + 1*2^4 + 0*2^5 + 1*2^6 + 0*2^7 + 1*2^8 + 0*2^9
# = 1 + 4 + 16 + 64 + 256 = 341

state_int = sum(isodd(i) ? 2^(i-1) : 0 for i in 1:n)
println("Alternating state integer: ", state_int)
println("Binary: ", digits(state_int, base=2, pad=n))

# Test Z on site 1 (should give -1 since site 1 is in |1⟩)
psum = PauliSum(n)
add!(psum, :Z, 1, 1.0)
result = overlapwithcomputational(psum, state_int)
println("\n⟨1010101010|Z_1|1010101010⟩ = ", result, " (expected: -1)")

# Test Z on site 10 (should give +1 since site 10 is in |0⟩)
psum10 = PauliSum(n)
add!(psum10, :Z, n, 1.0)
result10 = overlapwithcomputational(psum10, state_int)
println("⟨1010101010|Z_10|1010101010⟩ = ", result10, " (expected: +1)")

# The observable in PF2.ipynb is Z on the rightmost qubit = Z on site 1
# But wait - Qiskit 'IIIIIIIIIZ' puts Z on qubit 0 (rightmost)
# In PauliPropagation with site numbering 1-10:
# - Site 1 = qubit 0 (rightmost)
# - Site 10 = qubit 9 (leftmost)

# So for 'IIIIIIIIIZ' (Z on qubit 0), we use Z on site 1

println("\n--- Verifying observable ---")
println("Observable 'IIIIIIIIIZ' = Z on qubit 0 = Z on site 1")
println("Initial ⟨ψ|Z_1|ψ⟩ = ", result)

# Now test propagation
println("\n--- Propagation test ---")
# Build a simple gate sequence
gate = PauliRotation(:X, 1, 0.1)
result_prop = propagate([gate], psum)
println("After RX(0.1) on site 1:")
for (pauli_int, coeff) in result_prop
    str = inttostring(pauli_int, n)
    println("  ", str, " : ", coeff)
end

# Compute expectation value after propagation
expval = overlapwithcomputational(result_prop, state_int)
println("Expectation value after propagation: ", expval)
