using PauliPropagation

n = 10

# The function expects onebitinds to be a list of site indices where bit=1
# For alternating state |1010101010⟩, sites 1,3,5,7,9 are in |1⟩
onebitinds = [1, 3, 5, 7, 9]  # Odd sites are |1⟩

psum = PauliSum(n)
add!(psum, :Z, 1, 1.0)

result = overlapwithcomputational(psum, onebitinds)
println("Z_1 on |1010101010⟩ (via index list): ", result, " (expected -1)")

# Test Z on site 2 (which is |0⟩)
psum2 = PauliSum(n)
add!(psum2, :Z, 2, 1.0)
result2 = overlapwithcomputational(psum2, onebitinds)
println("Z_2 on |1010101010⟩ (via index list): ", result2, " (expected +1)")

# Test Z on site 10 (last qubit, which is |0⟩ since 10 is even)
psum10 = PauliSum(n)
add!(psum10, :Z, n, 1.0)
result10 = overlapwithcomputational(psum10, onebitinds)
println("Z_10 on |1010101010⟩ (via index list): ", result10, " (expected +1)")

# Now what about Z on site 9 (which is |1⟩)?
psum9 = PauliSum(n)
add!(psum9, :Z, 9, 1.0)
result9 = overlapwithcomputational(psum9, onebitinds)
println("Z_9 on |1010101010⟩ (via index list): ", result9, " (expected -1)")

println("\n--- Now let's verify the full circuit simulation ---")

# Build Trotter circuit for one step
circuit = []

# X rotations
for i in 1:n
    push!(circuit, PauliRotation(:X, i, 0.8 * 0.1))  # hx * dt
end

# Y rotations
for i in 1:n
    push!(circuit, PauliRotation(:Y, i, 0.9 * 0.1))  # hy * dt
end

# XX even bonds
for i in 1:2:n-1
    push!(circuit, PauliRotation([:X, :X], [i, i+1], 1.0 * 0.1))  # Jx * dt
end

# XX odd bonds
for i in 2:2:n-1
    push!(circuit, PauliRotation([:X, :X], [i, i+1], 1.0 * 0.1))
end

# Backward (same gates again for second-order)
for i in 2:2:n-1
    push!(circuit, PauliRotation([:X, :X], [i, i+1], 1.0 * 0.1))
end
for i in 1:2:n-1
    push!(circuit, PauliRotation([:X, :X], [i, i+1], 1.0 * 0.1))
end
for i in 1:n
    push!(circuit, PauliRotation(:Y, i, 0.9 * 0.1))
end
for i in 1:n
    push!(circuit, PauliRotation(:X, i, 0.8 * 0.1))
end

println("Circuit has $(length(circuit)) gates")

# Initialize observable: Z on site 1
obs = PauliSum(n)
add!(obs, :Z, 1, 1.0)

# Propagate for one Trotter step
obs_after = propagate(circuit, obs; max_weight=5)

println("\nAfter one Trotter step:")
println("Number of Pauli terms: ", length(obs_after))

# Compute expectation value
expval = overlapwithcomputational(obs_after, onebitinds)
println("⟨Z_1⟩ after one step: ", expval)
