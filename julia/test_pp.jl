using PauliPropagation

n = 4

# Create observable
psum = PauliSum(n)
add!(psum, :Z, 1, 1.0)

# Test encoding
println("--- Understanding Pauli encoding ---")
for (pauli_int, coeff) in psum
    println("Integer encoding: ", pauli_int, " Coeff: ", coeff)
    symbols = inttosymbol(pauli_int, n)
    println("  Symbols: ", symbols)
    str = inttostring(pauli_int, n)
    println("  String: ", str)
end

# Test expectation values
println("\n--- Expectation value tests ---")
println("⟨0000|Z_1|0000⟩ = ", overlapwithzero(psum), " (should be +1)")

println("\nExploring overlapwithcomputational...")
for state_int in 0:3
    bits = digits(state_int, base=2, pad=n)
    result = overlapwithcomputational(psum, state_int)
    println("  state=$state_int (bits=$bits): ⟨state|Z_1|state⟩ = $result")
end

# Test Z on different sites
println("\n--- Z on different sites ---")
for site in 1:4
    ps = PauliSum(n)
    add!(ps, :Z, site, 1.0)
    println("Z on site $site:")
    for state_int in 0:3
        result = overlapwithcomputational(ps, state_int)
        println("  state=$state_int: ⟨state|Z_$site|state⟩ = $result")
    end
end
