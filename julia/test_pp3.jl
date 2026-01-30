using PauliPropagation

n = 10

# Let me check the bit interpretation more carefully
# state_int = 341 = 0b0101010101
# This means in little-endian: bit 0 = 1, bit 1 = 0, bit 2 = 1, ...

println("Testing state encoding...")
println("state_int = 341 in binary: ", bitstring(UInt16(341)))

# Check: digits(341, base=2) = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
# This means: bit[1]=1, bit[2]=0, bit[3]=1, ...
# I.e., site 1 has bit value 1

# For Z_1, eigenvalue is +1 for |0⟩ and -1 for |1⟩
# If site 1 has bit 1, then we expect eigenvalue -1

# But we got +1, so maybe the state encoding is reversed?

# Let me test with explicit states
for test_state in [0, 1, 2, 341]
    println("\nState $test_state (bits: ", digits(test_state, base=2, pad=n), ")")
    for site in [1, 2]
        psum = PauliSum(n)
        add!(psum, :Z, site, 1.0)
        result = overlapwithcomputational(psum, test_state)
        println("  ⟨state|Z_$site|state⟩ = $result")
    end
end

# Based on the output above:
# state=1 (bits [1,0,...]) gives Z_1 = -1, Z_2 = +1
# This confirms: site i corresponds to bit i-1 in the integer
# And state 341 has bits [1,0,1,0,1,0,1,0,1,0]
# BUT wait - the test above showed ⟨1010101010|Z_1|1010101010⟩ = 1.0

# Let me look at state 341 more carefully
println("\n--- State 341 analysis ---")
b = digits(341, base=2, pad=n)
println("digits(341, base=2, pad=10) = ", b)
println("This means: bit[i] = b[i] for i=1..10")
println("bit[1] = ", b[1], " so Z_1 should give ", b[1] == 0 ? "+1" : "-1")

psum = PauliSum(n)
add!(psum, :Z, 1, 1.0)
result = overlapwithcomputational(psum, 341)
println("Actual Z_1 on state 341: ", result)

# Ah! The issue might be that 341 corresponds to |0101010101⟩ not |1010101010⟩
# In the digits representation, index 1 is the LSB
# If digits(341)[1] = 1, that means site 1 is in |1⟩

# So the expectation should be:
# Z_1: site 1 is |1⟩ → eigenvalue -1
# But we got +1. There must be a sign convention difference.

# Let me verify with state=1 which should definitely have site 1 in |1⟩
psum1 = PauliSum(n)
add!(psum1, :Z, 1, 1.0)
result1 = overlapwithcomputational(psum1, 1)
println("\nZ_1 on state 1: ", result1, " (should be -1 if site 1 is |1⟩)")
