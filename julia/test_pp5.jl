using PauliPropagation

n = 10

psum = PauliSum(n)
add!(psum, :Z, 1, 1.0)

# Check the type of state parameter
println("typeof(psum): ", typeof(psum))

# Try with explicit UInt32
state_uint32 = UInt32(341)
result = overlapwithcomputational(psum, state_uint32)
println("Z_1 on state UInt32(341): ", result)

# Let's see what overlapwithcomputational actually does
# Look at the source or test with simpler cases

# Test with specific states that should give definitive results
println("\n--- Systematic test ---")
for s in 0:10
    psum_test = PauliSum(n)
    add!(psum_test, :Z, 1, 1.0)
    r = overlapwithcomputational(psum_test, s)
    bits = bitstring(UInt32(s))
    println("state=$s, bit[0]=$(s & 1): Z_1 = $r")
end

# Based on state=1 giving -1, the formula seems correct
# Let me check if the problem is with how n affects the computation
println("\n--- Testing with n=8 ---")
n8 = 8
psum8 = PauliSum(n8)
add!(psum8, :Z, 1, 1.0)
# State 85 = 0b01010101
result8 = overlapwithcomputational(psum8, 85)
println("n=8, Z_1 on state 85 (0b01010101): ", result8)
println("Bit 0 of 85: ", 85 & 1)

# The issue is 85 & 1 = 1, so we expect -1 but get +1
# Unless the encoding is big-endian?

# Try: for n=8, state 85 = 0b01010101
# In big-endian interpretation: leftmost bit (site 1?) = 0, rightmost bit (site 8?) = 1
# This would mean site 1 is |0⟩ → Z_1 = +1

# But state=1 gave -1 for Z_1, which matches site 1 = bit 0 = |1⟩

# Wait - maybe for n=10, something else is happening
# Let me verify the encoding step by step
println("\n--- Step by step ---")
n10 = 10
# Create state where only site 1 is |1⟩
state_site1 = 1  # 0b0000000001
psum_z1 = PauliSum(n10)
add!(psum_z1, :Z, 1, 1.0)
r1 = overlapwithcomputational(psum_z1, state_site1)
println("state with bit 0 = 1: Z_1 = ", r1, " (expected -1 since site 1 is |1⟩)")

# Create state where only site 2 is |1⟩
state_site2 = 2  # 0b0000000010
psum_z2 = PauliSum(n10)
add!(psum_z2, :Z, 2, 1.0)
r2 = overlapwithcomputational(psum_z2, state_site2)
println("state with bit 1 = 1: Z_2 = ", r2, " (expected -1 since site 2 is |1⟩)")

# OK so far so good. Now for 341:
# 341 = 0b0101010101 = bits 0,2,4,6,8 are set = sites 1,3,5,7,9 are |1⟩
# So Z_1 should give -1 since site 1 is |1⟩
# But we get +1...

# Unless there's parity involved?
# For a product state |ψ⟩, ⟨ψ|Z_1|ψ⟩ should just be ±1 based on site 1's state

# Let me read what overlapwithcomputational does
@show methods(overlapwithcomputational)
