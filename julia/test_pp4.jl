using PauliPropagation

n = 10

# Verify: 341 in binary should have bit[1]=1
println("341 in binary:")
for i in 0:9
    println("  bit $i = ", (341 >> i) & 1)
end

# But let's also check if maybe the state representation uses UInt8
# which might truncate 341
println("\nChecking 341 as different integer types:")
println("  341 % 256 = ", 341 % 256)  # = 85
println("  85 in binary: ", bitstring(UInt8(85)))

# So state 341 with UInt8 becomes 85!
# 85 = 0b01010101 which is different

# Let me verify
println("\n--- State 85 analysis ---")
b85 = digits(85, base=2, pad=8)
println("digits(85, base=2, pad=8) = ", b85)
psum = PauliSum(n)
add!(psum, :Z, 1, 1.0)
result85 = overlapwithcomputational(psum, 85)
println("Z_1 on state 85: ", result85)

# Hmm, if the system uses UInt8, then we need n <= 8
# Let me check what the default type is for the PauliSum
psum_check = PauliSum(n)
println("\n--- PauliSum type info ---")
println("typeof(psum): ", typeof(psum_check))

# For n=10, we need at least UInt16
# Let me create a PauliSum with larger integer type
psum16 = PauliSum(n, UInt16, Float64)
add!(psum16, :Z, 1, 1.0)
result16 = overlapwithcomputational(psum16, UInt16(341))
println("\nWith UInt16: Z_1 on state 341 = ", result16)
