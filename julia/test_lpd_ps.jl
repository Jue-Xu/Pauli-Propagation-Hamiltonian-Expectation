include("pf2_lpd_paulistrings.jl")

function test_lpd()
    # Quick test
    println("Testing expect_alternating_state_v2...")
    O_test = ps.Operator(n)
    O_test += "Z", 1
    exp_val = expect_alternating_state_v2(O_test)
    println("Z on site 1: ", exp_val, " (expected: -1)")

    O_test2 = ps.Operator(n)
    O_test2 += "Z", 2
    exp_val2 = expect_alternating_state_v2(O_test2)
    println("Z on site 2: ", exp_val2, " (expected: +1)")

    # Test ZZ
    O_test3 = ps.Operator(n)
    O_test3 += "Z", 1, "Z", 2
    exp_val3 = expect_alternating_state_v2(O_test3)
    println("ZZ on sites 1,2: ", exp_val3, " (expected: -1)")

    # Test the main function
    println("\nRunning LPD evolution test (5 steps)...")
    H = build_hamiltonian_ps(n, hx, hy, Jx)
    O_init = ps.Operator(n)
    O_init += "Z", 1

    # Just 5 steps for testing
    expvals = Float64[]
    O = deepcopy(O_init)
    push!(expvals, expect_alternating_state_v2(O))
    
    for step in 1:5
        O = ps.rk4(H, O, dt; heisenberg=true)
        O = ps.truncate(O, w_threshold)
        push!(expvals, expect_alternating_state_v2(O))
        coeffs, _ = ps.op_to_strings(O)
        println("Step $step: #Paulis = ", length(coeffs), ", expval = ", round(expvals[end], digits=6))
    end
    
    println("\nExpectation values: ", expvals)
end

test_lpd()
