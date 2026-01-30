using PauliStrings
const ps = PauliStrings

function test_paulistrings()
    n = 10

    # Build an operator
    H = ps.Operator(n)

    # Add X field terms
    for i in 1:n
        H += 0.8, "X", i
    end

    # Add Y field terms
    for i in 1:n
        H += 0.9, "Y", i
    end

    # Add XX terms
    for i in 1:n-1
        H += 1.0, "X", i, "X", i+1
    end

    println("Hamiltonian:")
    println(H)

    # Create observable Z on site 1
    O = ps.Operator(n)
    O += "Z", 1
    println("\nObservable:")
    println(O)

    # Test op_to_strings
    coeffs, strings = ps.op_to_strings(O)
    println("\nop_to_strings:")
    for (c, s) in zip(coeffs, strings)
        println("  coeff=$c, string=$s")
    end

    # Test evolution
    println("\n--- Testing evolution ---")
    dt = 0.1
    O_evolved = ps.rk4(H, O, dt; heisenberg=true)
    println("After rk4 evolution:")
    println("Number of terms: ", length(ps.op_to_strings(O_evolved)[1]))

    # Test truncation
    O_truncated = ps.truncate(O_evolved, 5)
    println("After truncation to weight 5:")
    println("Number of terms: ", length(ps.op_to_strings(O_truncated)[1]))
    
    return H, O
end

test_paulistrings()
