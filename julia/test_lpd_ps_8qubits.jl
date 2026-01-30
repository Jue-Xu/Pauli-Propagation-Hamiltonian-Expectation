# Test 8-qubit version for faster runtime
using PauliStrings
const ps = PauliStrings

# Override parameters for 8 qubits
const n = 8
const t_total = 5.0
const r = 50
const dt = t_total / r
const hx = 0.8
const hy = 0.9
const Jx = 1.0
const w_threshold = 5

# Include the functions (but skip the main execution)
function build_hamiltonian_ps(n::Int, hx::Float64, hy::Float64, Jx::Float64)
    H = ps.Operator(n)
    for i in 1:n
        H += hx, "X", i
    end
    for i in 1:n
        H += hy, "Y", i
    end
    for i in 1:n-1
        H += Jx, "X", i, "X", i+1
    end
    return H
end

function expect_alternating_state_v2(O)
    total = 0.0 + 0.0im
    coeffs, strings = ps.op_to_strings(O)
    for (coeff, str) in zip(coeffs, strings)
        is_z_type = true
        sign = 1
        for (i, c) in enumerate(str)
            if c == 'X' || c == 'Y'
                is_z_type = false
                break
            elseif c == 'Z'
                if isodd(i)
                    sign *= -1
                end
            end
        end
        if is_z_type
            total += coeff * sign
        end
    end
    return real(total)
end

# Main test with continuous evolution (rk4) - much simpler
function run_lpd_continuous(n, r, dt, w_threshold)
    println("Building Hamiltonian for n=$n qubits...")
    H = build_hamiltonian_ps(n, hx, hy, Jx)
    
    # Observable: Z on site 1
    O_init = ps.Operator(n)
    O_init += "Z", 1
    
    println("Initial ⟨Z₁⟩ = ", expect_alternating_state_v2(O_init))
    
    expvals = Float64[]
    O = deepcopy(O_init)
    push!(expvals, expect_alternating_state_v2(O))
    
    println("\nRunning LPD continuous evolution (rk4)...")
    for step in 1:r
        O = ps.rk4(H, O, dt; heisenberg=true)
        O = ps.truncate(O, w_threshold)
        O = ps.trim(ps.cutoff(O, 1e-15), 10^7)
        push!(expvals, expect_alternating_state_v2(O))
        
        if step % 10 == 0
            coeffs, _ = ps.op_to_strings(O)
            println("Step $step: #Paulis = ", length(coeffs), ", ⟨Z₁⟩ = ", round(expvals[end], digits=6))
        end
    end
    
    return expvals
end

println("="^60)
println("LPD Test with n=$n qubits, w*=$w_threshold")
println("="^60)

@time expvals = run_lpd_continuous(n, r, dt, w_threshold)

println("\n--- Summary ---")
for t_check in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    idx = Int(t_check / dt) + 1
    println("  t=$t_check: ⟨Z₁⟩ = ", round(expvals[idx], digits=6))
end

# Save results
using DelimitedFiles
times = collect(0:dt:t_total)
writedlm("pf2_lpd_paulistrings_8qubits.csv", hcat(times, expvals), ',')
println("\nResults saved to pf2_lpd_paulistrings_8qubits.csv")
