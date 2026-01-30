#=
PauliStrings.jl - Discrete Gate Implementation via Commutator

Uses the Heisenberg evolution formula:
  O' = cos(θ)O + (i·sin(θ)/2)[G, O]

This leverages PauliStrings.jl's optimized commutator() function.
=#

using PauliStrings
using DelimitedFiles
using ProgressMeter
using Printf
using JSON

const ps = PauliStrings

# =============================================================================
# Parameters (matching Python implementation)
# =============================================================================
const n = 10
const t_total = 5.0
const r = 50
const dt = t_total / r
const hx = 0.8
const hy = 0.9
const Jx = 1.0
const w_threshold = 5

# =============================================================================
# Core Functions
# =============================================================================
"""
Apply exp(-iθG/2) to operator O in Heisenberg picture.

Correct formula for Pauli rotation where G² = I:
  O' = cos²(θ/2)*O + sin²(θ/2)*G*O*G + (i*sin(θ)/2)*[G, O]

This handles both commuting and anticommuting parts correctly:
  - Commuting parts (GOG = O): unchanged
  - Anticommuting parts (GOG = -O): O → cos(θ)*O + i*sin(θ)*G*O
"""
function apply_pauli_rotation(O::ps.Operator, G::ps.Operator, θ::Float64)
    c2 = cos(θ/2)^2  # cos²(θ/2) = (1 + cos(θ))/2
    s2 = sin(θ/2)^2  # sin²(θ/2) = (1 - cos(θ))/2
    s = sin(θ)

    GOG = G * O * G  # Conjugation by G
    comm = ps.commutator(G, O)  # [G, O]

    return c2 * O + s2 * GOG + (1im * s / 2) * comm
end

"""
Build gate list for second-order Trotter decomposition.
Order: X → Y → XX_even → XX_odd → XX_odd → XX_even → Y → X
"""
function build_trotter_gates(n, dt, hx, hy, Jx)
    gates = Tuple{ps.Operator, Float64}[]

    # Forward half: X, Y, XX_even, XX_odd
    for i in 1:n
        G = ps.Operator(n); G += "X", i
        push!(gates, (G, hx * dt))
    end
    for i in 1:n
        G = ps.Operator(n); G += "Y", i
        push!(gates, (G, hy * dt))
    end
    for i in 1:2:n-1  # even bonds (1-2, 3-4, ...)
        G = ps.Operator(n); G += "X", i, "X", i+1
        push!(gates, (G, Jx * dt))
    end
    for i in 2:2:n-1  # odd bonds (2-3, 4-5, ...)
        G = ps.Operator(n); G += "X", i, "X", i+1
        push!(gates, (G, Jx * dt))
    end

    # Backward half: XX_odd, XX_even, Y, X (symmetric)
    for i in 2:2:n-1
        G = ps.Operator(n); G += "X", i, "X", i+1
        push!(gates, (G, Jx * dt))
    end
    for i in 1:2:n-1
        G = ps.Operator(n); G += "X", i, "X", i+1
        push!(gates, (G, Jx * dt))
    end
    for i in 1:n
        G = ps.Operator(n); G += "Y", i
        push!(gates, (G, hy * dt))
    end
    for i in 1:n
        G = ps.Operator(n); G += "X", i
        push!(gates, (G, hx * dt))
    end

    return gates
end

"""
Compute expectation value ⟨ψ|O|ψ⟩ for |ψ⟩ = |1010101010⟩
Only Z-type strings (containing only I and Z) contribute.
"""
function expect_ps(O)
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
                # |ψ⟩ has qubit i in |1⟩ if i is even
                if iseven(i)
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

# =============================================================================
# Main Evolution
# =============================================================================
function run_paulistrings_discrete()
    println("="^70)
    println("PauliStrings.jl - Discrete Gates via Commutator")
    println("="^70)
    println("n=$n, t=$t_total, r=$r, dt=$dt, w*=$w_threshold")
    println("="^70)

    # Build gates
    gates = build_trotter_gates(n, dt, hx, hy, Jx)
    println("\nNumber of gates per Trotter step: $(length(gates))")

    # Initialize observable O = Z₁
    O = ps.Operator(n)
    O += "Z", 1

    # Storage
    expvals = Float64[]
    npauli_list = Int[]
    t_list = Float64[]

    # Record initial
    push!(t_list, 0.0)
    push!(expvals, expect_ps(O))
    coeffs, _ = ps.op_to_strings(O)
    push!(npauli_list, length(coeffs))

    println("\nRunning evolution...")
    p = Progress(r; desc="PS.jl discrete: ", showspeed=true)

    for step in 1:r
        # Apply all gates in one Trotter step
        for (G, θ) in gates
            O = apply_pauli_rotation(O, G, θ)
        end

        # Truncate after each Trotter step
        O = ps.truncate(O, w_threshold)
        O = ps.trim(ps.cutoff(O, 1e-15), 10^7)

        # Record results
        push!(t_list, step * dt)
        push!(expvals, expect_ps(O))
        coeffs, _ = ps.op_to_strings(O)
        push!(npauli_list, length(coeffs))

        next!(p; showvalues=[(:step, step), (:nPaulis, length(coeffs))])
    end

    return t_list, expvals, npauli_list
end

function main()
    t_list, expvals, npauli_list = run_paulistrings_discrete()

    # Save results to CSV
    script_dir = @__DIR__
    output_csv = joinpath(script_dir, "paulistrings_discrete_results.csv")
    results = hcat(t_list, expvals, npauli_list)
    writedlm(output_csv, results, ',')
    println("\n\nResults saved to: $output_csv")

    # Save to JSON for easy Python comparison
    output_json = joinpath(script_dir, "paulistrings_discrete_results.json")
    data = Dict(
        "parameters" => Dict(
            "n" => n,
            "t_total" => t_total,
            "r" => r,
            "dt" => dt,
            "hx" => hx,
            "hy" => hy,
            "Jx" => Jx,
            "w_threshold" => w_threshold,
            "method" => "discrete_commutator"
        ),
        "t_list" => t_list,
        "expvals" => expvals,
        "npauli_list" => npauli_list
    )
    open(output_json, "w") do f
        JSON.print(f, data, 2)
    end
    println("Results saved to: $output_json")

    # Print summary table
    println("\n" * "="^60)
    println("RESULTS SUMMARY")
    println("="^60)
    println("     t | ⟨Z₁⟩       | #Paulis")
    println("-"^60)
    for t_check in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        idx = Int(t_check / dt) + 1
        @printf("  %3.1f | %10.6f | %7d\n", t_check, expvals[idx], npauli_list[idx])
    end
    println("-"^60)
    println("Final #Paulis: $(npauli_list[end])")
end

main()
