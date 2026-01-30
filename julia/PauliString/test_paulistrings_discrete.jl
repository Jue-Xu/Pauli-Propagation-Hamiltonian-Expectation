#=
PauliStrings.jl Test: Discrete Gate Implementation

This file demonstrates discrete gate Heisenberg evolution using PauliStrings.jl
and compares results against a Python reference implementation.

Physics: Time evolution of ⟨Z₁⟩ for a 1D Ising model with transverse fields:
  H = Jx Σᵢ XᵢXᵢ₊₁ + hx Σᵢ Xᵢ + hy Σᵢ Yᵢ

Evolution uses second-order Trotter decomposition with discrete gates.
Heisenberg evolution formula for Pauli rotation exp(-iθG/2) where G² = I:
  O' = cos²(θ/2)·O + sin²(θ/2)·G·O·G + (i·sin(θ)/2)·[G, O]

Author: [Your name]
Date: January 2025
=#

using PauliStrings
using JSON
using Plots

const ps = PauliStrings

# =============================================================================
# Parameters
# =============================================================================
const n = 10           # Number of qubits
const t_total = 5.0    # Total evolution time
const r = 50           # Number of Trotter steps
const dt = t_total / r # Time step
const hx = 0.8         # X field strength
const hy = 0.9         # Y field strength
const Jx = 1.0         # XX coupling strength
const w_threshold = 5  # Weight truncation threshold

# =============================================================================
# Discrete Gate Evolution
# =============================================================================
"""
Apply Pauli rotation exp(-iθG/2) to operator O in Heisenberg picture.

For Pauli generator G (where G² = I):
  O' = cos²(θ/2)·O + sin²(θ/2)·G·O·G + (i·sin(θ)/2)·[G, O]

This correctly handles:
  - Commuting parts [G,O]=0: O unchanged
  - Anticommuting parts {G,O}=0: O → cos(θ)·O + i·sin(θ)·G·O
"""
function apply_pauli_rotation(O::ps.Operator, G::ps.Operator, θ::Float64)
    c2 = cos(θ/2)^2
    s2 = sin(θ/2)^2
    s = sin(θ)

    GOG = G * O * G              # Conjugation by G
    comm = ps.commutator(G, O)   # [G, O]

    return c2 * O + s2 * GOG + (1im * s / 2) * comm
end

"""
Build second-order Trotter gate sequence.
Order: X → Y → XX_even → XX_odd → XX_odd → XX_even → Y → X
"""
function build_trotter_gates(n, dt, hx, hy, Jx)
    gates = Tuple{ps.Operator, Float64}[]

    # Forward half
    for i in 1:n
        G = ps.Operator(n); G += "X", i
        push!(gates, (G, hx * dt))
    end
    for i in 1:n
        G = ps.Operator(n); G += "Y", i
        push!(gates, (G, hy * dt))
    end
    for i in 1:2:n-1  # even bonds
        G = ps.Operator(n); G += "X", i, "X", i+1
        push!(gates, (G, Jx * dt))
    end
    for i in 2:2:n-1  # odd bonds
        G = ps.Operator(n); G += "X", i, "X", i+1
        push!(gates, (G, Jx * dt))
    end

    # Backward half (symmetric)
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
Compute expectation value ⟨ψ|O|ψ⟩ for |ψ⟩ = |1010101010⟩.
Only Z-type strings contribute (those with only I and Z).
"""
function expect_neel_state(O)
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
                # |ψ⟩ = |1010...⟩: qubit i is |1⟩ if i is even
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
function run_evolution()
    println("="^60)
    println("PauliStrings.jl Discrete Gate Test")
    println("="^60)
    println("Parameters:")
    println("  n = $n qubits")
    println("  t_total = $t_total")
    println("  r = $r Trotter steps")
    println("  dt = $dt")
    println("  (hx, hy, Jx) = ($hx, $hy, $Jx)")
    println("  w* = $w_threshold (weight truncation)")
    println("="^60)

    gates = build_trotter_gates(n, dt, hx, hy, Jx)
    println("\nGates per Trotter step: $(length(gates))")

    # Initialize O = Z₁
    O = ps.Operator(n)
    O += "Z", 1

    # Storage
    t_list = Float64[0.0]
    expvals = Float64[expect_neel_state(O)]
    npauli_list = Int[1]

    println("\nRunning evolution...")
    for step in 1:r
        # Apply all gates in Trotter step
        for (G, θ) in gates
            O = apply_pauli_rotation(O, G, θ)
        end

        # Truncate
        O = ps.truncate(O, w_threshold)
        O = ps.trim(ps.cutoff(O, 1e-15), 10^7)

        # Record
        push!(t_list, step * dt)
        push!(expvals, expect_neel_state(O))
        coeffs, _ = ps.op_to_strings(O)
        push!(npauli_list, length(coeffs))

        if step % 10 == 0
            println("  Step $step/$r: #Paulis = $(npauli_list[end])")
        end
    end

    return t_list, expvals, npauli_list
end

# =============================================================================
# Load Python Data
# =============================================================================
function load_python_data()
    script_dir = @__DIR__
    py_file = joinpath(script_dir, "pf2_python_results.json")

    if !isfile(py_file)
        error("Python data not found: $py_file")
    end

    data = JSON.parsefile(py_file)
    return (
        t_list = Float64.(data["t_list"]),
        ideal = Float64.(data["ideal_expvals"]),
        trotter = Float64.(data["trott_expvals"]),
        lpd = Float64.(data["lpd_expvals"])
    )
end

# =============================================================================
# Plotting
# =============================================================================
function plot_comparison(jl_t, jl_expvals, py_data, output_file)
    # Left panel: Expectation values
    p1 = plot(py_data.t_list, py_data.ideal,
        color=:black, lw=2, label="Ideal",
        xlabel="Evolution time t", ylabel="⟨Z₁⟩",
        title="Expectation Value Comparison",
        legend=:topright, xlim=(0, 5))

    plot!(p1, py_data.t_list, py_data.trotter,
        color=:gray, lw=1.5, ls=:dash, label="Trotter (r=50)")

    plot!(p1, py_data.t_list, py_data.lpd,
        color=:forestgreen, lw=2.5, label="LPD (Python)")

    plot!(p1, jl_t, jl_expvals,
        color=:darkorange, lw=2.5, ls=:dash, label="LPD (PauliStrings.jl)")

    # Right panel: Difference
    min_len = min(length(py_data.lpd), length(jl_expvals))
    diff = py_data.lpd[1:min_len] .- jl_expvals[1:min_len]
    t_diff = py_data.t_list[1:min_len]

    max_diff = maximum(abs.(diff))
    mean_diff = sum(abs.(diff)) / length(diff)

    p2 = plot(t_diff, diff,
        color=:crimson, lw=2, label="",
        xlabel="Evolution time t", ylabel="Δ⟨Z₁⟩ (Python - Julia)",
        title="Difference Between Implementations",
        xlim=(0, 5))

    hline!(p2, [0], color=:black, lw=0.5, ls=:dash, label="")

    annotate!(p2, 4.5, maximum(diff) * 0.8,
        text("Max |diff|: $(round(max_diff, sigdigits=3))\nMean |diff|: $(round(mean_diff, sigdigits=3))",
             :right, 9))

    # Combine plots
    fig = plot(p1, p2, layout=(1, 2), size=(1200, 450))
    savefig(fig, output_file)
    println("\nSaved: $output_file")

    return fig
end

# =============================================================================
# Main
# =============================================================================
function main()
    # Run Julia evolution
    jl_t, jl_expvals, npauli_list = run_evolution()

    # Load Python reference
    println("\nLoading Python reference data...")
    py_data = load_python_data()

    # Print comparison table
    println("\n" * "="^70)
    println("RESULTS COMPARISON")
    println("="^70)
    println("     t  │    Ideal    │   Trotter   │  Python LPD │ PauliStrings │    Diff")
    println("─"^70)

    for t_check in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        idx = Int(t_check / dt) + 1
        diff = py_data.lpd[idx] - jl_expvals[idx]
        println("  $(lpad(t_check, 4)) │ $(lpad(round(py_data.ideal[idx], digits=6), 11)) │ " *
                "$(lpad(round(py_data.trotter[idx], digits=6), 11)) │ " *
                "$(lpad(round(py_data.lpd[idx], digits=6), 11)) │ " *
                "$(lpad(round(jl_expvals[idx], digits=6), 12)) │ $(round(diff, sigdigits=2))")
    end
    println("─"^70)
    println("Final #Paulis: $(npauli_list[end])")

    # Plot comparison
    script_dir = @__DIR__
    output_file = joinpath(script_dir, "test_paulistrings_comparison.pdf")
    plot_comparison(jl_t, jl_expvals, py_data, output_file)

    # Save Julia results to JSON
    output_json = joinpath(script_dir, "test_paulistrings_results.json")
    results = Dict(
        "parameters" => Dict(
            "n" => n,
            "t_total" => t_total,
            "r" => r,
            "dt" => dt,
            "hx" => hx,
            "hy" => hy,
            "Jx" => Jx,
            "w_threshold" => w_threshold
        ),
        "t_list" => jl_t,
        "expvals" => jl_expvals,
        "npauli_list" => npauli_list
    )
    open(output_json, "w") do f
        JSON.print(f, results, 2)
    end
    println("Saved: $output_json")

    println("\n✓ Test complete!")
end

main()
