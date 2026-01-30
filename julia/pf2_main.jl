#=
PF2 Main Script - Complete Reproduction

This script combines all methods to reproduce Figure a from PF2.ipynb:
1. Ideal (exact) evolution
2. Trotter evolution (r=50, second-order)
3. LPD (discrete) using PauliStrings.jl
4. LPD (continuous) using PauliStrings.jl

Run this script to generate the comparison plot.
=#

using Plots
using DelimitedFiles

# Include the modules
include("pf2_reproduction.jl")
include("pf2_lpd_paulistrings.jl")

# =============================================================================
# Main function to run all methods
# =============================================================================
function run_all_methods()
    println("="^60)
    println("PF2 Reproduction - Julia Implementation")
    println("="^60)
    println("\nParameters:")
    println("  n = $n qubits")
    println("  t = $t_total")
    println("  r = $r Trotter steps")
    println("  dt = $dt")
    println("  hx = $hx, hy = $hy, Jx = $Jx")
    println("  w* = $w_threshold (weight threshold)")
    println("  Initial state: |1010101010⟩")
    println("  Observable: Z on qubit $n")
    println()

    # =========================================================================
    # 1. Exact and Trotter Evolution (matrix methods)
    # =========================================================================
    println("="^60)
    println("1. EXACT AND TROTTER EVOLUTION")
    println("="^60)

    H = build_hamiltonian_matrix(n, hx, hy, Jx)
    H_list = build_hamiltonian_terms(n, hx, hy, Jx)
    ψ0 = build_initial_state("1010101010")
    Z1 = build_observable_Z1(n)
    times = collect(0:dt:t_total)

    println("\nComputing Ideal Evolution...")
    @time ideal_expvals = ideal_evolution(H, ψ0, Z1, times)

    println("\nComputing Trotter Evolution (r=$r)...")
    @time trott_expvals = trotter_evolution(H_list, ψ0, Z1, r, dt)

    # =========================================================================
    # 2. LPD using PauliStrings.jl
    # =========================================================================
    println("\n" * "="^60)
    println("2. LPD USING PauliStrings.jl")
    println("="^60)

    H_ps = build_hamiltonian_ps(n, hx, hy, Jx)
    gates = build_trotter_gate_sequence(n, dt, hx, hy, Jx)

    O_init = ps.Operator(n)
    O_init += "Z", n

    println("\nComputing LPD Discrete Evolution (w*=$w_threshold)...")
    @time lpd_discrete_expvals = lpd_discrete_evolution(n, gates, deepcopy(O_init), r, w_threshold)

    println("\nComputing LPD Continuous Evolution (w*=$w_threshold)...")
    @time lpd_continuous_expvals = lpd_continuous_evolution(H_ps, deepcopy(O_init), dt, r, w_threshold)

    # =========================================================================
    # 3. Create comparison plot
    # =========================================================================
    println("\n" * "="^60)
    println("3. GENERATING PLOT")
    println("="^60)

    p = plot(times, ideal_expvals,
             label="Ideal",
             color=:black,
             linewidth=2,
             linestyle=:solid)

    plot!(p, times, trott_expvals,
          label="Trotter (r=$r)",
          color=:magenta,
          linewidth=2,
          linestyle=:dashdot,
          marker=:circle,
          markersize=3)

    plot!(p, times, lpd_discrete_expvals,
          label="LPD discrete (w*=$w_threshold)",
          color=:green,
          linewidth=2,
          linestyle=:dash)

    plot!(p, times, lpd_continuous_expvals,
          label="LPD continuous (w*=$w_threshold)",
          color=:blue,
          linewidth=2,
          linestyle=:dot)

    xlabel!(p, "Evolution time t")
    ylabel!(p, "⟨Z₁⟩")
    title!(p, "MFI Model: n=$n, hx=$hx, hy=$hy, Jx=$Jx")

    # Save plot
    savefig(p, "pf2_julia_comparison.pdf")
    savefig(p, "pf2_julia_comparison.png")
    println("\nPlot saved to pf2_julia_comparison.pdf and pf2_julia_comparison.png")

    # =========================================================================
    # 4. Save numerical results
    # =========================================================================
    println("\n" * "="^60)
    println("4. SAVING RESULTS")
    println("="^60)

    results = hcat(times, ideal_expvals, trott_expvals, lpd_discrete_expvals, lpd_continuous_expvals)
    header = "time,ideal,trotter,lpd_discrete,lpd_continuous"
    open("pf2_julia_results.csv", "w") do io
        println(io, header)
        writedlm(io, results, ',')
    end
    println("Results saved to pf2_julia_results.csv")

    # =========================================================================
    # 5. Print comparison summary
    # =========================================================================
    println("\n" * "="^60)
    println("5. COMPARISON SUMMARY")
    println("="^60)

    println("\nExpectation values at selected times:")
    println("-"^70)
    println("  t    |   Ideal   |  Trotter  | LPD discrete | LPD continuous")
    println("-"^70)
    for i in [1, 11, 21, 31, 41, 51]  # t = 0, 1, 2, 3, 4, 5
        t = times[i]
        id = ideal_expvals[i]
        tr = trott_expvals[i]
        ld = lpd_discrete_expvals[i]
        lc = lpd_continuous_expvals[i]
        @printf(" %4.1f  | %9.5f | %9.5f |   %9.5f  |   %9.5f\n", t, id, tr, ld, lc)
    end
    println("-"^70)

    # Compute errors
    trotter_error = maximum(abs.(trott_expvals .- ideal_expvals))
    lpd_discrete_vs_trotter = maximum(abs.(lpd_discrete_expvals .- trott_expvals))
    lpd_continuous_vs_trotter = maximum(abs.(lpd_continuous_expvals .- trott_expvals))
    lpd_discrete_vs_continuous = maximum(abs.(lpd_discrete_expvals .- lpd_continuous_expvals))

    println("\nMax errors:")
    println("  |Trotter - Ideal|_max         = $trotter_error")
    println("  |LPD_discrete - Trotter|_max  = $lpd_discrete_vs_trotter")
    println("  |LPD_continuous - Trotter|_max = $lpd_continuous_vs_trotter")
    println("  |LPD_discrete - LPD_continuous|_max = $lpd_discrete_vs_continuous")

    return times, ideal_expvals, trott_expvals, lpd_discrete_expvals, lpd_continuous_expvals
end

# Need Printf for formatted output
using Printf

# Run everything
if abspath(PROGRAM_FILE) == @__FILE__
    run_all_methods()
end
