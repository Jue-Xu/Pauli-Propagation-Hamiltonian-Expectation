"""
Exact (Ideal) and Trotter evolution for PF2 simulation.

This script computes:
1. Ideal evolution using exact matrix exponentiation
2. Second-order Trotter (PF2) evolution with r=50 steps

Uses direct matrix exponentiation for simplicity and accuracy.
"""

include("pf2_utils.jl")
using .PF2Utils
using LinearAlgebra
using SparseArrays
using JLD2

function run_exact_trotter()
    println("="^60)
    println("PF2 Exact and Trotter Evolution")
    println("="^60)
    println("Parameters: n=$n, t=$t_final, r=$r, dt=$dt")
    println("Hamiltonian: MFI with Jx=$Jx, hx=$hx, hy=$hy")
    println("Initial state: |1010101010⟩ (alternating)")
    println("Observable: Z on qubit 1 (rightmost)")
    println("="^60)

    # Build Hamiltonian terms
    println("\nBuilding Hamiltonian...")
    H_terms = build_mfi_hamiltonian_terms(n, hx, hy, Jx)
    H_full = H_terms[:H_x] + H_terms[:H_y] + H_terms[:H_xx_even] + H_terms[:H_xx_odd]

    # Z observable on qubit 1
    Z1 = single_qubit_op(pauli_Z, Z1_qubit, n)

    # Initial state |1010101010⟩
    ψ0 = initial_state_vector(initial_state_pattern, n)
    println("Initial state prepared: norm = ", norm(ψ0))

    # ============================================================
    # 1. IDEAL (EXACT) EVOLUTION
    # ============================================================
    println("\n" * "="^60)
    println("Computing Ideal Evolution...")
    println("="^60)

    # Dense time points for smooth ideal curve
    t_dense_list = range(0, t_final, length=t_num_dense+1)

    # Compute ideal evolution at dense time points
    ideal_expvals = Float64[]
    H_dense = Matrix(H_full)  # Convert to dense for expm

    for (i, t) in enumerate(t_dense_list)
        # U = exp(-i H t)
        U_ideal = exp(-im * H_dense * t)
        ψt = U_ideal * ψ0
        expval = real(dot(ψt, Z1 * ψt))
        push!(ideal_expvals, expval)
        if i % 10 == 1
            println("  t = $(round(t, digits=2)): ⟨Z₁⟩ = $(round(expval, digits=6))")
        end
    end

    println("\nIdeal evolution complete. $(length(ideal_expvals)) time points.")

    # ============================================================
    # 2. SECOND-ORDER TROTTER (PF2) EVOLUTION
    # ============================================================
    println("\n" * "="^60)
    println("Computing Trotter (PF2) Evolution...")
    println("="^60)
    println("Trotter order: 2, steps: $r, dt: $dt")

    # Precompute matrix exponentials for Trotter steps
    # The ordering from Python is: [H_x, H_y, H_xx_even, H_xx_odd, H_xx_odd, H_xx_even, H_y, H_x]

    # Convert to dense for matrix exponential
    H_x_dense = Matrix(H_terms[:H_x])
    H_y_dense = Matrix(H_terms[:H_y])
    H_xx_even_dense = Matrix(H_terms[:H_xx_even])
    H_xx_odd_dense = Matrix(H_terms[:H_xx_odd])

    # Compute exp(-i dt/2 H) for each term
    exp_Hx = exp(-im * (dt/2) * H_x_dense)
    exp_Hy = exp(-im * (dt/2) * H_y_dense)
    exp_Hxx_even = exp(-im * (dt/2) * H_xx_even_dense)
    exp_Hxx_odd = exp(-im * (dt/2) * H_xx_odd_dense)

    # Second-order Trotter step: forward then backward
    function trotter_step_pf2(ψ)
        # Forward: X, Y, XX_even, XX_odd
        ψ = exp_Hx * ψ
        ψ = exp_Hy * ψ
        ψ = exp_Hxx_even * ψ
        ψ = exp_Hxx_odd * ψ
        # Backward: XX_odd, XX_even, Y, X
        ψ = exp_Hxx_odd * ψ
        ψ = exp_Hxx_even * ψ
        ψ = exp_Hy * ψ
        ψ = exp_Hx * ψ
        return ψ
    end

    # Compute Trotter evolution
    trott_expvals = Float64[]
    ψ = copy(ψ0)

    for step in 0:r
        # Measure expectation value
        expval = real(dot(ψ, Z1 * ψ))
        push!(trott_expvals, expval)

        if step % 10 == 0
            println("  step $step (t=$(round(step*dt, digits=2))): ⟨Z₁⟩ = $(round(expval, digits=6))")
        end

        # Apply Trotter step (except after last measurement)
        if step < r
            ψ = trotter_step_pf2(ψ)
        end
    end

    println("\nTrotter evolution complete. $(length(trott_expvals)) time points.")

    # ============================================================
    # Compare results
    # ============================================================
    println("\n" * "="^60)
    println("Comparison at key time points:")
    println("="^60)
    println("  t=0: Ideal=$(round(ideal_expvals[1], digits=6)), Trotter=$(round(trott_expvals[1], digits=6))")

    # Find indices for comparison (Trotter uses dt=0.1, ideal uses same spacing)
    for t_check in [1.0, 2.0, 3.0, 4.0, 5.0]
        trott_idx = Int(t_check / dt) + 1
        ideal_idx = Int(t_check / (t_final/t_num_dense)) + 1
        println("  t=$t_check: Ideal=$(round(ideal_expvals[ideal_idx], digits=6)), Trotter=$(round(trott_expvals[trott_idx], digits=6))")
    end

    # ============================================================
    # Save results
    # ============================================================
    t_list = collect(0:dt:t_final)
    results = Dict(
        "t_dense_list" => collect(t_dense_list),
        "ideal_expvals" => ideal_expvals,
        "t_list" => t_list,
        "trott_expvals" => trott_expvals,
        "parameters" => Dict(
            "n" => n,
            "Jx" => Jx,
            "hx" => hx,
            "hy" => hy,
            "t_final" => t_final,
            "r" => r,
            "dt" => dt
        )
    )

    output_file = joinpath(@__DIR__, "pf2_exact_results.jld2")
    @save output_file results
    println("\nResults saved to: $output_file")

    println("\n" * "="^60)
    println("Done!")
    println("="^60)

    return ideal_expvals, trott_expvals
end

# Run if this is the main script
if abspath(PROGRAM_FILE) == @__FILE__
    run_exact_trotter()
end
