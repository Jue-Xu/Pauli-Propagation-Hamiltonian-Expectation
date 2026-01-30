#=
LPD Implementation using PauliPropagation.jl

This script implements Low-weight Pauli Dynamics using the PauliPropagation.jl
package, which uses discrete gate-based propagation that matches the Python
implementation exactly.

Key features:
- Discrete Pauli rotation gates (same as Python LPD)
- Weight truncation via max_weight parameter
- Expectation value for alternating |1010101010⟩ state
=#

using PauliPropagation
using LinearAlgebra
using JLD2

# Parameters (matching PF2.ipynb)
const n = 10           # Number of qubits
const t_total = 5.0    # Total evolution time
const r = 50           # Number of Trotter steps
const dt = t_total / r # Time step = 0.1
const hx = 0.8         # X field strength
const hy = 0.9         # Y field strength
const Jx = 1.0         # XX coupling strength
const w_threshold = 5  # Weight truncation threshold

# =============================================================================
# Build second-order Trotter circuit using PauliRotation gates
#
# For Heisenberg picture evolution with Pauli rotations:
#   exp(i G θ) P exp(-i G θ)
# where G is the Pauli generator of the rotation and θ is the angle.
#
# In PauliPropagation.jl:
#   PauliRotation(:X, i, θ) represents exp(-i θ/2 X_i)
#   PauliRotation([:X,:X], [i,j], θ) represents exp(-i θ/2 X_i X_j)
#
# For Hamiltonian H with coefficient c, evolution exp(-i c H dt/2) corresponds to
# rotation angle θ = c * dt for the Pauli generator.
# =============================================================================
function build_pf2_circuit(n::Int, dt::Float64, hx::Float64, hy::Float64, Jx::Float64)
    # Second-order Trotter structure: [H_x, H_y, H_xx_even, H_xx_odd, H_xx_odd, H_xx_even, H_y, H_x]
    # Each term gets dt/2, so rotation angle = coefficient * dt

    circuit = Vector{Any}()

    # Forward half: X terms
    for i in 1:n
        push!(circuit, PauliRotation(:X, i, hx * dt))
    end

    # Forward half: Y terms
    for i in 1:n
        push!(circuit, PauliRotation(:Y, i, hy * dt))
    end

    # Forward half: XX even bonds (1-2, 3-4, 5-6, 7-8, 9-10)
    for i in 1:2:n-1
        push!(circuit, PauliRotation([:X, :X], [i, i+1], Jx * dt))
    end

    # Forward half: XX odd bonds (2-3, 4-5, 6-7, 8-9)
    for i in 2:2:n-1
        push!(circuit, PauliRotation([:X, :X], [i, i+1], Jx * dt))
    end

    # Backward half: XX odd bonds
    for i in 2:2:n-1
        push!(circuit, PauliRotation([:X, :X], [i, i+1], Jx * dt))
    end

    # Backward half: XX even bonds
    for i in 1:2:n-1
        push!(circuit, PauliRotation([:X, :X], [i, i+1], Jx * dt))
    end

    # Backward half: Y terms
    for i in 1:n
        push!(circuit, PauliRotation(:Y, i, hy * dt))
    end

    # Backward half: X terms
    for i in 1:n
        push!(circuit, PauliRotation(:X, i, hx * dt))
    end

    return circuit
end

# =============================================================================
# Expectation value for |1010101010⟩ state using overlapwithcomputational
#
# For alternating pattern |1010101010⟩ (Qiskit convention):
# - Qubit 0 (site 1 in Julia) in |1⟩
# - Qubit 1 (site 2 in Julia) in |0⟩
# - etc.
# So odd sites (1,3,5,7,9) are in |1⟩, even sites (2,4,6,8,10) in |0⟩
#
# overlapwithcomputational expects a list of site indices where bit=1
# =============================================================================
function expect_alternating_state_pp(psum, n_qubits::Int)
    # Sites with |1⟩ state: odd sites 1, 3, 5, 7, 9
    onebitinds = collect(1:2:n_qubits)
    return overlapwithcomputational(psum, onebitinds)
end

# =============================================================================
# LPD Evolution with PauliPropagation.jl
# =============================================================================
function lpd_pauliprop_evolution(circuit, n_qubits::Int, obs_init, nsteps::Int, max_weight::Int)
    expvals = Float64[]

    # Initialize observable
    obs = deepcopy(obs_init)

    for step in 0:nsteps
        # Compute expectation value
        expval = expect_alternating_state_pp(obs, n_qubits)
        push!(expvals, expval)

        if step % 10 == 0
            nterms = length(obs)
            println("  Step $step: ⟨Z₁⟩ = $(round(expval, digits=6)), #Paulis = $nterms")
        end

        if step < nsteps
            # Propagate through circuit (Heisenberg picture)
            obs = propagate(circuit, obs; max_weight=max_weight)
        end
    end

    return expvals
end

# =============================================================================
# Main execution
# =============================================================================
function run_lpd_pauliprop()
    println("="^60)
    println("PF2 LPD Evolution using PauliPropagation.jl")
    println("="^60)
    println("Parameters: n=$n, t=$t_total, r=$r, dt=$dt")
    println("Weight threshold: w* = $w_threshold")
    println("Method: Discrete gate-based propagation")
    println("="^60)

    # Build Trotter circuit
    println("\nBuilding second-order Trotter circuit...")
    circuit = build_pf2_circuit(n, dt, hx, hy, Jx)
    println("Number of gates per Trotter step: $(length(circuit))")

    # Initialize observable: Z on site 1 (matching Python's rightmost qubit)
    # Qiskit 'IIIIIIIIIZ' = Z on qubit 0 = Z on site 1 in PauliPropagation
    println("\nInitializing observable: Z on site 1...")
    obs_init = PauliSum(n)
    add!(obs_init, :Z, 1, 1.0)

    # Verify initial expectation value
    # Site 1 is in |1⟩ for alternating state, so Z_1 should give -1
    init_expval = expect_alternating_state_pp(obs_init, n)
    println("Initial ⟨Z₁⟩ = $init_expval (expected: -1)")

    # Run LPD evolution
    println("\n--- Running LPD Evolution (w*=$w_threshold) ---")
    @time lpd_expvals = lpd_pauliprop_evolution(circuit, n, obs_init, r, w_threshold)

    # Save results
    t_list = collect(0:dt:t_total)
    results = Dict(
        "t_list" => t_list,
        "lpd_pauliprop_expvals" => lpd_expvals,
        "w_threshold" => w_threshold,
        "method" => "PauliPropagation.jl discrete",
        "parameters" => Dict(
            "n" => n,
            "Jx" => Jx,
            "hx" => hx,
            "hy" => hy,
            "t_total" => t_total,
            "r" => r,
            "dt" => dt
        )
    )

    output_file = joinpath(@__DIR__, "pf2_lpd_pauliprop_results.jld2")
    @save output_file results
    println("\nResults saved to: $output_file")

    return t_list, lpd_expvals
end

# Run if this is the main script
if abspath(PROGRAM_FILE) == @__FILE__
    t_list, lpd_expvals = run_lpd_pauliprop()

    println("\n--- Results Summary ---")
    println("Time points: $(length(t_list))")
    println("LPD expvals at t=0,1,2,3,4,5:")
    for t_check in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        idx = Int(t_check / dt) + 1
        println("  t=$t_check: ⟨Z⟩ = $(round(lpd_expvals[idx], digits=6))")
    end
end
