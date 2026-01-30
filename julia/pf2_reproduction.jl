#=
PF2 Reproduction in Julia
Reproduces the numerical results from PF2.ipynb using Julia packages

This script computes ⟨Z₁⟩ for:
1. Ideal (exact) evolution
2. Trotter evolution (r=50, second-order)
3. LPD with w*=5 truncation using:
   - PauliStrings.jl (continuous rk4)
   - PauliStrings.jl (discrete Pauli rotations)
   - PauliPropagation.jl (discrete gates)
=#

using LinearAlgebra
using SparseArrays
using Plots

# =============================================================================
# Parameters (matching PF2.ipynb)
# =============================================================================
const n = 10           # Number of qubits
const t_total = 5.0    # Total evolution time
const r = 50           # Number of Trotter steps
const dt = t_total / r # Time step = 0.1
const hx = 0.8         # X field strength
const hy = 0.9         # Y field strength
const Jx = 1.0         # XX coupling strength
const w_threshold = 5  # Weight truncation threshold

# =============================================================================
# Pauli matrices
# =============================================================================
const σI = [1.0+0im 0; 0 1]
const σX = [0.0+0im 1; 1 0]
const σY = [0.0+0im -1im; 1im 0]
const σZ = [1.0+0im 0; 0 -1]

# =============================================================================
# Build full Hamiltonian matrix
# =============================================================================
function kron_n(ops::Vector{Matrix{ComplexF64}})
    result = ops[1]
    for i in 2:length(ops)
        result = kron(result, ops[i])
    end
    return result
end

function single_site_op(op::Matrix{ComplexF64}, site::Int, n::Int)
    ops = [i == site ? op : σI for i in 1:n]
    return kron_n(ops)
end

function two_site_op(op1::Matrix{ComplexF64}, site1::Int,
                     op2::Matrix{ComplexF64}, site2::Int, n::Int)
    ops = [σI for _ in 1:n]
    ops[site1] = op1
    ops[site2] = op2
    return kron_n(ops)
end

function build_hamiltonian_matrix(n::Int, hx::Float64, hy::Float64, Jx::Float64)
    dim = 2^n
    H = zeros(ComplexF64, dim, dim)

    # X field terms: hx * Σ Xᵢ
    for i in 1:n
        H += hx * single_site_op(σX, i, n)
    end

    # Y field terms: hy * Σ Yᵢ
    for i in 1:n
        H += hy * single_site_op(σY, i, n)
    end

    # XX coupling terms: Jx * Σ XᵢXⱼ (nearest neighbor, open BC)
    for i in 1:n-1
        H += Jx * two_site_op(σX, i, σX, i+1, n)
    end

    return H
end

function build_hamiltonian_terms(n::Int, hx::Float64, hy::Float64, Jx::Float64)
    dim = 2^n

    # H_x: all X terms
    H_x = zeros(ComplexF64, dim, dim)
    for i in 1:n
        H_x += hx * single_site_op(σX, i, n)
    end

    # H_y: all Y terms
    H_y = zeros(ComplexF64, dim, dim)
    for i in 1:n
        H_y += hy * single_site_op(σY, i, n)
    end

    # H_xx_even: XX on even bonds (1-2, 3-4, 5-6, ...)
    H_xx_even = zeros(ComplexF64, dim, dim)
    for i in 1:2:n-1
        H_xx_even += Jx * two_site_op(σX, i, σX, i+1, n)
    end

    # H_xx_odd: XX on odd bonds (2-3, 4-5, 6-7, ...)
    H_xx_odd = zeros(ComplexF64, dim, dim)
    for i in 2:2:n-1
        H_xx_odd += Jx * two_site_op(σX, i, σX, i+1, n)
    end

    return [H_x, H_y, H_xx_even, H_xx_odd]
end

# =============================================================================
# Build initial state |1010101010⟩
# =============================================================================
function build_initial_state(pattern::String)
    n = length(pattern)
    dim = 2^n
    ψ = zeros(ComplexF64, dim)

    # Convert binary pattern to index
    # Julia uses 1-indexing, bit pattern maps to computational basis
    # |1010101010⟩ means qubit 1 is |1⟩, qubit 2 is |0⟩, etc.
    idx = 0
    for (i, c) in enumerate(pattern)
        if c == '1'
            idx += 2^(n - i)  # Big-endian convention
        end
    end
    ψ[idx + 1] = 1.0  # +1 for Julia 1-indexing
    return ψ
end

# =============================================================================
# Observable: Z on last qubit (matching Qiskit convention 'IIIIIIIIIZ')
# =============================================================================
function build_observable_Z1(n::Int)
    # Z on the last qubit (qubit n in Julia 1-indexing)
    return single_site_op(σZ, n, n)
end

# =============================================================================
# Expectation value
# =============================================================================
function expect_value(O::Matrix{ComplexF64}, ψ::Vector{ComplexF64})
    return real(ψ' * O * ψ)
end

# =============================================================================
# 1. IDEAL EVOLUTION (exact matrix exponentiation)
# =============================================================================
function ideal_evolution(H::Matrix{ComplexF64}, ψ0::Vector{ComplexF64},
                        Z1::Matrix{ComplexF64}, times::Vector{Float64})
    expvals = Float64[]

    for t in times
        if t == 0.0
            push!(expvals, expect_value(Z1, ψ0))
        else
            U = exp(-1im * H * t)
            ψ_t = U * ψ0
            push!(expvals, expect_value(Z1, ψ_t))
        end
    end

    return expvals
end

# =============================================================================
# 2. TROTTER EVOLUTION (second-order product formula)
# =============================================================================
function trotter_step_pf2(ψ::Vector{ComplexF64}, H_list::Vector{Matrix{ComplexF64}}, dt::Float64)
    # Second-order Trotter-Suzuki: symmetric formula
    # Forward: exp(-i dt/2 H_x) exp(-i dt/2 H_y) exp(-i dt/2 H_xx_even) exp(-i dt/2 H_xx_odd)
    # Backward: exp(-i dt/2 H_xx_odd) exp(-i dt/2 H_xx_even) exp(-i dt/2 H_y) exp(-i dt/2 H_x)

    ψ_new = copy(ψ)

    # Forward half-step
    for H in H_list
        U = exp(-1im * H * dt / 2)
        ψ_new = U * ψ_new
    end

    # Backward half-step (reverse order)
    for H in reverse(H_list)
        U = exp(-1im * H * dt / 2)
        ψ_new = U * ψ_new
    end

    return ψ_new
end

function trotter_evolution(H_list::Vector{Matrix{ComplexF64}}, ψ0::Vector{ComplexF64},
                          Z1::Matrix{ComplexF64}, r::Int, dt::Float64)
    expvals = Float64[]
    ψ = copy(ψ0)

    # Record at step 0
    push!(expvals, expect_value(Z1, ψ))

    # Evolve for r steps
    for step in 1:r
        ψ = trotter_step_pf2(ψ, H_list, dt)
        push!(expvals, expect_value(Z1, ψ))
    end

    return expvals
end

# =============================================================================
# Main execution
# =============================================================================
function run_exact_and_trotter()
    println("Building Hamiltonian...")
    H = build_hamiltonian_matrix(n, hx, hy, Jx)
    H_list = build_hamiltonian_terms(n, hx, hy, Jx)

    println("Building initial state |1010101010⟩...")
    ψ0 = build_initial_state("1010101010")

    println("Building observable Z₁...")
    Z1 = build_observable_Z1(n)

    # Time points
    times = collect(0:dt:t_total)

    println("\n--- Computing Ideal Evolution ---")
    @time ideal_expvals = ideal_evolution(H, ψ0, Z1, times)

    println("\n--- Computing Trotter Evolution (r=$r) ---")
    @time trott_expvals = trotter_evolution(H_list, ψ0, Z1, r, dt)

    return times, ideal_expvals, trott_expvals
end

# Run if this is the main script
if abspath(PROGRAM_FILE) == @__FILE__
    times, ideal_expvals, trott_expvals = run_exact_and_trotter()

    println("\n--- Results ---")
    println("Time points: ", length(times))
    println("Ideal expvals[1:5]: ", ideal_expvals[1:5])
    println("Trotter expvals[1:5]: ", trott_expvals[1:5])

    # Save results for comparison
    using DelimitedFiles
    results = hcat(times, ideal_expvals, trott_expvals)
    writedlm("pf2_exact_trotter_results.csv", results, ',')
    println("\nResults saved to pf2_exact_trotter_results.csv")
end
