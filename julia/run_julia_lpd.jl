#=
Run both Julia LPD implementations and save results for comparison with Python.

This script runs both:
1. PauliPropagation.jl (discrete gates)
2. PauliStrings.jl (continuous rk4)

And saves results to CSV files for plotting.

IMPORTANT: Uses the correct Qiskit convention:
- Initial state '1010101010' means:
  - qubit 0 (rightmost) = '0' → |0⟩
  - qubit 1 = '1' → |1⟩
  - qubit 2 = '0' → |0⟩
  - etc.
- So Z on qubit 0 (site 1) gives eigenvalue +1 at t=0
=#

using PauliPropagation
using PauliStrings
using DelimitedFiles
using LinearAlgebra
using SparseArrays

const ps = PauliStrings

# =============================================================================
# Parameters (matching Python PF2.ipynb)
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
# Initial state convention (matching Python)
# Python: '1010101010' → qubit 0='0', qubit 1='1', qubit 2='0', ...
# Julia site 1 = qubit 0, etc.
# So: site 1 in |0⟩, site 2 in |1⟩, site 3 in |0⟩, etc.
# Pattern: even sites (2,4,6,8,10) in |1⟩, odd sites (1,3,5,7,9) in |0⟩
# =============================================================================

# =============================================================================
# 1. PauliPropagation.jl Implementation
# =============================================================================
function build_pf2_circuit_pp(n::Int, dt::Float64, hx::Float64, hy::Float64, Jx::Float64)
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

function expect_state_pp(psum, n_qubits::Int)
    # Correct convention: even sites (2,4,6,8,10) in |1⟩
    onebitinds = collect(2:2:n_qubits)  # Sites with |1⟩ state
    return overlapwithcomputational(psum, onebitinds)
end

function run_lpd_pauliprop()
    println("\n" * "="^60)
    println("Running PauliPropagation.jl LPD")
    println("="^60)

    circuit = build_pf2_circuit_pp(n, dt, hx, hy, Jx)

    # Observable: Z on site 1
    obs = PauliSum(n)
    add!(obs, :Z, 1, 1.0)

    init_expval = expect_state_pp(obs, n)
    println("Initial ⟨Z₁⟩ = $init_expval (expected: +1)")

    expvals = Float64[]

    for step in 0:r
        expval = expect_state_pp(obs, n)
        push!(expvals, expval)

        if step % 10 == 0
            println("  Step $step (t=$(step*dt)): ⟨Z₁⟩ = $(round(expval, digits=6))")
        end

        if step < r
            obs = propagate(circuit, obs; max_weight=w_threshold)
        end
    end

    return expvals
end

# =============================================================================
# 2. PauliStrings.jl Implementation
# =============================================================================
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

function expect_state_ps(O)
    # Correct convention: even sites (2,4,6,8,10) in |1⟩
    # Site i in |1⟩ means Z_i gives -1, in |0⟩ means Z_i gives +1
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
                # Even sites (2,4,6,8,10) are in |1⟩, Z gives -1
                if iseven(i)
                    sign *= -1
                end
                # Odd sites (1,3,5,7,9) are in |0⟩, Z gives +1
            end
        end

        if is_z_type
            total += coeff * sign
        end
    end

    return real(total)
end

function run_lpd_paulistrings()
    println("\n" * "="^60)
    println("Running PauliStrings.jl LPD (continuous rk4)")
    println("="^60)

    H = build_hamiltonian_ps(n, hx, hy, Jx)

    # Observable: Z on site 1
    O = ps.Operator(n)
    O += "Z", 1

    init_expval = expect_state_ps(O)
    println("Initial ⟨Z₁⟩ = $init_expval (expected: +1)")

    expvals = Float64[]
    push!(expvals, init_expval)

    for step in 1:r
        O = ps.rk4(H, O, dt; heisenberg=true)
        O = ps.truncate(O, w_threshold)
        O = ps.trim(ps.cutoff(O, 1e-15), 10^7)

        expval = expect_state_ps(O)
        push!(expvals, expval)

        if step % 10 == 0
            println("  Step $step (t=$(step*dt)): ⟨Z₁⟩ = $(round(expval, digits=6))")
        end
    end

    return expvals
end

# =============================================================================
# 3. Exact and Trotter Evolution (for reference)
# =============================================================================
const σI = sparse(ComplexF64[1 0; 0 1])
const σX = sparse(ComplexF64[0 1; 1 0])
const σY = sparse(ComplexF64[0 -im; im 0])
const σZ = sparse(ComplexF64[1 0; 0 -1])

function kron_n(ops)
    result = ops[1]
    for i in 2:length(ops)
        result = kron(result, ops[i])
    end
    return result
end

function single_qubit_op(op, qubit, n_qubits)
    ops = [σI for _ in 1:n_qubits]
    ops[n_qubits - qubit + 1] = op
    return kron_n(ops)
end

function two_qubit_op(op1, qubit1, op2, qubit2, n_qubits)
    ops = [σI for _ in 1:n_qubits]
    ops[n_qubits - qubit1 + 1] = op1
    ops[n_qubits - qubit2 + 1] = op2
    return kron_n(ops)
end

function run_exact_trotter()
    println("\n" * "="^60)
    println("Running Exact and Trotter Evolution")
    println("="^60)

    dim = 2^n

    # Build Hamiltonian terms
    H_x = spzeros(ComplexF64, dim, dim)
    for i in 1:n
        H_x += hx * single_qubit_op(σX, i, n)
    end

    H_y = spzeros(ComplexF64, dim, dim)
    for i in 1:n
        H_y += hy * single_qubit_op(σY, i, n)
    end

    H_xx_even = spzeros(ComplexF64, dim, dim)
    for i in 1:2:n-1
        H_xx_even += Jx * two_qubit_op(σX, i, σX, i+1, n)
    end

    H_xx_odd = spzeros(ComplexF64, dim, dim)
    for i in 2:2:n-1
        H_xx_odd += Jx * two_qubit_op(σX, i, σX, i+1, n)
    end

    H_full = H_x + H_y + H_xx_even + H_xx_odd
    H_dense = Matrix(H_full)

    # Z observable on site 1
    Z1 = single_qubit_op(σZ, 1, n)

    # Initial state: site 1,3,5,7,9 in |0⟩, site 2,4,6,8,10 in |1⟩
    # Bit pattern: [0,1,0,1,0,1,0,1,0,1] for sites 1..10
    pattern = [iseven(i) ? 1 : 0 for i in 1:n]
    idx = sum(pattern[i] * 2^(i-1) for i in 1:n)
    ψ0 = zeros(ComplexF64, dim)
    ψ0[idx + 1] = 1.0

    println("Initial state: ", pattern)
    println("Initial ⟨Z₁⟩ = $(real(dot(ψ0, Z1 * ψ0))) (expected: +1)")

    # Ideal evolution
    println("\nComputing Ideal evolution...")
    ideal_expvals = Float64[]
    for i in 0:r
        t_val = i * dt
        U = exp(-im * H_dense * t_val)
        ψt = U * ψ0
        push!(ideal_expvals, real(dot(ψt, Z1 * ψt)))
    end

    # Trotter evolution
    println("Computing Trotter evolution...")
    H_x_dense = Matrix(H_x)
    H_y_dense = Matrix(H_y)
    H_xx_even_dense = Matrix(H_xx_even)
    H_xx_odd_dense = Matrix(H_xx_odd)

    exp_Hx = exp(-im * (dt/2) * H_x_dense)
    exp_Hy = exp(-im * (dt/2) * H_y_dense)
    exp_Hxx_even = exp(-im * (dt/2) * H_xx_even_dense)
    exp_Hxx_odd = exp(-im * (dt/2) * H_xx_odd_dense)

    function trotter_step(ψ)
        ψ = exp_Hx * ψ
        ψ = exp_Hy * ψ
        ψ = exp_Hxx_even * ψ
        ψ = exp_Hxx_odd * ψ
        ψ = exp_Hxx_odd * ψ
        ψ = exp_Hxx_even * ψ
        ψ = exp_Hy * ψ
        ψ = exp_Hx * ψ
        return ψ
    end

    trott_expvals = Float64[]
    ψ = copy(ψ0)
    for step in 0:r
        push!(trott_expvals, real(dot(ψ, Z1 * ψ)))
        if step < r
            ψ = trotter_step(ψ)
        end
    end

    return ideal_expvals, trott_expvals
end

# =============================================================================
# Main execution
# =============================================================================
function main()
    println("="^70)
    println("PF2 Julia LPD Comparison - Correct Qiskit Convention")
    println("="^70)
    println("Parameters: n=$n, t=$t_total, r=$r, dt=$dt")
    println("w* = $w_threshold, hx=$hx, hy=$hy, Jx=$Jx")
    println("Initial state: |1010101010⟩ (Qiskit)")
    println("  → site 1 (qubit 0) = |0⟩, site 2 (qubit 1) = |1⟩, ...")
    println("Observable: Z on qubit 0 (site 1)")
    println("="^70)

    # Run all methods
    ideal_expvals, trott_expvals = run_exact_trotter()
    lpd_pp_expvals = run_lpd_pauliprop()
    lpd_ps_expvals = run_lpd_paulistrings()

    # Save results
    t_list = collect(0:dt:t_total)

    # Save to CSV
    results = hcat(t_list, ideal_expvals, trott_expvals, lpd_pp_expvals, lpd_ps_expvals)
    output_file = joinpath(@__DIR__, "julia_lpd_results.csv")
    writedlm(output_file, results, ',')
    println("\nResults saved to: $output_file")

    # Print comparison table
    println("\n" * "="^70)
    println("COMPARISON TABLE")
    println("="^70)
    println("     t |       Ideal |     Trotter |  LPD(PP.jl) |  LPD(PS.jl)")
    println("-"^70)
    for t_check in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        idx = Int(t_check / dt) + 1
        println("  $(round(t_check, digits=1)) | $(round(ideal_expvals[idx], digits=6)) | $(round(trott_expvals[idx], digits=6)) | $(round(lpd_pp_expvals[idx], digits=6)) | $(round(lpd_ps_expvals[idx], digits=6))")
    end
    println("-"^70)

    return t_list, ideal_expvals, trott_expvals, lpd_pp_expvals, lpd_ps_expvals
end

# Run
main()
