#=
Run all Julia LPD implementations and save results for comparison with Python.

This script runs:
1. PauliPropagation.jl (discrete gates)
2. PauliStrings.jl - Discrete (gate-based, matching Python)
3. PauliStrings.jl - Continuous (rk4)

And saves results to CSV files for plotting.
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
# 1. PauliPropagation.jl Implementation (Discrete)
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
    # Forward half: XX even bonds
    for i in 1:2:n-1
        push!(circuit, PauliRotation([:X, :X], [i, i+1], Jx * dt))
    end
    # Forward half: XX odd bonds
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
    onebitinds = collect(2:2:n_qubits)
    return overlapwithcomputational(psum, onebitinds)
end

function run_lpd_pauliprop()
    println("\n" * "="^60)
    println("Running PauliPropagation.jl LPD (Discrete)")
    println("="^60)

    circuit = build_pf2_circuit_pp(n, dt, hx, hy, Jx)

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
# 2. PauliStrings.jl - Common utilities
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
                if iseven(i)  # Even sites in |1⟩, Z gives -1
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
# 3. PauliStrings.jl - Discrete Evolution (matching Python)
# =============================================================================

# Check if two Pauli strings anticommute
function check_anticommute(p1::String, p2::String)
    cnt = 0
    for (c1, c2) in zip(p1, p2)
        if c1 != '1' && c2 != '1' && c1 != c2
            cnt += 1
        end
    end
    return isodd(cnt)
end

# Single Pauli multiplication table (PauliStrings uses '1' for identity)
function multiply_single_paulis(a::Char, b::Char)
    if a == '1'
        return b, 1.0
    elseif b == '1'
        return a, 1.0
    elseif a == b
        return '1', 1.0
    else
        if (a == 'X' && b == 'Y')
            return 'Z', 1im
        elseif (a == 'Y' && b == 'X')
            return 'Z', -1im
        elseif (a == 'Y' && b == 'Z')
            return 'X', 1im
        elseif (a == 'Z' && b == 'Y')
            return 'X', -1im
        elseif (a == 'Z' && b == 'X')
            return 'Y', 1im
        elseif (a == 'X' && b == 'Z')
            return 'Y', -1im
        end
    end
    error("Invalid Pauli: $a, $b")
end

function multiply_pauli_strings(p1::String, p2::String)
    result = Char[]
    phase = 1.0 + 0.0im
    for (c1, c2) in zip(p1, p2)
        r, ph = multiply_single_paulis(c1, c2)
        push!(result, r)
        phase *= ph
    end
    return String(result), phase
end

function apply_pauli_rotation_ps(O, G, θ::Float64)
    # Apply e^{iGθ} O e^{-iGθ}
    # For anticommuting: P → cos(2θ)P + i*sin(2θ)GP
    n_qubits = typeof(O.strings[1]).parameters[1]
    O_new = ps.Operator(n_qubits)

    c = cos(2θ)
    s = sin(2θ)

    G_coeffs, G_strings = ps.op_to_strings(G)
    g_str = G_strings[1]
    g_coeff = G_coeffs[1]

    O_coeffs, O_strings = ps.op_to_strings(O)

    for (p_coeff, p_str) in zip(O_coeffs, O_strings)
        anticommutes = check_anticommute(p_str, g_str)

        if anticommutes
            O_new += c * p_coeff, p_str
            gp_str, gp_phase = multiply_pauli_strings(g_str, p_str)
            O_new += 1im * s * p_coeff * g_coeff * gp_phase, gp_str
        else
            O_new += p_coeff, p_str
        end
    end

    return O_new
end

function pauli_weight_ps(str::String)
    return count(c -> c != '1', str)
end

function truncate_by_weight_ps(O, max_weight::Int)
    n_qubits = typeof(O.strings[1]).parameters[1]
    O_new = ps.Operator(n_qubits)
    O_coeffs, O_strings = ps.op_to_strings(O)

    for (coeff, str) in zip(O_coeffs, O_strings)
        if pauli_weight_ps(str) <= max_weight
            O_new += coeff, str
        end
    end

    return O_new
end

function build_trotter_gates_ps(n::Int, dt::Float64, hx::Float64, hy::Float64, Jx::Float64)
    gates = []

    # Forward: X terms
    for i in 1:n
        G = ps.Operator(n)
        G += "X", i
        push!(gates, (G, hx * dt / 2))
    end
    # Forward: Y terms
    for i in 1:n
        G = ps.Operator(n)
        G += "Y", i
        push!(gates, (G, hy * dt / 2))
    end
    # Forward: XX even bonds
    for i in 1:2:n-1
        G = ps.Operator(n)
        G += "X", i, "X", i+1
        push!(gates, (G, Jx * dt / 2))
    end
    # Forward: XX odd bonds
    for i in 2:2:n-1
        G = ps.Operator(n)
        G += "X", i, "X", i+1
        push!(gates, (G, Jx * dt / 2))
    end
    # Backward: XX odd bonds
    for i in 2:2:n-1
        G = ps.Operator(n)
        G += "X", i, "X", i+1
        push!(gates, (G, Jx * dt / 2))
    end
    # Backward: XX even bonds
    for i in 1:2:n-1
        G = ps.Operator(n)
        G += "X", i, "X", i+1
        push!(gates, (G, Jx * dt / 2))
    end
    # Backward: Y terms
    for i in 1:n
        G = ps.Operator(n)
        G += "Y", i
        push!(gates, (G, hy * dt / 2))
    end
    # Backward: X terms
    for i in 1:n
        G = ps.Operator(n)
        G += "X", i
        push!(gates, (G, hx * dt / 2))
    end

    return gates
end

function run_lpd_paulistrings_discrete()
    println("\n" * "="^60)
    println("Running PauliStrings.jl LPD (Discrete Gates)")
    println("="^60)

    gates = build_trotter_gates_ps(n, dt, hx, hy, Jx)
    println("Number of gates per Trotter step: $(length(gates))")

    O = ps.Operator(n)
    O += "Z", 1

    init_expval = expect_state_ps(O)
    println("Initial ⟨Z₁⟩ = $init_expval (expected: +1)")

    expvals = Float64[]
    push!(expvals, init_expval)

    for step in 1:r
        # Apply all gates
        for (G, θ) in gates
            O = apply_pauli_rotation_ps(O, G, θ)
        end

        # Truncate by weight
        O = truncate_by_weight_ps(O, w_threshold)
        O = ps.trim(ps.cutoff(O, 1e-15), 10^7)

        expval = expect_state_ps(O)
        push!(expvals, expval)

        # Print progress like Python
        coeffs, _ = ps.op_to_strings(O)
        println("  Step $step/$r: #Paulis = $(length(coeffs))")
    end

    return expvals
end

# =============================================================================
# 4. PauliStrings.jl - Continuous Evolution (rk4)
# =============================================================================
function run_lpd_paulistrings_continuous()
    println("\n" * "="^60)
    println("Running PauliStrings.jl LPD (Continuous rk4)")
    println("="^60)

    H = build_hamiltonian_ps(n, hx, hy, Jx)

    O = ps.Operator(n)
    O += "Z", 1

    init_expval = expect_state_ps(O)
    println("Initial ⟨Z₁⟩ = $init_expval (expected: +1)")

    expvals = Float64[]
    push!(expvals, init_expval)

    for step in 1:r
        # Runge-Kutta step in Heisenberg picture
        O = ps.rk4(H, O, dt; heisenberg=true)

        # Truncate by weight
        O = ps.truncate(O, w_threshold)
        O = ps.trim(ps.cutoff(O, 1e-15), 10^7)

        expval = expect_state_ps(O)
        push!(expvals, expval)

        # Print progress like Python
        coeffs, _ = ps.op_to_strings(O)
        println("  Step $step/$r: #Paulis = $(length(coeffs))")
    end

    return expvals
end

# =============================================================================
# 5. Exact and Trotter Evolution
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

    Z1 = single_qubit_op(σZ, 1, n)

    # Initial state: even sites in |1⟩
    pattern = [iseven(i) ? 1 : 0 for i in 1:n]
    idx = sum(pattern[i] * 2^(i-1) for i in 1:n)
    ψ0 = zeros(ComplexF64, dim)
    ψ0[idx + 1] = 1.0

    println("Initial ⟨Z₁⟩ = $(real(dot(ψ0, Z1 * ψ0))) (expected: +1)")

    # Ideal evolution
    println("Computing Ideal evolution...")
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
    println("PF2 Julia LPD Full Comparison")
    println("="^70)
    println("Parameters: n=$n, t=$t_total, r=$r, dt=$dt")
    println("w* = $w_threshold, hx=$hx, hy=$hy, Jx=$Jx")
    println("="^70)

    # Run all methods
    ideal_expvals, trott_expvals = run_exact_trotter()
    lpd_pp_expvals = run_lpd_pauliprop()
    lpd_ps_discrete_expvals = run_lpd_paulistrings_discrete()
    lpd_ps_continuous_expvals = run_lpd_paulistrings_continuous()

    # Save results
    t_list = collect(0:dt:t_total)

    results = hcat(t_list, ideal_expvals, trott_expvals,
                   lpd_pp_expvals, lpd_ps_discrete_expvals, lpd_ps_continuous_expvals)
    output_file = joinpath(@__DIR__, "julia_lpd_full_results.csv")
    writedlm(output_file, results, ',')
    println("\nResults saved to: $output_file")

    # Print comparison table
    println("\n" * "="^90)
    println("COMPARISON TABLE")
    println("="^90)
    println("     t |    Ideal |  Trotter | LPD(PP.jl) | LPD(PS Disc) | LPD(PS Cont)")
    println("-"^90)
    for t_check in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        idx = Int(t_check / dt) + 1
        println("  $(round(t_check, digits=1)) | $(round(ideal_expvals[idx], digits=5)) | $(round(trott_expvals[idx], digits=5)) | $(round(lpd_pp_expvals[idx], digits=5)) | $(round(lpd_ps_discrete_expvals[idx], digits=5)) | $(round(lpd_ps_continuous_expvals[idx], digits=5))")
    end
    println("-"^90)

    return t_list, ideal_expvals, trott_expvals, lpd_pp_expvals, lpd_ps_discrete_expvals, lpd_ps_continuous_expvals
end

main()
