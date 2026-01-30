# Comparison of all methods for 8-qubit case

using LinearAlgebra
using SparseArrays
using PauliStrings
using PauliPropagation
using DelimitedFiles
using Printf

const ps = PauliStrings

# Parameters for 8 qubits
const n = 8
const t_total = 5.0
const r = 50
const dt = t_total / r
const hx = 0.8
const hy = 0.9
const Jx = 1.0
const w_threshold = 5

# Initial state pattern: odd sites in |1⟩, even sites in |0⟩
const initial_state_pattern = [isodd(i) ? 1 : 0 for i in 1:n]

function run_comparison()
    println("="^70)
    println("PF2 Comparison - 8 Qubits")
    println("="^70)
    println("Parameters: n=$n, t=$t_total, r=$r, dt=$dt")
    println("w* = $w_threshold, hx=$hx, hy=$hy, Jx=$Jx")
    println("Initial state: |10101010⟩ (alternating)")
    println("Observable: Z on site 1 (rightmost)")
    println("="^70)

    # =============================================================================
    # Pauli matrices and Hamiltonian construction for exact evolution
    # =============================================================================
    σI = sparse(ComplexF64[1 0; 0 1])
    σX = sparse(ComplexF64[0 1; 1 0])
    σY = sparse(ComplexF64[0 -im; im 0])
    σZ = sparse(ComplexF64[1 0; 0 -1])

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

    function build_hamiltonian_terms(n_qubits)
        dim = 2^n_qubits
        
        H_x = spzeros(ComplexF64, dim, dim)
        for i in 1:n_qubits
            H_x += hx * single_qubit_op(σX, i, n_qubits)
        end
        
        H_y = spzeros(ComplexF64, dim, dim)
        for i in 1:n_qubits
            H_y += hy * single_qubit_op(σY, i, n_qubits)
        end
        
        H_xx_even = spzeros(ComplexF64, dim, dim)
        for i in 1:2:(n_qubits-1)
            H_xx_even += Jx * two_qubit_op(σX, i, σX, i+1, n_qubits)
        end
        
        H_xx_odd = spzeros(ComplexF64, dim, dim)
        for i in 2:2:(n_qubits-1)
            H_xx_odd += Jx * two_qubit_op(σX, i, σX, i+1, n_qubits)
        end
        
        return Dict(:H_x => H_x, :H_y => H_y, :H_xx_even => H_xx_even, :H_xx_odd => H_xx_odd)
    end

    function initial_state_vector(pattern, n_qubits)
        idx = sum(pattern[i] * 2^(i-1) for i in 1:n_qubits)
        state = zeros(ComplexF64, 2^n_qubits)
        state[idx + 1] = 1.0
        return state
    end

    # =============================================================================
    # 1. Exact Evolution
    # =============================================================================
    println("\n" * "-"^70)
    println("1. EXACT (IDEAL) EVOLUTION")
    println("-"^70)

    H_terms = build_hamiltonian_terms(n)
    H_full = H_terms[:H_x] + H_terms[:H_y] + H_terms[:H_xx_even] + H_terms[:H_xx_odd]
    H_dense = Matrix(H_full)
    Z1 = single_qubit_op(σZ, 1, n)
    ψ0 = initial_state_vector(initial_state_pattern, n)

    ideal_expvals = Float64[]
    for i in 0:r
        t = i * dt
        U = exp(-im * H_dense * t)
        ψt = U * ψ0
        push!(ideal_expvals, real(dot(ψt, Z1 * ψt)))
    end
    println("  Computed $(length(ideal_expvals)) time points")

    # =============================================================================
    # 2. Trotter Evolution
    # =============================================================================
    println("\n" * "-"^70)
    println("2. TROTTER (PF2) EVOLUTION")
    println("-"^70)

    H_x_dense = Matrix(H_terms[:H_x])
    H_y_dense = Matrix(H_terms[:H_y])
    H_xx_even_dense = Matrix(H_terms[:H_xx_even])
    H_xx_odd_dense = Matrix(H_terms[:H_xx_odd])

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
    println("  Computed $(length(trott_expvals)) time points")

    # =============================================================================
    # 3. LPD with PauliStrings.jl (continuous rk4)
    # =============================================================================
    println("\n" * "-"^70)
    println("3. LPD (PauliStrings.jl - continuous rk4)")
    println("-"^70)

    function build_hamiltonian_ps(n, hx, hy, Jx)
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

    function expect_alternating_ps(O)
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

    H_ps = build_hamiltonian_ps(n, hx, hy, Jx)
    O_ps = ps.Operator(n)
    O_ps += "Z", 1

    lpd_ps_expvals = Float64[]
    O = deepcopy(O_ps)
    push!(lpd_ps_expvals, expect_alternating_ps(O))

    for step in 1:r
        O = ps.rk4(H_ps, O, dt; heisenberg=true)
        O = ps.truncate(O, w_threshold)
        O = ps.trim(ps.cutoff(O, 1e-15), 10^7)
        push!(lpd_ps_expvals, expect_alternating_ps(O))
    end
    println("  Computed $(length(lpd_ps_expvals)) time points")

    # =============================================================================
    # 4. LPD with PauliPropagation.jl (discrete gates)
    # =============================================================================
    println("\n" * "-"^70)
    println("4. LPD (PauliPropagation.jl - discrete gates)")
    println("-"^70)

    function build_pf2_circuit_pp(n, dt, hx, hy, Jx)
        circuit = []
        for i in 1:n
            push!(circuit, PauliRotation(:X, i, hx * dt))
        end
        for i in 1:n
            push!(circuit, PauliRotation(:Y, i, hy * dt))
        end
        for i in 1:2:n-1
            push!(circuit, PauliRotation([:X, :X], [i, i+1], Jx * dt))
        end
        for i in 2:2:n-1
            push!(circuit, PauliRotation([:X, :X], [i, i+1], Jx * dt))
        end
        for i in 2:2:n-1
            push!(circuit, PauliRotation([:X, :X], [i, i+1], Jx * dt))
        end
        for i in 1:2:n-1
            push!(circuit, PauliRotation([:X, :X], [i, i+1], Jx * dt))
        end
        for i in 1:n
            push!(circuit, PauliRotation(:Y, i, hy * dt))
        end
        for i in 1:n
            push!(circuit, PauliRotation(:X, i, hx * dt))
        end
        return circuit
    end

    circuit = build_pf2_circuit_pp(n, dt, hx, hy, Jx)
    onebitinds = collect(1:2:n)  # Sites with |1⟩

    O_pp = PauliSum(n)
    add!(O_pp, :Z, 1, 1.0)

    lpd_pp_expvals = Float64[]
    obs = deepcopy(O_pp)
    push!(lpd_pp_expvals, overlapwithcomputational(obs, onebitinds))

    for step in 1:r
        obs = propagate(circuit, obs; max_weight=w_threshold)
        push!(lpd_pp_expvals, overlapwithcomputational(obs, onebitinds))
    end
    println("  Computed $(length(lpd_pp_expvals)) time points")

    # =============================================================================
    # Comparison
    # =============================================================================
    println("\n" * "="^70)
    println("COMPARISON")
    println("="^70)

    println("\nExpectation values ⟨Z₁⟩ at selected times:")
    println("-"^70)
    @printf("  %-6s | %-10s | %-10s | %-12s | %-12s\n", "t", "Ideal", "Trotter", "LPD(PS)", "LPD(PP)")
    println("-"^70)

    for t_check in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        idx = Int(t_check / dt) + 1
        @printf("  %-6.1f | %-10.6f | %-10.6f | %-12.6f | %-12.6f\n",
                t_check, ideal_expvals[idx], trott_expvals[idx],
                lpd_ps_expvals[idx], lpd_pp_expvals[idx])
    end
    println("-"^70)

    # Errors
    trott_vs_ideal = maximum(abs.(trott_expvals .- ideal_expvals))
    lpd_ps_vs_trott = maximum(abs.(lpd_ps_expvals .- trott_expvals))
    lpd_pp_vs_trott = maximum(abs.(lpd_pp_expvals .- trott_expvals))
    lpd_ps_vs_pp = maximum(abs.(lpd_ps_expvals .- lpd_pp_expvals))

    println("\nMax errors:")
    @printf("  |Trotter - Ideal|_max     = %.6f\n", trott_vs_ideal)
    @printf("  |LPD(PS) - Trotter|_max   = %.6f\n", lpd_ps_vs_trott)
    @printf("  |LPD(PP) - Trotter|_max   = %.6f\n", lpd_pp_vs_trott)
    @printf("  |LPD(PS) - LPD(PP)|_max   = %.6f\n", lpd_ps_vs_pp)

    # Save results
    times = collect(0:dt:t_total)
    results = hcat(times, ideal_expvals, trott_expvals, lpd_ps_expvals, lpd_pp_expvals)
    writedlm("pf2_comparison_8qubits.csv", results, ',')
    println("\nResults saved to pf2_comparison_8qubits.csv")

    println("\n" * "="^70)
    println("Done!")
    println("="^70)
    
    return ideal_expvals, trott_expvals, lpd_ps_expvals, lpd_pp_expvals
end

run_comparison()
