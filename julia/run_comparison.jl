#=
Run Julia LPD implementations with progress bars.

Compares:
1. PauliPropagation.jl (discrete gates) - fast, optimized
2. PauliStrings.jl (continuous rk4) - uses optimized internal functions
3. PauliStrings.jl (discrete gates via commutator) - uses optimized commutator()

The discrete PauliStrings.jl implementation uses the commutator formula:
  O' = cos(θ)O + (i·sin(θ)/2)[G, O]
which leverages the optimized commutator() function for efficiency.
=#

using PauliPropagation
using PauliStrings
using DelimitedFiles
using LinearAlgebra
using SparseArrays
using ProgressMeter
using Printf

const ps = PauliStrings

# Parameters
const n = 10
const t_total = 5.0
const r = 50
const dt = t_total / r
const hx = 0.8
const hy = 0.9
const Jx = 1.0
const w_threshold = 5

# =============================================================================
# PauliPropagation.jl (Discrete)
# =============================================================================
function build_circuit_pp(n, dt, hx, hy, Jx)
    circuit = Vector{Any}()
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

function expect_pp(psum, n)
    onebitinds = collect(2:2:n)  # even sites in |1⟩
    return overlapwithcomputational(psum, onebitinds)
end

function run_pauliprop()
    println("\n[PauliPropagation.jl - Discrete]")
    circuit = build_circuit_pp(n, dt, hx, hy, Jx)

    obs = PauliSum(n)
    add!(obs, :Z, 1, 1.0)

    expvals = Float64[]
    npauli_list = Int[]

    p = Progress(r+1; desc="PP.jl:  ", showspeed=true)

    for step in 0:r
        push!(expvals, expect_pp(obs, n))
        push!(npauli_list, length(obs))
        next!(p; showvalues=[(:step, step), (:nPaulis, length(obs))])

        if step < r
            obs = propagate(circuit, obs; max_weight=w_threshold)
        end
    end

    return expvals, npauli_list
end

# =============================================================================
# PauliStrings.jl (Continuous rk4)
# =============================================================================
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

function run_paulistrings_rk4()
    println("\n[PauliStrings.jl - Continuous rk4]")
    H = build_hamiltonian_ps(n, hx, hy, Jx)

    O = ps.Operator(n)
    O += "Z", 1

    expvals = Float64[]
    npauli_list = Int[]

    push!(expvals, expect_ps(O))
    coeffs, _ = ps.op_to_strings(O)
    push!(npauli_list, length(coeffs))

    p = Progress(r; desc="PS.jl:  ", showspeed=true)

    for step in 1:r
        O = ps.rk4(H, O, dt; heisenberg=true)
        O = ps.truncate(O, w_threshold)
        O = ps.trim(ps.cutoff(O, 1e-15), 10^7)

        push!(expvals, expect_ps(O))
        coeffs, _ = ps.op_to_strings(O)
        push!(npauli_list, length(coeffs))

        next!(p; showvalues=[(:step, step), (:nPaulis, length(coeffs))])
    end

    return expvals, npauli_list
end

# =============================================================================
# PauliStrings.jl (Discrete Gates via Commutator)
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

function build_trotter_gates_ps(n, dt, hx, hy, Jx)
    gates = Tuple{ps.Operator, Float64}[]

    # Forward: X, Y, XX_even, XX_odd
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
    # Backward: XX_odd, XX_even, Y, X (symmetric for 2nd order Trotter)
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

function run_paulistrings_discrete()
    println("\n[PauliStrings.jl - Discrete Gates via Commutator]")

    gates = build_trotter_gates_ps(n, dt, hx, hy, Jx)
    ngates = length(gates)
    println("Number of gates per Trotter step: $ngates")

    O = ps.Operator(n)
    O += "Z", 1

    expvals = Float64[]
    npauli_list = Int[]

    push!(expvals, expect_ps(O))
    coeffs, _ = ps.op_to_strings(O)
    push!(npauli_list, length(coeffs))

    p = Progress(r; desc="PS.jl discrete: ", showspeed=true)

    for step in 1:r
        # Apply all gates in one Trotter step
        for (G, θ) in gates
            O = apply_pauli_rotation(O, G, θ)
        end

        # Truncate after each Trotter step
        O = ps.truncate(O, w_threshold)
        O = ps.trim(ps.cutoff(O, 1e-15), 10^7)

        push!(expvals, expect_ps(O))
        coeffs, _ = ps.op_to_strings(O)
        push!(npauli_list, length(coeffs))

        next!(p; showvalues=[(:step, step), (:nPaulis, length(coeffs))])
    end

    return expvals, npauli_list
end

# =============================================================================
# Exact and Trotter
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

function single_op(op, q, nq)
    ops = [σI for _ in 1:nq]
    ops[nq - q + 1] = op
    return kron_n(ops)
end

function two_op(op1, q1, op2, q2, nq)
    ops = [σI for _ in 1:nq]
    ops[nq - q1 + 1] = op1
    ops[nq - q2 + 1] = op2
    return kron_n(ops)
end

function run_exact_trotter()
    println("\n[Exact & Trotter Evolution]")
    dim = 2^n

    H_x = spzeros(ComplexF64, dim, dim)
    H_y = spzeros(ComplexF64, dim, dim)
    H_xx_even = spzeros(ComplexF64, dim, dim)
    H_xx_odd = spzeros(ComplexF64, dim, dim)

    for i in 1:n
        H_x += hx * single_op(σX, i, n)
        H_y += hy * single_op(σY, i, n)
    end
    for i in 1:2:n-1
        H_xx_even += Jx * two_op(σX, i, σX, i+1, n)
    end
    for i in 2:2:n-1
        H_xx_odd += Jx * two_op(σX, i, σX, i+1, n)
    end

    H_dense = Matrix(H_x + H_y + H_xx_even + H_xx_odd)
    Z1 = single_op(σZ, 1, n)

    pattern = [iseven(i) ? 1 : 0 for i in 1:n]
    idx = sum(pattern[i] * 2^(i-1) for i in 1:n)
    ψ0 = zeros(ComplexF64, dim)
    ψ0[idx + 1] = 1.0

    # Ideal
    println("  Computing Ideal...")
    ideal = Float64[]
    p = Progress(r+1; desc="Ideal:  ", showspeed=true)
    for i in 0:r
        U = exp(-im * H_dense * i * dt)
        ψt = U * ψ0
        push!(ideal, real(dot(ψt, Z1 * ψt)))
        next!(p)
    end

    # Trotter
    println("  Computing Trotter...")
    exp_Hx = exp(-im * (dt/2) * Matrix(H_x))
    exp_Hy = exp(-im * (dt/2) * Matrix(H_y))
    exp_Hxx_even = exp(-im * (dt/2) * Matrix(H_xx_even))
    exp_Hxx_odd = exp(-im * (dt/2) * Matrix(H_xx_odd))

    trotter_step(ψ) = exp_Hx * exp_Hy * exp_Hxx_even * exp_Hxx_odd * exp_Hxx_odd * exp_Hxx_even * exp_Hy * exp_Hx * ψ

    trott = Float64[]
    ψ = copy(ψ0)
    p = Progress(r+1; desc="Trott:  ", showspeed=true)
    for step in 0:r
        push!(trott, real(dot(ψ, Z1 * ψ)))
        next!(p)
        if step < r
            ψ = trotter_step(ψ)
        end
    end

    return ideal, trott
end

# =============================================================================
# Main
# =============================================================================
function main()
    println("="^70)
    println("PF2 Julia LPD Comparison")
    println("="^70)
    println("n=$n, t=$t_total, r=$r, dt=$dt, w*=$w_threshold")
    println("="^70)

    ideal, trott = run_exact_trotter()
    pp_expvals, pp_npauli = run_pauliprop()
    ps_rk4_expvals, ps_rk4_npauli = run_paulistrings_rk4()
    ps_disc_expvals, ps_disc_npauli = run_paulistrings_discrete()

    # Save results (6 columns now)
    t_list = collect(0:dt:t_total)
    results = hcat(t_list, ideal, trott, pp_expvals, ps_rk4_expvals, ps_disc_expvals)
    output_file = joinpath(@__DIR__, "julia_lpd_full_results.csv")
    writedlm(output_file, results, ',')
    println("\n\nResults saved to: $output_file")

    # Print comparison
    println("\n" * "="^95)
    println("COMPARISON TABLE")
    println("="^95)
    println("     t |    Ideal |  Trotter | LPD(PP.jl) | LPD(PS.jl rk4) | LPD(PS.jl disc)")
    println("-"^95)
    for t_check in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        idx = Int(t_check / dt) + 1
        @printf("  %3.1f | %8.5f | %8.5f | %10.5f | %14.5f | %14.5f\n",
                t_check, ideal[idx], trott[idx], pp_expvals[idx], ps_rk4_expvals[idx], ps_disc_expvals[idx])
    end
    println("-"^95)

    # Max errors
    println("\nMax differences:")
    @printf("  |PP.jl - Trotter|:         %.6f\n", maximum(abs.(pp_expvals .- trott)))
    @printf("  |PS.jl rk4 - Trotter|:     %.6f\n", maximum(abs.(ps_rk4_expvals .- trott)))
    @printf("  |PS.jl disc - Trotter|:    %.6f\n", maximum(abs.(ps_disc_expvals .- trott)))
    @printf("  |PP.jl - PS.jl rk4|:       %.6f\n", maximum(abs.(pp_expvals .- ps_rk4_expvals)))
    @printf("  |PP.jl - PS.jl disc|:      %.6f\n", maximum(abs.(pp_expvals .- ps_disc_expvals)))
    @printf("  |PS.jl rk4 - PS.jl disc|:  %.6f\n", maximum(abs.(ps_rk4_expvals .- ps_disc_expvals)))
end

main()
