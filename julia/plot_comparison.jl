# Plot comparison matching Python PF2.ipynb results
# The key is to match the Qiskit qubit convention exactly

using LinearAlgebra
using SparseArrays
using PauliStrings
using PauliPropagation
using Plots
using Printf

const ps = PauliStrings

# Parameters (matching Python)
const n = 10
const t_total = 5.0
const r = 50
const dt = t_total / r
const hx = 0.8
const hy = 0.9
const Jx = 1.0
const w_threshold = 5

# Qiskit convention for '1010101010':
# - Rightmost character is qubit 0: '0' → |0⟩
# - Next is qubit 1: '1' → |1⟩
# - etc.
# So: qubit 0 = |0⟩, qubit 1 = |1⟩, qubit 2 = |0⟩, ...
# In Julia 1-indexing (site i = qubit i-1):
# site 1 (qubit 0) = |0⟩, site 2 (qubit 1) = |1⟩, site 3 (qubit 2) = |0⟩, ...
# Pattern: odd sites are |0⟩, even sites are |1⟩
const initial_state_pattern = [isodd(i) ? 0 : 1 for i in 1:n]  # [0,1,0,1,0,1,0,1,0,1]

function run_all_methods()
    println("="^70)
    println("PF2 Comparison - Matching Python Convention")
    println("="^70)
    println("n=$n, t=$t_total, r=$r, dt=$dt, w*=$w_threshold")
    println("Initial state: |1010101010⟩ (Qiskit convention)")
    println("  → site 1 (qubit 0) = |0⟩, site 2 (qubit 1) = |1⟩, ...")
    println("Observable: Z on qubit 0 (site 1)")
    println("="^70)

    # Pauli matrices
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

    # ==========================================================================
    # 1. EXACT EVOLUTION
    # ==========================================================================
    println("\n1. Computing Exact Evolution...")
    H_terms = build_hamiltonian_terms(n)
    H_full = H_terms[:H_x] + H_terms[:H_y] + H_terms[:H_xx_even] + H_terms[:H_xx_odd]
    H_dense = Matrix(H_full)
    Z1 = single_qubit_op(σZ, 1, n)  # Z on site 1 = qubit 0
    ψ0 = initial_state_vector(initial_state_pattern, n)
    
    # Dense time points for smooth curve
    t_dense = collect(0:0.1:t_total)
    ideal_expvals = Float64[]
    for t in t_dense
        U = exp(-im * H_dense * t)
        ψt = U * ψ0
        push!(ideal_expvals, real(dot(ψt, Z1 * ψt)))
    end
    println("   Initial ⟨Z₁⟩ = $(ideal_expvals[1]) (expected: +1)")

    # ==========================================================================
    # 2. TROTTER EVOLUTION
    # ==========================================================================
    println("2. Computing Trotter Evolution...")
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
    println("   Initial ⟨Z₁⟩ = $(trott_expvals[1])")

    # ==========================================================================
    # 3. LPD with PauliPropagation.jl
    # ==========================================================================
    println("3. Computing LPD (PauliPropagation.jl)...")
    
    function build_circuit(n, dt, hx, hy, Jx)
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

    circuit = build_circuit(n, dt, hx, hy, Jx)
    # Sites with |1⟩: even sites (2,4,6,8,10)
    onebitinds = collect(2:2:n)
    
    O_pp = PauliSum(n)
    add!(O_pp, :Z, 1, 1.0)  # Z on site 1

    lpd_expvals = Float64[]
    obs = deepcopy(O_pp)
    push!(lpd_expvals, overlapwithcomputational(obs, onebitinds))

    for step in 1:r
        obs = propagate(circuit, obs; max_weight=w_threshold)
        push!(lpd_expvals, overlapwithcomputational(obs, onebitinds))
    end
    println("   Initial ⟨Z₁⟩ = $(lpd_expvals[1])")

    # ==========================================================================
    # PLOT
    # ==========================================================================
    println("\n4. Creating plot...")
    
    t_trott = collect(0:dt:t_total)
    
    # Plot style matching Python
    gr()
    p = plot(size=(500, 400), dpi=150,
             xlabel="Evolution time t",
             ylabel="⟨Z₁⟩",
             legend=:topright,
             legendfontsize=10,
             tickfontsize=10,
             guidefontsize=12,
             margin=5Plots.mm)
    
    # Ideal (black solid line)
    plot!(p, t_dense, ideal_expvals,
          label="Ideal",
          color=:black,
          linewidth=2,
          linestyle=:solid)
    
    # Trotter (magenta dash-dot with markers)
    plot!(p, t_trott, trott_expvals,
          label="Trotter (r=$r)",
          color=:magenta,
          linewidth=2,
          linestyle=:dashdot,
          marker=:circle,
          markersize=4,
          markerstrokewidth=0)
    
    # LPD (green dashed)
    plot!(p, t_trott, lpd_expvals,
          label="LPD (w*=$w_threshold)",
          color=RGB(0.0, 0.5, 0.5),  # teal color
          linewidth=2,
          linestyle=:dash)
    
    # Save
    savefig(p, "pf2_julia_comparison.png")
    savefig(p, "pf2_julia_comparison.pdf")
    println("   Saved to pf2_julia_comparison.png/pdf")

    # Print comparison table
    println("\n" * "="^70)
    println("COMPARISON TABLE")
    println("="^70)
    @printf("  %-6s | %-10s | %-10s | %-10s\n", "t", "Ideal", "Trotter", "LPD")
    println("-"^50)
    for t_check in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        idx = Int(t_check / dt) + 1
        @printf("  %-6.1f | %-10.6f | %-10.6f | %-10.6f\n",
                t_check, ideal_expvals[idx], trott_expvals[idx], lpd_expvals[idx])
    end
    println("-"^50)

    return p
end

run_all_methods()
