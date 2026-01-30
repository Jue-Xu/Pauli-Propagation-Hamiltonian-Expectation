#=
LPD Implementation using PauliStrings.jl

This script implements Low-weight Pauli Dynamics with two methods:
1. Continuous evolution (rk4 - Runge-Kutta 4th order)
2. Discrete evolution (Pauli rotation gates)

The discrete method uses the Pauli rotation formula:
  e^{iG dt/2} P e^{-iG dt/2} =
    - P                           if [P, G] = 0 (commute)
    - cos(dt)P + i*sin(dt)GP      if {P, G} = 0 (anticommute)

where GP = [G, P]/2 is a new Pauli string.
=#

using PauliStrings
const ps = PauliStrings

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
# Build Hamiltonian using PauliStrings.jl
# =============================================================================
function build_hamiltonian_ps(n::Int, hx::Float64, hy::Float64, Jx::Float64)
    H = ps.Operator(n)

    # X field terms: hx * Σ Xᵢ
    for i in 1:n
        H += hx, "X", i
    end

    # Y field terms: hy * Σ Yᵢ
    for i in 1:n
        H += hy, "Y", i
    end

    # XX coupling terms: Jx * Σ XᵢXⱼ (nearest neighbor, open BC)
    for i in 1:n-1
        H += Jx, "X", i, "X", i+1
    end

    return H
end

# =============================================================================
# Build operator sequence for second-order Trotter
# Returns list of (Pauli generator, coefficient) for each gate
# =============================================================================
function build_trotter_gate_sequence(n::Int, dt::Float64, hx::Float64, hy::Float64, Jx::Float64)
    # Second-order Trotter: [H_x, H_y, H_xx_even, H_xx_odd, H_xx_odd, H_xx_even, H_y, H_x]
    # Each term gets dt/2 coefficient
    gates = []

    # Forward: X terms
    for i in 1:n
        G = ps.Operator(n)
        G += "X", i
        push!(gates, (G, hx * dt / 2))  # angle = hx * dt/2
    end

    # Forward: Y terms
    for i in 1:n
        G = ps.Operator(n)
        G += "Y", i
        push!(gates, (G, hy * dt / 2))
    end

    # Forward: XX even bonds (1-2, 3-4, ...)
    for i in 1:2:n-1
        G = ps.Operator(n)
        G += "X", i, "X", i+1
        push!(gates, (G, Jx * dt / 2))
    end

    # Forward: XX odd bonds (2-3, 4-5, ...)
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

# =============================================================================
# Expectation value for |1010101010⟩ state
#
# For product state |1010101010⟩ (Qiskit convention, rightmost is qubit 0):
#   - Site 1 (qubit 0, rightmost) in |1⟩
#   - Site 2 (qubit 1) in |0⟩
#   - Site 3 (qubit 2) in |1⟩
#   - ... alternating
# So odd sites (1,3,5,7,9) are in |1⟩, even sites (2,4,6,8,10) in |0⟩
#
# Only Z-type Paulis (containing only I/1 and Z) contribute.
# Z on |0⟩ gives eigenvalue +1
# Z on |1⟩ gives eigenvalue -1
#
# Note: PauliStrings.jl uses '1' for identity, not 'I'
# =============================================================================
function expect_alternating_state(O)
    # Get number of qubits from the first PauliString type parameter
    n_qubits = typeof(O.strings[1]).parameters[1]
    total = 0.0 + 0.0im

    # Iterate through all Pauli strings and coefficients
    for (pauli_str, coeff) in zip(O.strings, O.coeffs)
        is_z_type = true
        sign = 1

        # Check each qubit - use string representation
        str = string(pauli_str)
        for (i, p) in enumerate(str)
            if p == 'X' || p == 'Y'
                is_z_type = false
                break
            elseif p == 'Z'
                # Odd sites (1,3,5,...) are in |1⟩, Z gives -1
                if isodd(i)
                    sign *= -1
                end
                # Even sites (2,4,6,...) are in |0⟩, Z gives +1
            end
            # '1' (identity) contributes nothing to the sign
        end

        if is_z_type
            total += coeff * sign
        end
    end

    return real(total)
end

# Alternative implementation using PauliStrings internal functions (op_to_strings)
function expect_alternating_state_v2(O)
    # Pattern: sites 1,3,5,7,9 in |1⟩ (odd), sites 2,4,6,8,10 in |0⟩ (even)
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
                if isodd(i)  # Site in |1⟩
                    sign *= -1
                end
            end
            # '1' (identity) contributes nothing
        end

        if is_z_type
            total += coeff * sign
        end
    end

    return real(total)
end

# =============================================================================
# Discrete Pauli rotation evolution
#
# Applies the transformation:
#   e^{iG θ} P e^{-iG θ} =
#     - P                   if [P, G] = 0
#     - cos(2θ)P + i*sin(2θ)GP   if {P, G} = 0
# =============================================================================
function apply_pauli_rotation(O::ps.Operator, G::ps.Operator, θ::Float64)
    # O is the observable (sum of Pauli strings)
    # G is the generator (single Pauli string)
    # θ is the rotation angle

    # For each Pauli string P in O:
    # If [P, G] = 0: P → P (unchanged)
    # If {P, G} = 0: P → cos(2θ)P + i*sin(2θ)GP

    c = cos(2θ)
    s = sin(2θ)

    # Commuting part stays the same, anticommuting part gets transformed
    # commutator([G, O])/2 gives the anticommuting part composed with G
    # Let A = anticommuting part of O with G
    # A → cos(2θ)A + i*sin(2θ)GA

    # Use PauliStrings commutator: [G, O] = GO - OG
    comm = ps.commutator(G, O)

    # O_new = O * cos(2θ) + comm * (i*sin(2θ)/2)
    # But this only works for the anticommuting part

    # More precise: split O into commuting and anticommuting parts
    # O = O_c + O_a where [G, O_c] = 0 and {G, O_a} = 0
    # Then: O_new = O_c + cos(2θ)O_a + i*sin(2θ)G*O_a

    # Since [G, O_c] = 0 and [G, O_a] = 2*G*O_a (anticommutator relation)
    # comm = [G, O] = [G, O_a] = 2*G*O_a
    # So G*O_a = comm/2

    # O_new = O_c + cos(2θ)O_a + i*sin(2θ)*comm/2
    #       = O_c + O_a + (cos(2θ)-1)*O_a + i*sin(2θ)*comm/2
    #       = O + (cos(2θ)-1)*O_a + i*sin(2θ)*comm/2

    # O_a = (comm / 2) * G^{-1} but G^{-1} = G for Paulis
    # Actually: O_a = comm / (2G) is tricky, let's use a different approach

    # Direct approach: iterate through strings
    # Get number of qubits from the parametric type
    n_qubits = typeof(O.strings[1]).parameters[1]
    O_new = ps.Operator(n_qubits)

    # Get the single Pauli string from G
    G_coeffs, G_strings = ps.op_to_strings(G)
    if length(G_strings) != 1
        error("Generator G must be a single Pauli string")
    end
    g_str = G_strings[1]
    g_coeff = G_coeffs[1]

    # Iterate through O
    O_coeffs, O_strings = ps.op_to_strings(O)

    for (p_coeff, p_str) in zip(O_coeffs, O_strings)
        # Check if P anticommutes with G
        anticommutes = check_anticommute(p_str, g_str)

        if anticommutes
            # P → cos(2θ)P + i*sin(2θ)GP
            # Add cos(2θ)P
            O_new += c * p_coeff, p_str

            # Add i*sin(2θ)GP
            gp_str, gp_phase = multiply_pauli_strings(g_str, p_str)
            O_new += 1im * s * p_coeff * g_coeff * gp_phase, gp_str
        else
            # P stays unchanged
            O_new += p_coeff, p_str
        end
    end

    return O_new
end

# Check if two Pauli strings anticommute
function check_anticommute(p1::String, p2::String)
    # Two Pauli strings anticommute if they have an odd number of
    # positions where both are non-identity and different
    cnt = 0
    for (c1, c2) in zip(p1, p2)
        # Check for non-identity: '1' is identity in PauliStrings.jl
        if c1 != '1' && c2 != '1' && c1 != c2
            # Both non-identity and different: they anticommute at this site
            # X-Y, X-Z, Y-Z anticommute; same letters commute
            cnt += 1
        end
    end
    return isodd(cnt)
end

# Multiply two Pauli strings, return (result_string, phase)
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

# Single Pauli multiplication table
# Note: PauliStrings.jl uses '1' for identity
function multiply_single_paulis(a::Char, b::Char)
    # Returns (result, phase) where result is '1', 'X', 'Y', or 'Z'
    # and phase is the multiplicative factor

    if a == '1'
        return b, 1.0
    elseif b == '1'
        return a, 1.0
    elseif a == b
        return '1', 1.0  # X*X = Y*Y = Z*Z = I
    else
        # Different non-identity Paulis
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

# =============================================================================
# Weight truncation
# =============================================================================
function pauli_weight(str::String)
    # In PauliStrings.jl, identity is '1'
    return count(c -> c != '1', str)
end

function truncate_by_weight(O, max_weight::Int)
    # Get number of qubits from the parametric type
    n_qubits = typeof(O.strings[1]).parameters[1]
    O_new = ps.Operator(n_qubits)

    O_coeffs, O_strings = ps.op_to_strings(O)

    for (coeff, str) in zip(O_coeffs, O_strings)
        if pauli_weight(str) <= max_weight
            O_new += coeff, str
        end
    end

    return O_new
end

# =============================================================================
# LPD with Discrete Pauli Rotations
# =============================================================================
function lpd_discrete_evolution(n::Int, gates, O_init::ps.Operator,
                                 nsteps::Int, max_weight::Int)
    expvals = Float64[]
    O = O_init

    # Initial expectation value
    push!(expvals, expect_alternating_state_v2(O))

    for step in 1:nsteps
        # Apply all gates in the Trotter step
        for (G, θ) in gates
            O = apply_pauli_rotation(O, G, θ)
        end

        # Truncate by weight
        O = truncate_by_weight(O, max_weight)

        # Simplify/clean up the operator
        O = ps.trim(ps.cutoff(O, 1e-15), 10^7)  # Remove tiny coefficients

        # Record expectation value
        push!(expvals, expect_alternating_state_v2(O))

        if step % 10 == 0
            coeffs, _ = ps.op_to_strings(O)
            println("Step $step: #Paulis = $(length(coeffs))")
        end
    end

    return expvals
end

# =============================================================================
# LPD with Continuous rk4 Evolution
# =============================================================================
function lpd_continuous_evolution(H::ps.Operator, O_init::ps.Operator,
                                   dt::Float64, nsteps::Int, max_weight::Int)
    expvals = Float64[]
    O = O_init

    # Initial expectation value
    push!(expvals, expect_alternating_state_v2(O))

    for step in 1:nsteps
        # Runge-Kutta step in Heisenberg picture
        O = ps.rk4(H, O, dt; heisenberg=true)

        # Truncate by weight using PauliStrings truncate function
        O = ps.truncate(O, max_weight)

        # Clean up small coefficients
        O = ps.trim(ps.cutoff(O, 1e-15), 10^7)

        # Record expectation value
        push!(expvals, expect_alternating_state_v2(O))

        if step % 10 == 0
            coeffs, _ = ps.op_to_strings(O)
            println("Step $step: #Paulis = $(length(coeffs))")
        end
    end

    return expvals
end

# =============================================================================
# Main execution
# =============================================================================
function run_lpd_paulistrings()
    println("="^60)
    println("PF2 LPD Evolution using PauliStrings.jl")
    println("="^60)
    println("Parameters: n=$n, t=$t_total, r=$r, dt=$dt")
    println("Weight threshold: w* = $w_threshold")
    println("="^60)

    println("\nBuilding Hamiltonian using PauliStrings.jl...")
    H = build_hamiltonian_ps(n, hx, hy, Jx)

    # Observable: Z on site 1 (matching Qiskit 'IIIIIIIIIZ')
    println("Building observable Z on site 1 (rightmost qubit)...")
    O_init = ps.Operator(n)
    O_init += "Z", 1

    # Verify initial expectation value
    init_expval = expect_alternating_state_v2(O_init)
    println("Initial ⟨Z₁⟩ = $init_expval (expected: -1)")

    # Run continuous evolution (rk4) - more efficient than discrete
    println("\n--- LPD Continuous Evolution (rk4) ---")
    @time continuous_expvals = lpd_continuous_evolution(H, deepcopy(O_init), dt, r, w_threshold)

    return continuous_expvals
end

# Run if this is the main script
if abspath(PROGRAM_FILE) == @__FILE__
    continuous_expvals = run_lpd_paulistrings()

    println("\n--- Results Summary ---")
    times = collect(0:dt:t_total)
    for t_check in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        idx = Int(t_check / dt) + 1
        println("  t=$t_check: ⟨Z₁⟩ = $(round(continuous_expvals[idx], digits=6))")
    end

    # Save results
    using DelimitedFiles
    results = hcat(times, continuous_expvals)
    writedlm("pf2_lpd_paulistrings_results.csv", results, ',')
    println("\nResults saved to pf2_lpd_paulistrings_results.csv")
end
