#=
PF2 Plotting Script

This script loads results from all methods and creates comparison plots
matching the style of Figure a in PF2.ipynb.

Run this after running:
- pf2_exact.jl (ideal + Trotter)
- pf2_lpd_paulistrings.jl (LPD continuous)
- pf2_lpd_pauliprop.jl (LPD discrete)
=#

using Plots
using JLD2
using DelimitedFiles
using Printf

# Parameters (for reference)
const n = 10
const t_total = 5.0
const r = 50
const dt = t_total / r
const w_threshold = 5

# =============================================================================
# Load results from different methods
# =============================================================================
function load_all_results()
    results = Dict()
    base_dir = @__DIR__

    # Load exact + Trotter results
    exact_file = joinpath(base_dir, "pf2_exact_results.jld2")
    if isfile(exact_file)
        @load exact_file results as exact_results
        results["exact"] = exact_results
        println("Loaded exact results from: $exact_file")
    else
        println("Warning: $exact_file not found")
    end

    # Load PauliStrings.jl LPD results (CSV format)
    ps_file = joinpath(base_dir, "pf2_lpd_paulistrings_results.csv")
    if isfile(ps_file)
        data = readdlm(ps_file, ',')
        results["paulistrings"] = Dict(
            "t_list" => data[:, 1],
            "discrete_expvals" => data[:, 2],
            "continuous_expvals" => data[:, 3]
        )
        println("Loaded PauliStrings results from: $ps_file")
    else
        println("Warning: $ps_file not found")
    end

    # Load PauliPropagation.jl LPD results
    pp_file = joinpath(base_dir, "pf2_lpd_pauliprop_results.jld2")
    if isfile(pp_file)
        @load pp_file results as pp_results
        results["pauliprop"] = pp_results
        println("Loaded PauliPropagation results from: $pp_file")
    else
        println("Warning: $pp_file not found")
    end

    return results
end

# =============================================================================
# Create comparison plot (matching PF2.ipynb Figure a style)
# =============================================================================
function create_comparison_plot(results)
    # Set up plot with nice styling
    gr()  # Use GR backend
    default(fontfamily="serif", legendfontsize=10, guidefontsize=12, tickfontsize=10)

    p = plot(size=(600, 450), margin=5Plots.mm)

    # Colors similar to Python notebook
    color_ideal = :black
    color_trotter = :magenta
    color_lpd = RGB(0.267, 0.467, 0.267)  # viridis-like green

    # Plot Ideal evolution
    if haskey(results, "exact")
        t_dense = results["exact"]["t_dense_list"]
        ideal = results["exact"]["ideal_expvals"]
        plot!(p, t_dense, ideal,
              label="Ideal",
              color=color_ideal,
              linewidth=2,
              linestyle=:solid)
    end

    # Plot Trotter evolution
    if haskey(results, "exact")
        t_list = results["exact"]["t_list"]
        trott = results["exact"]["trott_expvals"]
        plot!(p, t_list, trott,
              label="Trotter (r=$r)",
              color=color_trotter,
              linewidth=2,
              linestyle=:dashdot,
              marker=:circle,
              markersize=3)
    end

    # Plot LPD from PauliStrings.jl (discrete)
    if haskey(results, "paulistrings")
        t_ps = results["paulistrings"]["t_list"]
        lpd_discrete = results["paulistrings"]["discrete_expvals"]
        plot!(p, t_ps, lpd_discrete,
              label="LPD PauliStrings (w*=$w_threshold)",
              color=color_lpd,
              linewidth=2,
              linestyle=:dash)
    end

    # Plot LPD from PauliPropagation.jl
    if haskey(results, "pauliprop")
        t_pp = results["pauliprop"]["t_list"]
        lpd_pp = results["pauliprop"]["lpd_pauliprop_expvals"]
        plot!(p, t_pp, lpd_pp,
              label="LPD PauliProp (w*=$w_threshold)",
              color=:blue,
              linewidth=2,
              linestyle=:dot)
    end

    xlabel!(p, "Evolution time t")
    ylabel!(p, "⟨Z₁⟩")
    title!(p, "MFI Model: n=$n, hx=0.8, hy=0.9, Jx=1.0")

    return p
end

# =============================================================================
# Print comparison table
# =============================================================================
function print_comparison_table(results)
    println("\n" * "="^80)
    println("COMPARISON TABLE: Expectation values ⟨Z₁⟩ at selected times")
    println("="^80)

    # Prepare data
    t_check = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    # Header
    println(@sprintf("  %-6s | %-10s | %-10s | %-12s | %-12s",
                     "t", "Ideal", "Trotter", "LPD(PS)", "LPD(PP)"))
    println("-"^80)

    for t in t_check
        idx = Int(t / dt) + 1

        ideal_val = haskey(results, "exact") ? results["exact"]["ideal_expvals"][idx] : NaN
        trott_val = haskey(results, "exact") ? results["exact"]["trott_expvals"][idx] : NaN
        ps_val = haskey(results, "paulistrings") ? results["paulistrings"]["discrete_expvals"][idx] : NaN
        pp_val = haskey(results, "pauliprop") ? results["pauliprop"]["lpd_pauliprop_expvals"][idx] : NaN

        println(@sprintf("  %-6.1f | %-10.6f | %-10.6f | %-12.6f | %-12.6f",
                        t, ideal_val, trott_val, ps_val, pp_val))
    end
    println("-"^80)

    # Compute errors
    if haskey(results, "exact")
        trott_err = maximum(abs.(results["exact"]["trott_expvals"] .-
                                 results["exact"]["ideal_expvals"][1:length(results["exact"]["trott_expvals"])]))
        println(@sprintf("\nMax |Trotter - Ideal| = %.6f", trott_err))
    end

    if haskey(results, "paulistrings") && haskey(results, "exact")
        ps_err = maximum(abs.(results["paulistrings"]["discrete_expvals"] .- results["exact"]["trott_expvals"]))
        println(@sprintf("Max |LPD(PS) - Trotter| = %.6f", ps_err))
    end

    if haskey(results, "pauliprop") && haskey(results, "exact")
        pp_err = maximum(abs.(results["pauliprop"]["lpd_pauliprop_expvals"] .- results["exact"]["trott_expvals"]))
        println(@sprintf("Max |LPD(PP) - Trotter| = %.6f", pp_err))
    end
end

# =============================================================================
# Save combined results to CSV
# =============================================================================
function save_combined_csv(results)
    t_list = collect(0:dt:t_total)
    n_points = length(t_list)

    # Initialize with NaN
    data = fill(NaN, n_points, 5)
    data[:, 1] = t_list

    if haskey(results, "exact")
        # Ideal may have different time resolution
        ideal_t = results["exact"]["t_dense_list"]
        ideal_v = results["exact"]["ideal_expvals"]
        # Interpolate to match t_list
        for (i, t) in enumerate(t_list)
            idx = findfirst(x -> x ≈ t, ideal_t)
            if !isnothing(idx)
                data[i, 2] = ideal_v[idx]
            end
        end
        data[:, 3] = results["exact"]["trott_expvals"]
    end

    if haskey(results, "paulistrings")
        data[:, 4] = results["paulistrings"]["discrete_expvals"]
    end

    if haskey(results, "pauliprop")
        data[:, 5] = results["pauliprop"]["lpd_pauliprop_expvals"]
    end

    output_file = joinpath(@__DIR__, "pf2_all_results.csv")
    open(output_file, "w") do io
        println(io, "time,ideal,trotter,lpd_paulistrings,lpd_pauliprop")
        writedlm(io, data, ',')
    end
    println("\nCombined results saved to: $output_file")
end

# =============================================================================
# Main execution
# =============================================================================
function main()
    println("="^60)
    println("PF2 Results Comparison and Plotting")
    println("="^60)

    # Load all results
    results = load_all_results()

    if isempty(results)
        println("\nNo results found! Run the simulation scripts first:")
        println("  julia pf2_exact.jl")
        println("  julia pf2_lpd_paulistrings.jl")
        println("  julia pf2_lpd_pauliprop.jl")
        return
    end

    # Print comparison table
    print_comparison_table(results)

    # Create plot
    println("\nGenerating comparison plot...")
    p = create_comparison_plot(results)

    # Save plot
    plot_file_pdf = joinpath(@__DIR__, "pf2_julia_comparison.pdf")
    plot_file_png = joinpath(@__DIR__, "pf2_julia_comparison.png")
    savefig(p, plot_file_pdf)
    savefig(p, plot_file_png)
    println("Plot saved to:")
    println("  $plot_file_pdf")
    println("  $plot_file_png")

    # Save combined CSV
    save_combined_csv(results)

    # Display plot
    display(p)

    println("\n" * "="^60)
    println("Done!")
    println("="^60)

    return results, p
end

# Run
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
