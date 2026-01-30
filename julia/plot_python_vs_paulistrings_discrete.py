"""
Comparison plot: Python LPD vs PauliStrings.jl (Discrete Gates via Commutator)

Both implementations use discrete Trotter gates, so results should match closely.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os

plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['font.family'] = 'serif'


def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Python LPD results
    with open(os.path.join(script_dir, 'pf2_python_results.json'), 'r') as f:
        py_data = json.load(f)

    # PauliStrings.jl discrete results
    ps_json_path = os.path.join(script_dir, 'paulistrings_discrete_results.json')
    ps_csv_path = os.path.join(script_dir, 'paulistrings_discrete_results.csv')

    if os.path.exists(ps_json_path):
        with open(ps_json_path, 'r') as f:
            ps_data = json.load(f)
        ps_t = np.array(ps_data['t_list'])
        ps_expvals = np.array(ps_data['expvals'])
        ps_npauli = np.array(ps_data['npauli_list'])
    elif os.path.exists(ps_csv_path):
        data = np.loadtxt(ps_csv_path, delimiter=',')
        ps_t = data[:, 0]
        ps_expvals = data[:, 1]
        ps_npauli = data[:, 2].astype(int)
    else:
        raise FileNotFoundError(
            "PauliStrings.jl discrete results not found!\n"
            "Run: julia run_paulistrings_discrete.jl"
        )

    return {
        'py': py_data,
        'ps_t': ps_t,
        'ps_expvals': ps_expvals,
        'ps_npauli': ps_npauli
    }


def plot_comparison(data, output_file):
    """Main comparison figure: Python LPD vs PauliStrings.jl discrete"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    py = data['py']
    t_dense = np.array(py['t_dense_list'])
    t_list = np.array(py['t_list'])

    # =========================================================================
    # Left panel: Expectation values
    # =========================================================================
    ax = axes[0]

    # Ideal (black solid)
    ax.plot(t_dense, py['ideal_expvals'], '-', color='black', lw=2, label='Ideal')

    # Trotter (gray dashed)
    ax.plot(t_list, py['trott_expvals'], '--', color='gray', lw=1.5,
            alpha=0.7, label='Trotter (r=50)')

    # LPD Python (green)
    ax.plot(t_list, py['lpd_expvals'], '-', color='forestgreen', lw=2.5,
            marker='o', markersize=4, markevery=5, label='LPD (Python)')

    # LPD PauliStrings.jl discrete (orange)
    ax.plot(data['ps_t'], data['ps_expvals'], '--', color='darkorange', lw=2.5,
            marker='s', markersize=4, markevery=5, label='LPD (PauliStrings.jl discrete)')

    ax.set_xlabel(r'Evolution time $t$')
    ax.set_ylabel(r'$\langle Z_1 \rangle$')
    ax.legend(loc='upper right', framealpha=0.95)
    ax.set_xlim(0, 5)
    ax.grid(True, alpha=0.3)
    ax.set_title('Expectation Value Comparison')

    # =========================================================================
    # Right panel: Difference between Python and PauliStrings.jl
    # =========================================================================
    ax = axes[1]

    py_lpd = np.array(py['lpd_expvals'])
    ps_lpd = data['ps_expvals']

    # Ensure same length
    min_len = min(len(py_lpd), len(ps_lpd))
    diff = py_lpd[:min_len] - ps_lpd[:min_len]
    t_diff = t_list[:min_len]

    ax.plot(t_diff, diff, '-', color='crimson', lw=2)
    ax.axhline(0, color='black', lw=0.5, ls='--')

    ax.set_xlabel(r'Evolution time $t$')
    ax.set_ylabel(r'$\Delta\langle Z_1 \rangle$ (Python - Julia)')
    ax.set_xlim(0, 5)
    ax.grid(True, alpha=0.3)
    ax.set_title('Difference: Python LPD - PauliStrings.jl discrete')

    # Add statistics text
    max_diff = np.max(np.abs(diff))
    mean_diff = np.mean(np.abs(diff))
    ax.text(0.95, 0.95, f'Max |diff|: {max_diff:.2e}\nMean |diff|: {mean_diff:.2e}',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    fig.savefig(output_file, bbox_inches='tight', dpi=150)
    print(f"Saved: {output_file}")
    plt.close()


def plot_single_comparison(data, output_file):
    """Single panel comparison figure"""
    fig, ax = plt.subplots(figsize=(8, 6))

    py = data['py']
    t_dense = np.array(py['t_dense_list'])
    t_list = np.array(py['t_list'])

    # Ideal (black solid)
    ax.plot(t_dense, py['ideal_expvals'], '-', color='black', lw=2, label='Ideal')

    # Trotter (magenta dash-dot)
    ax.plot(t_list, py['trott_expvals'], '-.', color='magenta', lw=1.5,
            marker='o', markersize=3, markevery=5, alpha=0.8, label='Trotter (r=50)')

    # LPD Python (green dashed)
    ax.plot(t_list, py['lpd_expvals'], '--', color='forestgreen', lw=2.5,
            label='LPD (Python, discrete)')

    # LPD PauliStrings.jl discrete (orange dotted)
    ax.plot(data['ps_t'], data['ps_expvals'], ':', color='darkorange', lw=3,
            label='LPD (PauliStrings.jl, discrete)')

    ax.set_xlabel(r'Evolution time $t$')
    ax.set_ylabel(r'$\langle Z_1 \rangle$')
    ax.legend(loc='upper right', framealpha=0.95)
    ax.set_xlim(0, 5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_file, bbox_inches='tight', dpi=150)
    print(f"Saved: {output_file}")
    plt.close()


def print_comparison_table(data):
    """Print detailed comparison table"""
    py = data['py']
    dt = 0.1

    print("\n" + "=" * 90)
    print("COMPARISON: Python LPD vs PauliStrings.jl (Discrete)")
    print("=" * 90)
    print(f"{'t':>6} | {'Ideal':>10} | {'Trotter':>10} | {'Python LPD':>12} | {'PS.jl disc':>12} | {'Diff':>12}")
    print("-" * 90)

    py_lpd = np.array(py['lpd_expvals'])
    ps_lpd = data['ps_expvals']

    for t_check in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]:
        idx = int(t_check / dt)
        diff = py_lpd[idx] - ps_lpd[idx]
        print(f"{t_check:>6.1f} | {py['ideal_expvals'][idx]:>10.6f} | {py['trott_expvals'][idx]:>10.6f} | "
              f"{py_lpd[idx]:>12.6f} | {ps_lpd[idx]:>12.6f} | {diff:>12.2e}")

    print("-" * 90)

    # Statistics
    min_len = min(len(py_lpd), len(ps_lpd))
    diff = py_lpd[:min_len] - ps_lpd[:min_len]

    print(f"\nStatistics:")
    print(f"  Max |Python - PS.jl discrete|: {np.max(np.abs(diff)):.6e}")
    print(f"  Mean |Python - PS.jl discrete|: {np.mean(np.abs(diff)):.6e}")
    print(f"  RMS difference: {np.sqrt(np.mean(diff**2)):.6e}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    print("Loading data...")
    try:
        data = load_data()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return

    print_comparison_table(data)

    print("\nCreating plots...")
    plot_comparison(data, os.path.join(script_dir, 'comparison_python_vs_paulistrings_discrete.pdf'))
    plot_single_comparison(data, os.path.join(script_dir, 'comparison_python_vs_paulistrings_discrete_single.pdf'))

    print("\nDone!")


if __name__ == "__main__":
    main()
