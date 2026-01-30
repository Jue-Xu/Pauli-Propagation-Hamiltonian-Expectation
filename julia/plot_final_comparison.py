"""
Final comparison plots: Python LPD vs Julia LPD implementations.

Creates two PDF figures:
1. Python vs PauliPropagation.jl (discrete)
2. Python vs PauliStrings.jl (continuous rk4)
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

    # Python results
    with open(os.path.join(script_dir, 'pf2_python_results.json'), 'r') as f:
        py_data = json.load(f)

    # Julia results
    jl_data = np.loadtxt(os.path.join(script_dir, 'julia_lpd_full_results.csv'), delimiter=',')

    return {
        'py': py_data,
        'jl_t': jl_data[:, 0],
        'jl_ideal': jl_data[:, 1],
        'jl_trott': jl_data[:, 2],
        'jl_pp': jl_data[:, 3],
        'jl_ps': jl_data[:, 4]
    }

def plot_comparison_pauliprop(data, output_file):
    """Figure a: Python vs PauliPropagation.jl"""
    fig, ax = plt.subplots(figsize=(7, 5))

    py = data['py']
    t_dense = np.array(py['t_dense_list'])
    t_list = np.array(py['t_list'])

    # Ideal (black solid)
    ax.plot(t_dense, py['ideal_expvals'], '-', color='black', lw=2, label='Ideal')

    # Trotter (magenta dash-dot)
    ax.plot(t_list, py['trott_expvals'], '-.', color='magenta', lw=2,
            marker='o', markersize=4, markevery=5, label='Trotter (r=50)')

    # LPD Python (green dashed)
    ax.plot(t_list, py['lpd_expvals'], '--', color='forestgreen', lw=2.5,
            label='LPD (Python)')

    # LPD Julia PauliPropagation.jl (blue dotted)
    ax.plot(data['jl_t'], data['jl_pp'], ':', color='royalblue', lw=2.5,
            label='LPD (PauliPropagation.jl)')

    ax.set_xlabel(r'Evolution time $t$')
    ax.set_ylabel(r'$\langle Z_1 \rangle$')
    ax.legend(loc='upper right', framealpha=0.95)
    ax.set_xlim(0, 5)
    ax.grid(True, alpha=0.3)
    ax.text(-0.12, 1.05, 'a', transform=ax.transAxes, fontsize=20, fontweight='bold', va='top')

    plt.tight_layout()
    fig.savefig(output_file, bbox_inches='tight', dpi=150)
    print(f"Saved: {output_file}")
    plt.close()

def plot_comparison_paulistrings(data, output_file):
    """Figure b: Python vs PauliStrings.jl (rk4)"""
    fig, ax = plt.subplots(figsize=(7, 5))

    py = data['py']
    t_dense = np.array(py['t_dense_list'])
    t_list = np.array(py['t_list'])

    # Ideal (black solid)
    ax.plot(t_dense, py['ideal_expvals'], '-', color='black', lw=2, label='Ideal')

    # Trotter (magenta dash-dot)
    ax.plot(t_list, py['trott_expvals'], '-.', color='magenta', lw=2,
            marker='o', markersize=4, markevery=5, label='Trotter (r=50)')

    # LPD Python (green dashed)
    ax.plot(t_list, py['lpd_expvals'], '--', color='forestgreen', lw=2.5,
            label='LPD (Python, discrete)')

    # LPD Julia PauliStrings.jl rk4 (orange dotted)
    ax.plot(data['jl_t'], data['jl_ps'], ':', color='darkorange', lw=2.5,
            label='LPD (PauliStrings.jl, rk4)')

    ax.set_xlabel(r'Evolution time $t$')
    ax.set_ylabel(r'$\langle Z_1 \rangle$')
    ax.legend(loc='upper right', framealpha=0.95)
    ax.set_xlim(0, 5)
    ax.grid(True, alpha=0.3)
    ax.text(-0.12, 1.05, 'b', transform=ax.transAxes, fontsize=20, fontweight='bold', va='top')

    plt.tight_layout()
    fig.savefig(output_file, bbox_inches='tight', dpi=150)
    print(f"Saved: {output_file}")
    plt.close()

def print_comparison(data):
    py = data['py']
    dt = 0.1

    print("\n" + "="*100)
    print("COMPARISON TABLE: Python vs Julia LPD Implementations")
    print("="*100)
    print(f"{'t':>6} | {'Ideal':>10} | {'Trotter':>10} | {'LPD(Py)':>10} | {'LPD(PP.jl)':>12} | {'LPD(PS.jl)':>12}")
    print("-"*100)

    for t_check in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]:
        idx = int(t_check / dt)
        print(f"{t_check:>6.1f} | {py['ideal_expvals'][idx]:>10.6f} | {py['trott_expvals'][idx]:>10.6f} | "
              f"{py['lpd_expvals'][idx]:>10.6f} | {data['jl_pp'][idx]:>12.6f} | {data['jl_ps'][idx]:>12.6f}")

    print("-"*100)

    lpd_py = np.array(py['lpd_expvals'])
    lpd_pp = data['jl_pp']
    lpd_ps = data['jl_ps']

    print(f"\nMax |LPD(Python) - LPD(PauliPropagation.jl)|: {np.max(np.abs(lpd_py - lpd_pp)):.6f}")
    print(f"Max |LPD(Python) - LPD(PauliStrings.jl rk4)|: {np.max(np.abs(lpd_py - lpd_ps)):.6f}")
    print(f"Max |LPD(PauliPropagation.jl) - LPD(PauliStrings.jl)|: {np.max(np.abs(lpd_pp - lpd_ps)):.6f}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    print("Loading data...")
    data = load_data()

    print_comparison(data)

    print("\nCreating plots...")
    plot_comparison_pauliprop(data, os.path.join(script_dir, 'comparison_python_vs_paulipropagation.pdf'))
    plot_comparison_paulistrings(data, os.path.join(script_dir, 'comparison_python_vs_paulistrings.pdf'))

    print("\nDone!")

if __name__ == "__main__":
    main()
