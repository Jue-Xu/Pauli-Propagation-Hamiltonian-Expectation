"""
Plot comparison between Python LPD and Julia LPD implementations.

Creates two separate PDF figures:
1. Python vs PauliPropagation.jl
2. Python vs PauliStrings.jl

Each figure shows: Ideal, Trotter, LPD (Python), LPD (Julia)
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os

# Set up plotting style
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['font.family'] = 'serif'

def load_python_results(filepath):
    """Load Python results from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def load_julia_results(filepath):
    """Load Julia results from CSV file."""
    data = np.loadtxt(filepath, delimiter=',')
    return {
        't_list': data[:, 0],
        'ideal_expvals': data[:, 1],
        'trott_expvals': data[:, 2],
        'lpd_pp_expvals': data[:, 3],
        'lpd_ps_expvals': data[:, 4]
    }

def plot_comparison_pauliprop(python_data, julia_data, output_file):
    """
    Create comparison figure for PauliPropagation.jl.
    Shows: Ideal, Trotter, LPD (Python), LPD (PauliPropagation.jl)
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    t_list = np.array(python_data['t_list'])
    t_dense = np.array(python_data['t_dense_list'])

    # Ideal (black solid line)
    ax.plot(t_dense, python_data['ideal_expvals'],
            '-', color='black', linewidth=2, label='Ideal')

    # Trotter (magenta dash-dot with markers)
    ax.plot(t_list, python_data['trott_expvals'],
            '-.', color='magenta', linewidth=2, marker='o',
            markersize=4, markevery=5, label=f'Trotter (r=50)')

    # LPD Python (green dashed)
    ax.plot(t_list, python_data['lpd_expvals'],
            '--', color='forestgreen', linewidth=2.5, label='LPD (Python)')

    # LPD Julia PauliPropagation.jl (blue dotted)
    ax.plot(julia_data['t_list'], julia_data['lpd_pp_expvals'],
            ':', color='royalblue', linewidth=2.5, label='LPD (PauliPropagation.jl)')

    ax.set_xlabel(r'Evolution time $t$')
    ax.set_ylabel(r'$\langle Z_1 \rangle$')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_xlim(0, 5)
    ax.grid(True, alpha=0.3)

    # Add title
    ax.set_title('Python vs Julia (PauliPropagation.jl)')

    # Add panel label
    ax.text(-0.12, 1.05, 'a', transform=ax.transAxes, fontsize=20,
            fontweight='bold', va='top')

    plt.tight_layout()
    fig.savefig(output_file, bbox_inches='tight', dpi=150)
    print(f"Saved: {output_file}")
    plt.close()

def plot_comparison_paulistrings(python_data, julia_data, output_file):
    """
    Create comparison figure for PauliStrings.jl.
    Shows: Ideal, Trotter, LPD (Python), LPD (PauliStrings.jl)
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    t_list = np.array(python_data['t_list'])
    t_dense = np.array(python_data['t_dense_list'])

    # Ideal (black solid line)
    ax.plot(t_dense, python_data['ideal_expvals'],
            '-', color='black', linewidth=2, label='Ideal')

    # Trotter (magenta dash-dot with markers)
    ax.plot(t_list, python_data['trott_expvals'],
            '-.', color='magenta', linewidth=2, marker='o',
            markersize=4, markevery=5, label=f'Trotter (r=50)')

    # LPD Python (green dashed)
    ax.plot(t_list, python_data['lpd_expvals'],
            '--', color='forestgreen', linewidth=2.5, label='LPD (Python)')

    # LPD Julia PauliStrings.jl (orange dotted)
    ax.plot(julia_data['t_list'], julia_data['lpd_ps_expvals'],
            ':', color='darkorange', linewidth=2.5, label='LPD (PauliStrings.jl)')

    ax.set_xlabel(r'Evolution time $t$')
    ax.set_ylabel(r'$\langle Z_1 \rangle$')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_xlim(0, 5)
    ax.grid(True, alpha=0.3)

    # Add title
    ax.set_title('Python vs Julia (PauliStrings.jl)')

    # Add panel label
    ax.text(-0.12, 1.05, 'b', transform=ax.transAxes, fontsize=20,
            fontweight='bold', va='top')

    plt.tight_layout()
    fig.savefig(output_file, bbox_inches='tight', dpi=150)
    print(f"Saved: {output_file}")
    plt.close()

def print_comparison_table(python_data, julia_data):
    """Print a detailed comparison table."""
    print("\n" + "="*90)
    print("DETAILED COMPARISON TABLE")
    print("="*90)
    print(f"{'t':>6} | {'Ideal':>10} | {'Trotter':>10} | {'LPD(Py)':>10} | {'LPD(PP.jl)':>12} | {'LPD(PS.jl)':>12}")
    print("-"*90)

    dt = 0.1
    for t_check in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]:
        idx = int(t_check / dt)

        ideal = python_data['ideal_expvals'][idx]
        trott = python_data['trott_expvals'][idx]
        lpd_py = python_data['lpd_expvals'][idx]
        lpd_pp = julia_data['lpd_pp_expvals'][idx]
        lpd_ps = julia_data['lpd_ps_expvals'][idx]

        print(f"{t_check:>6.1f} | {ideal:>10.6f} | {trott:>10.6f} | {lpd_py:>10.6f} | {lpd_pp:>12.6f} | {lpd_ps:>12.6f}")

    print("-"*90)

    # Compute max errors
    lpd_py = np.array(python_data['lpd_expvals'])
    lpd_pp = np.array(julia_data['lpd_pp_expvals'])
    lpd_ps = np.array(julia_data['lpd_ps_expvals'])
    trott = np.array(python_data['trott_expvals'])

    print(f"\nMax |LPD(Python) - LPD(PauliPropagation.jl)|: {np.max(np.abs(lpd_py - lpd_pp)):.6f}")
    print(f"Max |LPD(Python) - LPD(PauliStrings.jl)|:      {np.max(np.abs(lpd_py - lpd_ps)):.6f}")
    print(f"Max |LPD(PauliPropagation.jl) - Trotter|:      {np.max(np.abs(lpd_pp - trott)):.6f}")
    print(f"Max |LPD(PauliStrings.jl) - Trotter|:          {np.max(np.abs(lpd_ps - trott)):.6f}")

def main():
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load data
    python_file = os.path.join(script_dir, 'pf2_python_results.json')
    julia_file = os.path.join(script_dir, 'julia_lpd_results.csv')

    print("Loading Python results...")
    python_data = load_python_results(python_file)

    print("Loading Julia results...")
    julia_data = load_julia_results(julia_file)

    # Print comparison table
    print_comparison_table(python_data, julia_data)

    # Create comparison plots
    print("\nCreating comparison plots...")

    output1 = os.path.join(script_dir, 'comparison_python_vs_paulipropagation.pdf')
    plot_comparison_pauliprop(python_data, julia_data, output1)

    output2 = os.path.join(script_dir, 'comparison_python_vs_paulistrings.pdf')
    plot_comparison_paulistrings(python_data, julia_data, output2)

    print("\nDone!")

if __name__ == "__main__":
    main()
