import numpy as np
from qiskit.quantum_info import SparsePauliOp

class Local_Hamiltonian:
    def __init__(self, n: int, Jx=0, Jy=0, Jz=0, hx=0.0, hy=0.0, hz=0.0, pbc=False, verbose=False):
        # self.n, self.alpha = n, alpha
        self.n = n
        self.Jx, self.Jy, self.Jz = Jx, Jy, Jz
        self.hx, self.hy, self.hz = hx, hy, hz
        self.xx_tuples = [('XX', [i, j], Jx) for i in range(0, n-1) for j in range(i+1, n)]
        self.yy_tuples = [('YY', [i, j], Jy) for i in range(0, n-1) for j in range(i+1, n)]
        self.zz_tuples = [('ZZ', [i, j], Jz) for i in range(0, n-1) for j in range(i+1, n)]
        self.x_tuples = [('X', [i], hx) for i in range(0, n)] 
        self.y_tuples = [('Y', [i], hy) for i in range(0, n)] 
        self.z_tuples = [('Z', [i], hz) for i in range(0, n)] 
        if pbc: 
            # self.xx_tuples.append(('XX', [n-1, 0], Jx))
            # self.yy_tuples.append(('YY', [n-1, 0], Jy))
            # self.zz_tuples.append(('ZZ', [n-1, 0], Jz))
            raise ValueError(f'PBC is not defined!')

        self.ham = SparsePauliOp.from_sparse_list([*self.xx_tuples, *self.yy_tuples, *self.zz_tuples, *self.x_tuples, *self.y_tuples, *self.z_tuples], num_qubits=n).simplify()
        if verbose: print('The Hamiltonian: \n', self.ham)
        self.xyz_group()

    def xyz_group(self):
        self.x_terms = SparsePauliOp.from_sparse_list([*self.xx_tuples, *self.x_tuples], self.n).simplify()
        self.y_terms = SparsePauliOp.from_sparse_list([*self.yy_tuples, *self.y_tuples], self.n).simplify()
        self.z_terms = SparsePauliOp.from_sparse_list([*self.zz_tuples, *self.z_tuples], self.n).simplify()
        self.ham_xyz = [self.x_terms, self.y_terms, self.z_terms]
        self.ham_xyz = [item for item in self.ham_xyz if not np.all(abs(item.coeffs) == 0)]



def build_qmfi_hamiltonian(n, J_x, h_x, h_y, periodic=True):
    """
    Build the QMFI Hamiltonian as a SparsePauliOp
    H = J_x Σ X_j X_{j+1} + h_x Σ X_j + h_y Σ Y_j
    """
    pauli_list = []
    
    # XX interaction terms
    for j in range(n):
        if j < n-1:
            # X_j X_{j+1}
            pauli_str = ['I'] * n
            pauli_str[j] = 'X'
            pauli_str[j+1] = 'X'
            pauli_list.append((''.join(pauli_str[::-1]), J_x))
        elif periodic:
            # X_{n-1} X_0 for periodic boundary
            pauli_str = ['I'] * n
            pauli_str[n-1] = 'X'
            pauli_str[0] = 'X'
            pauli_list.append((''.join(pauli_str[::-1]), J_x))
    
    # X field terms
    for j in range(n):
        pauli_str = ['I'] * n
        pauli_str[j] = 'X'
        pauli_list.append((''.join(pauli_str[::-1]), h_x))
    
    # Y field terms
    for j in range(n):
        pauli_str = ['I'] * n
        pauli_str[j] = 'Y'
        pauli_list.append((''.join(pauli_str[::-1]), h_y))
    
    return SparsePauliOp.from_list(pauli_list)