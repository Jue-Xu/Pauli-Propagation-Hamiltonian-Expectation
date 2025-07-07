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
