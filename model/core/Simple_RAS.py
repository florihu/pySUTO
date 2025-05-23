
import numpy as np
import cvxpy as cp
import logging


class SimpleRAS:
    '''
    Simple RAS algorithm for balancing a matrix to given row and column sums.
    It is more thought of as a playground for testing the RAS algorithm.
    '''
    def __init__(self):
        self.tol = 1e-6
        self.max_iter = 1000
        self.results = {'A': None, 'r': None, 'c': None, 'iter': None, 'residual': None}

    def fit(self, A0, r, c, tol=None, max_iter=None):
        self.A0 = A0
        self.r = r
        self.c = c
        tol = tol if tol is not None else self.tol
        max_iter = max_iter if max_iter is not None else self.max_iter
        
        A = A0.astype(float).copy()

        c = c.astype(float)
        r = r.astype(float)

        for iteration in range(max_iter):
            A_prev = A.copy()
            # R-step: adjust rows
            row_sums = A.sum(axis=1)
            # Avoid division by zero
            row_mult = np.divide(r, row_sums, out=np.ones_like(r), where=row_sums != 0)
            A *= row_mult[:, np.newaxis]

            # S-step: adjust columns
            col_sums = A.sum(axis=0)
            col_mult = np.divide(c, col_sums, out=np.ones_like(c), where=col_sums != 0)
            A *= col_mult

            # Convergence check (Frobenius norm)
            residual = np.linalg.norm(A - A_prev, ord='fro')
            logging.debug(f"Iteration {iteration}: Residual = {residual}")

            if residual < tol:
                self.results.update({
                    'A': A,
                    'r': r,
                    'c': c,
                    'iter': iteration,
                    'residual': residual
                })
                break

            
if __name__ == "__main__":
    # Test the Simple RAS implementation
    A0 = np.array([[10, 20], [30, 40]])
    r = np.array([1.5, 0.5])
    c = np.array([20, .1])
    ras = SimpleRAS()
    ras.fit( A0, r, c)
    
    res = ras.results
    print("Initial Matrix:\n", A0)
    print("Balanced Row Factors:", res['r'])
    print("Balanced Column Factors:", res['c'])
    print("Residual:", res['residual'])
    print("Iterations:", res['iter'])
    print("Balanced Matrix:\n", res['A'])
    print("Row sums:", res['A'].sum(axis=1))
    print("Column sums:", res['A'].sum(axis=0))