
import numpy as np
import cvxpy as cp
import logging


class SimpleRAS:
    '''
    Simple RAS algorithm for balancing a matrix to given row and column sums.
    '''
    def __init__(self, tol=1e-6, max_iter=10000):
        self.tol = tol
        self.max_iter = max_iter
        self.results = {'A': None, 'r': None, 'c': None, 'iter': None}

    def fit(self, A0, r, c, tol=None, max_iter=None):
        tol = tol if tol is not None else self.tol
        max_iter = max_iter if max_iter is not None else self.max_iter
        
        A = A0.astype(float).copy()
        r = r.astype(float)
        c = c.astype(float)

        assert r.sum() == c.sum(), "Row and column sums must match."

        for iteration in range(max_iter):
            # R-step: adjust rows
            row_sums = A.sum(axis=1)
            row_factors = np.divide(r, row_sums, out=np.ones_like(r), where=row_sums != 0)
            A *= row_factors[:, np.newaxis]

            # S-step: adjust columns
            col_sums = A.sum(axis=0)
            col_factors = np.divide(c, col_sums, out=np.ones_like(c), where=col_sums != 0)
            A *= col_factors

            # Residuals
            row_residual = np.linalg.norm(A.sum(axis=1) - r)
            col_residual = np.linalg.norm(A.sum(axis=0) - c)
            

            if row_residual < tol and col_residual < tol:
                break

        else:
            logging.warning(f"RAS did not converge after {max_iter} iterations.")

        self.results.update({
            'A': A,
            'r': r,
            'c': c,
            'iter': iteration + 1,
        })
        return self

            
if __name__ == "__main__":
    A0 = np.array([[10, 20], [30, 20]])
    r = np.array([35, 80])
    c = np.array([40, 75])  # sum to 95, matching sum(r)

    ras = SimpleRAS()
    ras.fit(A0, r, c)
    
    res = ras.results
    print("Iterations:", res['iter'])
    print("Balanced matrix A:\n", res['A'])
    print("Row sums:", res['A'].sum(axis=1))
    print("Target r:", r)
    print("Column sums:", res['A'].sum(axis=0))
    print("Target c:", c)