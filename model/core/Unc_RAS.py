
import numpy as np

import logging


class UncRAS:
    '''
    RAS that includes an uncertainty assessment per propagation step

    '''
    def __init__(self):
        self.max_iter = 1000
        self.tol = 1e-6

        self.results = {'A': None, 'r': None, 'c': None, 'row_sums': None, 'col_sums': None, 'iter': None, 'residual': None}

    def fit(self, A0, r, c,E, tol=None, max_iter=None):
        self.A0 = A0
        self.r = r
        self.c = c
        self.E = E
        tol = tol if tol is not None else self.tol
        max_iter = max_iter if max_iter is not None else self.max_iter

        A = A0.astype(float).copy()

        c = c.astype(float)
        r = r.astype(float)

        E = E.astype(float)

        for iteration in range(max_iter):
            A_prev = A.copy()
            # R-step: adjust rows
            row_sums = A.sum(axis=1)
            # Avoid division by zero
            row_mult = np.divide(r, row_sums, out=np.ones_like(r), where=row_sums != 0)
            
            # S-step: adjust columns
            col_sums = A.sum(axis=0)
            col_mult = np.divide(c, col_sums, out=np.ones_like(c), where=col_sums != 0)

            # fixed part is the part of A0 which is not changed
            fixed_part = self.A0 - E
            # flexible part is the part of A0 which is changed by the RAS depending on how unceertain the part is
            # how much room is to reshape the matrix the bigge E the more flexible
            flex_part = row_mult[:, np.newaxis] * E * col_mult

            A = fixed_part + flex_part

            # Convergence check (Frobenius norm)
            residual = np.linalg.norm(A - A_prev, ord='fro')

            if residual < tol:
                self.results.update({
                    'A': A,
                    'r': r,
                    'c': c,
                    'row_sums': row_sums,
                    'col_sums': col_sums,
                    'iter': iteration,
                    'residual': residual
                })
                break


if __name__ == "__main__":
    # Test the Simple RAS implementation
    A0 = np.array([[10, 20], [30, 40]])
    E = np.array([[0, 2], [3, 4]])
    r = np.array([30, 80])
    c = np.array([50, 50])
    ras = UncRAS()
    ras.fit( A0, r, c, E)
    print(ras.results)