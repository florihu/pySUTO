
import numpy as np

import logging


class UncRAS:
    '''
    RAS that includes an uncertainty assessment per propagation step
    '''
    def __init__(self):
        self.max_iter = 10000
        self.tol = 1e-6
        self.results = {'A': None, 'r': None, 'c': None,
                        'row_sums': None, 'col_sums': None,
                        'iter': None, 'residual': None}

    def fit(self, A0, r, c, E, tol=None, max_iter=None):
        tol = tol if tol is not None else self.tol
        max_iter = max_iter if max_iter is not None else self.max_iter

        A0 = A0.astype(float)
        r = r.astype(float)
        c = c.astype(float)
        E = E.astype(float)

        assert r.sum() == c.sum(), "Row and column sums must match."
        assert A0.shape == E.shape, "A0 and E must have the same shape."

        # Fixed part is the part not subject to adjustment
        fixed_part = A0 - E

        # Start with initial A equal to fixed part + flexible part (initialize flexible part as E)
        A = fixed_part + E
        flexible_part = E.copy()

        for iteration in range(max_iter):
            # Row adjustment multiplier
            row_sums = A.sum(axis=1)
            with np.errstate(divide='ignore', invalid='ignore'):
                row_mult = np.divide(r, row_sums, out=np.ones_like(r), where=row_sums!=0)
            
            # Apply row multipliers only to flexible part E
            flexible_part = flexible_part * row_mult[:, np.newaxis]

            # New A after row scaling
            A = fixed_part + flexible_part

            # Column adjustment multiplier
            col_sums = A.sum(axis=0)
            with np.errstate(divide='ignore', invalid='ignore'):
                col_mult = np.divide(c, col_sums, out=np.ones_like(c), where=col_sums!=0)

            # Apply column multipliers only to flexible part E
            flexible_part = flexible_part * col_mult[np.newaxis, :]

            # Update A again
            A = fixed_part + flexible_part

            # Check residuals
            row_residual = np.linalg.norm(A.sum(axis=1) - r)
            col_residual = np.linalg.norm(A.sum(axis=0) - c)

            if row_residual < tol and col_residual < tol:
                self.results.update({
                    'A': A,
                    'r': r,
                    'c': c,
                    'row_sums': A.sum(axis=1),
                    'col_sums': A.sum(axis=0),
                    'iter': iteration,
                    'residual': (row_residual, col_residual)
                })
                break

            if iteration == max_iter - 1:
                logging.warning(f"RAS did not converge after {max_iter} iterations.")
                self.results.update({
                    'A': A,
                    'r': r,
                    'c': c,
                    'row_sums': A.sum(axis=1),
                    'col_sums': A.sum(axis=0),
                    'iter': iteration,
                    'residual': (row_residual, col_residual)
                })


if __name__ == "__main__":
    # Test the Simple RAS implementation
    A0 = np.array([[10, 20], [30, 40]])
    E = np.array([[0, 2], [3, 50]])

    r = np.array([30, 80])
    c = np.array([40, 70])  # sum to 110, matching sum(r)
    ras = UncRAS()
    ras.fit( A0, r, c, E)
    print(ras.results)