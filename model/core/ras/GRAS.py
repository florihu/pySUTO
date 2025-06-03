

import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np

import numpy as np

class GRAS:
    def __init__(self, A, u, v):
        """
        GRAS implementation with safe handling of zeros
         
        Parameters:
        A : numpy array - initial matrix (can contain positive and negative values)
        u : numpy array - target row sums
        v : numpy array - target column sums
        """
        self.A = np.array(A, dtype=float)
        self.u = np.array(u, dtype=float)
        self.v = np.array(v, dtype=float)
        
        # Check dimensions
        if self.A.shape != (len(u), len(v)):
            raise ValueError("Dimensions of A, u, and v don't match")
            
        # Create P and N matrices
        self.P = np.where(self.A > 0, self.A, 0)
        self.N = np.where(self.A < 0, -self.A, 0)
        
        # Initialize scaling factors
        self.r = np.ones(len(u))
        self.s = np.ones(len(v))
        
        # Results storage
        self.results = {}
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def solve(self, max_iter=1000, tol=1e-8):
        """
        Solve using iterative proportional fitting with safe division
        """
        A = self.A.copy()
        P_scaled = self.P.copy()
        N_scaled = self.N.copy()
        r = self.r.copy()
        s = self.s.copy()

        u_star = np.exp(1) * self.u
        v_star = np.exp(1) * self.v

        for iteration in range(max_iter):

            for j in range(len(s)):
                s[j] = (v_star[j] + np.sqrt(v_star[j]**2 + 4 * P_scaled[:, j].sum() * N_scaled[:, j].sum())) / (2 * P_scaled[:, j].sum())

            P_scaled = P_scaled * s[np.newaxis, :]
            N_scaled = N_scaled * (1/s)[np.newaxis, :]

            for i in range(len(r)):
                r[i] = (u_star[i] + np.sqrt(u_star[i]**2 + 4 * P_scaled[i, :].sum() * N_scaled[i, :].sum())) / (2 * P_scaled[i,  :].sum())

            # update P and N matrices
            P_scaled = P_scaled * r[:, np.newaxis]
            N_scaled = N_scaled * (1/r)[:, np.newaxis]
            
            A = P_scaled - N_scaled

            A_bal = A / np.exp(1)  # remove the 1/e scaling

            u_residual = np.linalg.norm(A_bal.sum(axis=1) - self.u)
            v_residual = np.linalg.norm(A_bal.sum(axis=0) - self.v)

            if u_residual < tol and v_residual < tol:
                self.logger.info(f"Converged after {iteration + 1} iterations.")
                self.results.update({
                    'A': A_bal,
                    'u': A_bal.sum(axis=1),
                    'v': A_bal.sum(axis=0),
                    'iter': iteration + 1,
                    'r': r,
                    's': s
                })
                break
            elif iteration == max_iter - 1:
                self.logger.warning(f"Did not converge after {max_iter} iterations.")
                self.results.update({
                    'A': X * np.exp(1),  # remove the 1/e scaling
                    'u': current_row_sums,
                    'v': current_col_sums,
                    'iter': iteration + 1,
                    'r': r,
                    's': s
                })



if __name__ == "__main__":
    A0 = np.array([
    [  7,  3,  5, -3],  # Goods row
    [  2,  9,  8,  1],  # Services row
    [-2,  0,  2,  1]    # Net taxes row
    ], dtype=float)


    # desired row and column sums
    u = np.array([15, 26, -1], dtype=float)  # New total outputs (row sums)
    v = np.array([9, 16, 17, -2], dtype=float)  # New total inputs (column sums)
    ras = GRAS(A0, u, v)
    ras.solve()
    print(ras.results)