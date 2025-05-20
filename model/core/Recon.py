
import numpy as np
import cvxpy as cp


class GeneralisedRAS:
    """
    Constrained GRAS (cRAS) variant after Tarançon & Del Río (2005) §3,
    using the Junius & Oosterhaven (2003) GRAS notation.

    A is decomposed into P ≥ 0 and N ≥ 0 so that A = P - N.
    Constraints are G vec(A) = c, where G = G+ - G-.

    uses : https://www.cvxpy.org/

    References:
      - Junius & Oosterhaven (2003), Generalised RAS for P-N matrices. :contentReference[oaicite:0]{index=0}
      - Lenzen et al. (2006), CRAS under conflicting information. :contentReference[oaicite:1]{index=1}
      - Tarançon & Del Río (2005), “cRAS” §3 (interval‐margins, reliability, conflict).
    """

    def __init__(self, A0, G_plus, G_minus, c, tol=1e-6, max_iter=500):
        """
        Parameters
        ----------
        A0 : 2D np.ndarray
            Initial seed matrix (can contain negative entries).
        G_plus, G_minus : lists of 2D masks (same shape as A0)
            Each constraint i has G_plus[i], G_minus[i] with entries 0/1,
            so that G_i•vec(A) = sum(G+ * A) - sum(G- * A).
        c : 1D array of length NC
            Right‐hand sides of the NC constraints.
        tol : float
            Stopping tolerance for ‖G vec(A) - c‖.
        max_iter : int
            Maximum number of full‐sweep iterations.
        """
        self.P = np.where(A0 > 0, A0, 0.0)  # positive part
        self.N = np.where(A0 < 0, -A0, 0.0) # negative part
        self.Gp = G_plus
        self.Gm = G_minus
        self.c = np.asarray(c, float)
        self.NC = len(c)
        self.tol = tol
        self.max_iter = max_iter

    def _apply_scaler(self, r_i, mask_p, mask_n):
        """
        Apply the i-th scaler r_i to P and N:
        P_ij ← P_ij * r_i^( mask_p_ij )
        N_ij ← N_ij * r_i^( mask_n_ij )
        Implements the sign‐aware exponent Sgn(Gij) in Eq 10.
        """
        # For entries where G+==1, exponent = +1; where G-==1, exponent = -1; else 0.
        exp = mask_p - mask_n
        # Scale
        self.P *= np.power(r_i, np.where(exp>0, exp, 0))
        self.N *= np.power(r_i, np.where(exp<0, -exp, 0))

    def fit(self):

        """
        Run the cRAS balancing procedure.

        Returns
        -------
        A_balanced : 2D np.ndarray
            The balanced matrix A = P - N.
        """
        for iteration in range(self.max_iter):
            # Save previous constraint residual for stopping test
            A_vec = (self.P - self.N).ravel()
            Gmat = np.stack([ (g_p - g_m).ravel() for g_p, g_m in zip(self.Gp, self.Gm) ])
            resid = Gmat.dot(A_vec) - self.c

            if np.linalg.norm(resid, ord=2) < self.tol * np.linalg.norm(self.c, ord=2):
                break  # Eq (11) satisfied

            # Sweep through each constraint i
            improved = False
            for i in range(self.NC):
                g_i = Gmat[i]      # vector of +1, -1, 0 masks flattened
                # Compute current constraint value Gi·A
                GiA = g_i.dot(A_vec)
                # Desired c_i; form scalar r_i to correct GiA→c_i:
                # r_i ^ (sum|g_i|) = c_i / GiA  ⇒  r_i = (c_i / GiA)^(1/∑|g_i|)
                weight = np.sum(np.abs(g_i))
                if GiA <= 0 or self.c[i] <= 0:
                    continue  # skip ill-defined
                r_i = (self.c[i] / GiA) ** (1.0 / weight)

                # Apply r_i to P and N
                self._apply_scaler(r_i, self.Gp[i], self.Gm[i])

                # Check if constraint improved
                A_vec = (self.P - self.N).ravel()
                new_resid = Gmat[i].dot(A_vec) - self.c[i]
                if abs(new_resid) < abs(resid[i]) * (1 - self.tol):
                    improved = True

            # Stopping condition if no improvement (Eq 12)
            if not improved:
                break

        return self.P - self.N

def test_cras_simple_2x2():
    # 1) Seed matrix A0 (can have positive & negative entries)
    A0 = np.array([
        [10.0,  5.0],
        [-3.0, 12.0]
    ])

    # 2) Build constraint masks for 4 constraints:
    #    C0: row 0 sum = 20
    #    C1: row 1 sum = 15
    #    C2: col 0 sum =  8
    #    C3: col 1 sum = 27

    # Masks: each is a 2×2 array of 0/1 indicating which cells enter G⁺;
    #       negatives enter via G⁻ mask.
    # Here all constraints involve P–N directly, so G_minus is zero.

    # Row-0 sum:
    Gp0 = np.array([[1, 1],
                    [0, 0]])
    Gm0 = np.zeros((2,2))

    # Row-1 sum:
    Gp1 = np.array([[0, 0],
                    [1, 1]])
    Gm1 = np.zeros((2,2))

    # Col-0 sum:
    Gp2 = np.array([[1, 0],
                    [1, 0]])
    Gm2 = np.zeros((2,2))

    # Col-1 sum:
    Gp3 = np.array([[0, 1],
                    [0, 1]])
    Gm3 = np.zeros((2,2))

    G_plus  = [Gp0, Gp1, Gp2, Gp3]
    G_minus = [Gm0, Gm1, Gm2, Gm3]
    c        = [20.0, 15.0, 8.0, 27.0]

    # 3) Run constrained RAS
    cras = GeneralisedRAS(A0, G_plus, G_minus, c, tol=1e-8, max_iter=1000)
    A_balanced = cras.fit()

    # 4) Check results
    print("Balanced Matrix:\n", A_balanced)
    print("Row sums:", A_balanced.sum(axis=1))   # expect [20.0, 15.0]
    print("Col sums:", A_balanced.sum(axis=0))   # expect [ 8.0, 27.0]
