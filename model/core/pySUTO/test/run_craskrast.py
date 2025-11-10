
import numpy as np
import sparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt


sys.path.append(str(Path(__file__).resolve().parent.parent))  # adds pySUTO to path
from opt.ProblemData import ProblemData
from opt.KRASCRASOptimizer import KRASCRASOptimizer
from opt.Diagnostics import Diagnostics

# Define toy problem matrices
G = sparse.COO.from_numpy(np.array([[1, 2, 0],
                                    [0, 1, 1]]))
c = np.array([4.0, 3.0])
sigma = np.ones_like(c) * 0.1
a0 = np.array([0.01, 0.5, 0.2])

problem = ProblemData(
    G=G,
    c=c,
    sigma=sigma,
    a0=a0,
    c_idx_map={0: "balance_1", 1: "balance_2"},
    vars_map={0: "a1", 1: "a2", 2: "a3"},
    c_to_var=None
)
optimizer = KRASCRASOptimizer(problem, max_iter=1000, tol=1e-8, verbose=True, n_jobs=1)

result = optimizer.solve()

result.dump('test_run')

result.plot_convergence(name='test_convergence.png', logy=True)



if __name__ == "__main__":
    run_path = Path('data', 'proc', 'opt', 'test_run')

    diag = Diagnostics.load(run_path)

    print("Diagnostics Summary:")
    print(diag.summary())