import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os


class Diagnostics:
    """Container and utility methods for KRAS-CRAS diagnostics."""
    def __init__(self, **kwargs):

        self.converged = kwargs.get("converged", False)
        self.params = kwargs.get("params", {})
        self.a_init = kwargs.get("a_init", np.array([]))
        self.a_final = kwargs.get("a_final", np.array([]))
        self.c_init = kwargs.get("c_init", np.array([]))
        self.c_final = kwargs.get("c_final", np.array([]))
        self.iterations = kwargs.get("iterations", 0)
        self.history = kwargs.get("history", {})
        self.final_residuals = kwargs.get("final_residuals", np.array([]))
        self.final_rel_resid = kwargs.get("final_rel_resid", np.array([]))
        self.logger = kwargs.get("logger", None)
        self.name = kwargs.get("out_name", os.getcwd())
        

        self.output_path = r'data/proc/opt'
        self.fig_path = r'figs\explo'
        

    # --- Standard analysis functions ---

    def summary(self):
        """Return a concise summary of solver performance."""
        return {
            "Converged": self.converged,
            "Iterations": self.iterations,
            "Final max rel resid": float(np.max(self.final_rel_resid)),
            "Final mean rel resid": float(np.mean(self.final_rel_resid)),
        }

    def plot_convergence(self, name ='convergence.png', logy=True):
        """Plot evolution of residuals over iterations."""
        if "max_rel_resid" not in self.history:
            raise ValueError("History does not contain 'max_rel_resid'.")
        y = np.array(self.history["max_rel_resid"])
        x = np.arange(1, len(y)+1)
        plt.figure(figsize=(6, 4))
        plt.plot(x, y, marker='o', markersize=3, linewidth=1)
        plt.xlabel("Iteration")
        plt.ylabel("Max relative residual")
        if logy:
            plt.yscale("log")
        plt.title("KRAS-CRAS Convergence")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        # store fig
        plt.savefig(f'{self.fig_path}/{name}', dpi=300)
        

    def to_dataframe(self):
        """Convert iteration history to a pandas DataFrame."""
        df = pd.DataFrame(self.history)
        df["iteration"] = np.arange(1, len(df) + 1)
        return df
    
    def plot_a_init_final(self, name='a_comparison.png'):
        """Plot initial vs final variable scatterplot."""
        plt.figure(figsize=(6, 6))
        plt.scatter(self.a_init, self.a_final, s=20)
        max_val = max(np.max(self.a_init), np.max(self.a_final))
        plt.plot([0, max_val], [0, max_val], 'r--', label='y=x')
        plt.xlabel("Initial variable values")
        plt.ylabel("Final variable values")
        plt.title("Initial vs Final Variable Values")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        # store fig
        plt.savefig(f'{self.fig_path}/{name}', dpi=300)
        

    def plot_c_init_final(self, name='c_comparison.png'):
        """Plot initial vs final constraint scatterplot."""
        plt.figure(figsize=(6, 6))
        plt.scatter(self.c_init, self.c_final,  s=20)
        max_val = max(np.max(self.c_init), np.max(self.c_final))
        plt.plot([0, max_val], [0, max_val], 'r--', label='y=x')
        plt.xlabel("Initial constraint values")
        plt.ylabel("Final constraint values")
        plt.title("Initial vs Final Constraint Values")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        # store fig
        plt.savefig(f'{self.fig_path}/{name}', dpi=300)
        

    def to_json(self, fold_name='res'):
        """Save diagnostics to JSON and arrays to .npy files."""
        path = os.path.join(self.output_path, fold_name)
        os.makedirs(path, exist_ok=True)

        # store main arrays
        for attr in ['a_init', 'a_final', 'c_init', 'c_final']:
            np.save(os.path.join(path, f'{attr}.npy'), getattr(self, attr))

        # store history arrays and keep filenames for metadata
        history_files = {}
        for key, value in self.history.items():
            if isinstance(value, np.ndarray):
                fname = f'history_{key}.npy'
                np.save(os.path.join(path, fname), value)
                history_files[key] = fname
            else:
                # keep non-array entries as-is
                history_files[key] = value

        # store other large arrays separately
        np.save(os.path.join(path, 'final_residuals.npy'), self.final_residuals)
        np.save(os.path.join(path, 'final_rel_resid.npy'), self.final_rel_resid)

        # store metadata
        meta = {
            "converged": self.converged,
            "params": self.params,
            "iterations": self.iterations,
            "final_residuals": self.final_residuals.tolist(),
            "final_rel_resid": self.final_rel_resid.tolist(),
        }

        # write metadata JSON
        with open(os.path.join(path, 'meta.json'), 'w') as f:
            json.dump(meta, f, indent=4)

        self.logger.info(f"Diagnostics saved to {path}")
