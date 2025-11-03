import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os


class Diagnostics:
    """Container and utility methods for KRAS-CRAS diagnostics."""

    
    DEFAULTS = {
        "converged": False,
        "params": {},
        "a_init": np.array([]),
        "a_final": np.array([]),
        "c_init": np.array([]),
        "c_final": np.array([]),
        "iterations": 0,
        "history": {},
        "final_residuals": np.array([]),
        "final_rel_resid": np.array([]),
        "logger": None,
        "out_name": 'res',
        "idx_map_c": {},
        "output_path": r'data/proc/opt',
        "fig_path": r'figs\explo',
        }



    def __init__(self, **kwargs):

        """Initialize diagnostics with default or provided values."""
        for k, v in self.DEFAULTS.items():
            setattr(self, k, kwargs.get(k, v))
        if hasattr(self, "logger") and self.logger:
            self.logger.debug("Diagnostics initialized.")
    
    def update(self, **kwargs):
        """Update diagnostic attributes dynamically."""
        for k, v in kwargs.items():
            setattr(self, k, v)
        if hasattr(self, "logger") and self.logger:
            self.logger.debug(f"Updated diagnostics with: {list(kwargs.keys())}")

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
        plt.plot(x, y, marker='o', markersize=3, linewidth=1, alpha = 0.5, color='black')
        plt.xlabel("Iteration")
        plt.ylabel("Max relative residual")
        if logy:
            plt.yscale("log")
        plt.title("KRAS-CRAS Convergence")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
        plt.tight_layout()
        plt.show()
        # store fig
        plt.savefig(f'{self.fig_path}/{name}', dpi=300)
        

    def plot_rel_resid(self, name='rel_resid_per_constraint.png'):
        """Plot the relative residual per constraint over iterations, colored by constraint."""
        rel_resid = np.array(self.history["rel_resid"])  # shape (iterations, n_constraints)
        iterations = rel_resid.shape[0]
        n_constraints = rel_resid.shape[1]

        plt.figure(figsize=(8, 6))
        cmap = plt.cm.get_cmap('tab10', n_constraints)  # you can change 'tab10' to 'viridis', 'plasma', etc.

        for i in range(n_constraints):
            plt.plot(
                np.arange(1, iterations + 1).astype(int),
                rel_resid[:, i],
                color=cmap(i),
                alpha=0.7,
                label=f'Constraint {i+1}'
            )

        plt.xlabel("Iteration")
        plt.ylabel("Relative Residual per Constraint")
        #plt.yscale("log")
        plt.title("Relative Residuals per Constraint over Iterations")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
        plt.legend(fontsize='small', ncol=2)
        plt.tight_layout()

        plt.savefig(f'{self.fig_path}/{name}', dpi=300)
        

    def to_dataframe(self):
        """Convert iteration history to a pandas DataFrame."""
        df = pd.DataFrame(self.history)
        df["iteration"] = np.arange(1, len(df) + 1)
        return df
    


    def plot_a_init_final(self, name='a_comparison.png'):
        """Plot initial vs final variable scatterplot."""
        plt.figure(figsize=(6, 6))
        plt.scatter(self.a_init, self.a_final, s=20, c='black')
        max_val = max(np.max(self.a_init), np.max(self.a_final))
        plt.plot([0, max_val], [0, max_val], 'r--', label='y=x', alpha=0.5)
        plt.xlabel("Initial variable values")
        plt.ylabel("Final variable values")
        plt.title("Initial vs Final Variable Values")
        plt.legend()
        plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
        plt.tight_layout()
        plt.show()
        # store fig
        plt.savefig(f'{self.fig_path}/{name}', dpi=300)


    def plot_a_diff_histogram(self, abs = False,  name='a_diff_histogram.png'):
        """Plot histogram of differences between initial and final variable values."""

        if abs:
            diffs = np.abs(self.a_final - self.a_init)
        else:
            diffs = self.a_final - self.a_init

        plt.figure(figsize=(8, 5))
        plt.hist(diffs, bins=30, color='blue', alpha=0.3, edgecolor='black')
        plt.xlabel("Difference (Final - Initial)")
        plt.ylabel("Frequency")
        plt.title("Histogram of Variable Value Differences")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
        plt.tight_layout()
        plt.show()
        # store fig
        plt.savefig(f'{self.fig_path}/{name}', dpi=300)
        


    def plot_c_init_final(self, name='c_comparison.png'):
        """Plot initial vs final constraint scatterplot."""
        plt.figure(figsize=(6, 6))
        plt.scatter(self.c_init, self.c_final, c='black',  s=20)
        max_val = max(np.max(self.c_init), np.max(self.c_final))
        plt.plot([0, max_val], [0, max_val], 'r--', label='y=x', alpha=0.5)
        plt.xlabel("Initial constraint values")
        plt.ylabel("Final constraint values")
        plt.title("Initial vs Final Constraint Values")
        plt.legend()
        plt.grid(False)
        plt.tight_layout()
        plt.show()
        # store fig
        plt.savefig(f'{self.fig_path}/{name}', dpi=300)
        

    def dump(self, fold_name='res'):

        """Save diagnostics to JSON and arrays to .npy files."""
        path = os.path.join(self.output_path, fold_name)
        os.makedirs(path, exist_ok=True)
        
        # store main arrays
        for attr in ['a_init', 'a_final', 'c_init', 'c_final']:
            np.save(os.path.join(path, f'{attr}.npy'), getattr(self, attr))

        # store history arrays and keep filenames for metadata

        per_row = ['residuals', 'rel_resid'] # nested arrays
        per_it = ['max_rel_resid', 'mean_rel_resid', 'alpha', 'iteration'] # single value per iteration


        # store the nested arrays keys of history as 2d arrays and th eper it as 1d array
        for key in per_row:
            if key in self.history:
                arr_2d = np.array(self.history[key])
                np.save(os.path.join(path, f'history_{key}.npy'), arr_2d)
        for key in per_it:
            if key in self.history:
                arr_1d = np.array(self.history[key])
                np.save(os.path.join(path, f'history_{key}.npy'), arr_1d)

        

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
