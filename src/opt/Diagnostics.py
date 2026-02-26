import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import logging
from pathlib import Path
from scipy.stats import gaussian_kde


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
        "output_path": Path('data', 'proc', 'opt'),
        "fig_path": Path('figs', 'explo'),
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

        self.fig_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{self.fig_path}/{name}', dpi=300)
        

    def to_dataframe(self):
        """Convert iteration history to a pandas DataFrame."""
        df = pd.DataFrame(self.history)
        df["iteration"] = np.arange(1, len(df) + 1)
        return df
    

    def plot_a_init_final(self, name='a_comparison.png', xlog=False, ylog=False):
        """Plot initial vs final variable scatterplot with optional contour lines."""
        x = np.array(self.a_init)
        y = np.array(self.a_final)

        plt.figure(figsize=(6, 6))

        # --- Base scatter plot ---
        plt.scatter(x, y, s=10, c='gray', alpha=0.5, label='Variables')

        # --- Diagonal reference line ---
        max_val = max(np.max(self.a_init), np.max(self.a_final))
        plt.plot([0, max_val], [0, max_val], 'r--', label='y=x', alpha=0.7)

        # --- Axis scaling and labels ---
        if xlog:
            plt.xscale("log")
        if ylog:
            plt.yscale("log")

        plt.xlabel("Initial variable values")
        plt.ylabel("Final variable values")
        plt.title("Initial vs Final Variable Values (Rocket Plot)")
        plt.legend()
        plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
        plt.tight_layout()

        # --- Save figure ---
        Path(self.fig_path).mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{self.fig_path}/{name}', dpi=300)
        plt.show()



    def plot_a_diff_histogram(self, abs = False,  name='a_diff_histogram.png', xlog=False, ylog=True):
        """Plot histogram of differences between initial and final variable values."""

        if abs:
            diffs = np.abs(self.a_final - self.a_init)
        else:
            diffs = self.a_final - self.a_init

        plt.figure(figsize=(8, 5))
        plt.hist(diffs, bins=30, color='blue', alpha=0.3, edgecolor='black')
        if xlog:
            plt.xscale("log")
        if ylog:
            plt.yscale("log")
        plt.xlabel("Difference (Final - Initial)")
        plt.ylabel("Frequency")
        plt.title("Histogram of Variable Value Differences")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
        plt.tight_layout()
        plt.show()
        # store fig
        self.fig_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{self.fig_path}/{name}', dpi=300)
        


    def plot_c_init_final(self, name='c_comparison.png', xlog=False, ylog=False):
        """Plot initial vs final constraint scatterplot."""
        plt.figure(figsize=(6, 6))
        plt.scatter(self.c_init, self.c_final, c='black',  s=20)
        max_val = max(np.max(self.c_init), np.max(self.c_final))
        plt.plot([0, max_val], [0, max_val], 'r--', label='y=x', alpha=0.5)
        if xlog:
            plt.xscale("log")
        if ylog:
            plt.yscale("log")
        plt.xlabel("Initial constraint values")
        plt.ylabel("Final constraint values")
        plt.title("Initial vs Final Constraint Values")
        plt.legend()
        plt.grid(False)
        plt.tight_layout()
        plt.show()
        # store fig
        self.fig_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{self.fig_path}/{name}', dpi=300)

    def plot_c_diff_histogram(self, abs = False, name='c_diff_histogram.png', xlog=False, ylog=True):
        """Plot histogram of differences between initial and final constraint values."""
        if abs:
            diffs = np.abs(self.c_final - self.c_init)
        else:
            diffs = self.c_final - self.c_init

        plt.figure(figsize=(8, 5))
        plt.hist(diffs, bins=30, color='green', alpha=0.3, edgecolor='black')
        if xlog:
            plt.xscale("log")
        if ylog:
            plt.yscale("log")
        plt.xlabel("Difference (Final - Initial)")
        plt.ylabel("Frequency")
        plt.title("Histogram of Constraint Value Differences")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
        plt.tight_layout()
        plt.show()
        # store fig
        self.fig_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{self.fig_path}/{name}', dpi=300)
        

    def dump(self, fold_name='res'):
        """Save diagnostics (arrays + metadata)."""
        path = Path(self.output_path) / fold_name
        path.mkdir(parents=True, exist_ok=True)

        data_manifest = {}

        # Automatically handle known array attributes
        for attr in ['a_init', 'a_final', 'c_init', 'c_final']:
            np.save(path / f"{attr}.npy", getattr(self, attr))
            data_manifest[attr] = f"{attr}.npy"

        # Save history arrays
        for key, arr in self.history.items():
            np.save(path / f"history_{key}.npy", np.array(arr))
            data_manifest[f"history.{key}"] = f"history_{key}.npy"

        # Save final arrays
        np.save(path / "final_residuals.npy", self.final_residuals)
        np.save(path / "final_rel_resid.npy", self.final_rel_resid)
        data_manifest["final_residuals"] = "final_residuals.npy"
        data_manifest["final_rel_resid"] = "final_rel_resid.npy"

        # Write metadata (manifest + meta info)
        meta = {
            "manifest": data_manifest,
            "converged": self.converged,
            "params": self.params,
            "iterations": self.iterations,
        }

        with open(path / "meta.json", "w") as f:
            json.dump(meta, f, indent=4)

        self.logger.info(f"Diagnostics saved to {path}")

    def hist(self, name, a_x_log = True):
        #  facet plot of c and a both in init and final 2x2
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        axs[0, 0].hist(self.c_init, bins=100, color='blue', alpha=0.5, edgecolor='black')
        axs[0, 0].set_title('Initial Constraint Values')
        axs[0, 1].hist(self.c_final, bins=100, color='green', alpha=0.5, edgecolor='black')
        axs[0, 1].set_title('Final Constraint Values')
        axs[1, 0].hist(self.a_init, bins=100, color='orange', alpha=0.5, edgecolor='black')
        axs[1, 0].set_title('Initial Variable Values')
        axs[1, 1].hist(self.a_final, bins=100, color='red', alpha=0.5, edgecolor='black')
        axs[1, 1].set_title('Final Variable Values')
        for ax in axs.flat:
            ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)

        if a_x_log:
            axs[1, 0].set_xscale("log")
            axs[1, 1].set_xscale("log")
        plt.tight_layout()
        self.fig_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{self.fig_path}/{name}', dpi=300)




    @classmethod
    def load(cls, path):
        """Load diagnostics from a folder and reconstruct object."""
        path = Path(path)
        with open(path / "meta.json") as f:
            meta = json.load(f)

        obj = cls(output_path=path.parent)
        manifest = meta["manifest"]

        # Restore arrays
        for key, filename in manifest.items():
            arr = np.load(path / filename, allow_pickle=True)
            if key.startswith("history."):
                hist_key = key.split(".", 1)[1]
                obj.history[hist_key] = arr
            else:
                setattr(obj, key, arr)

        # Restore metadata
        obj.converged = meta["converged"]
        obj.params = meta["params"]
        obj.iterations = meta["iterations"]

        return obj
