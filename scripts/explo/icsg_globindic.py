
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import os

figp = r'/mnt/nfs_fineprint/tmp/PYSUT/pySUTO/figs/ex'




def plot_cap(df):
    """
    Plot Mine Capacity Utilization (%) and Refineries Capacity Utilization (%)
    as time series, with a linear regression line for each series.
    """
    sel = [
        "Mine Capacity Utilization (%)",
        "Refineries Capacity Utilization (%)",
    ]

    d = df[df["Series"].isin(sel)].copy()
    d["Year"] = pd.to_numeric(d["Year"], errors="coerce")
    d["Value"] = pd.to_numeric(d["Value"], errors="coerce")
    d = d.dropna(subset=["Year", "Value"])
    d = d.sort_values(["Series", "Year"])

    fig, ax = plt.subplots(figsize=(11, 6))

    colors = {
        "Mine Capacity Utilization (%)": "tab:blue",
        "Refineries Capacity Utilization (%)": "tab:orange",
    }

    for series_name in sel:
        s = d[d["Series"] == series_name]
        if s.empty:
            continue

        x = s["Year"].to_numpy()
        y = s["Value"].to_numpy()
        c = colors.get(series_name, None)

        mean = np.mean(y)
        std = np.std(y)

        
        # Time series
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2,
            color=c,
            label=series_name,
        )

        #plot only mean
        ax.axhline(mean, color=c, linestyle=":", linewidth=4, alpha=0.9, label=f"{series_name} mean, std={std:.1f}%")

        # Linear regression trend line
        if len(x) >= 2:
            m, b = np.polyfit(x, y, 1)
            x_fit = np.linspace(x.min(), x.max(), 200)
            y_fit = m * x_fit + b
            ax.plot(
                x_fit,
                y_fit,
                linestyle="--",
                linewidth=2,
                color=c,
                alpha=0.8,
                label=f"{series_name} trend",
            )

    ax.set_title("Capacity Utilization Time Series with Linear Trend")
    ax.set_xlabel("Year")
    ax.set_ylabel("Utilization (%)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    
    path = os.path.join(figp, "icsg_capacity_utilization.png")
    os.makedirs(figp, exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight")

    plt.show()

    print(f"Plot saved to {path}")

    return fig, ax








if __name__ == "__main__":
    p = r"data/proc/datafeed/icsg_clean/icsg_global_indicators.csv"
    df = pd.read_csv(p)
    plot_cap(df)


