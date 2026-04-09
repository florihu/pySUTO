import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.font_manager as fm
import numpy as np
import os
from matplotlib.patches import Patch

p = 'data/proc/feedprep/icsg_clean/icsg_flows_comb.csv'
figp = r'/mnt/nfs_fineprint/tmp/PYSUT/pySUTO/figs/ex'


def _best_font():
    available = {f.name for f in fm.fontManager.ttflist}
    for candidate in ['DejaVu Sans', 'Liberation Sans', 'FreeSans',
                      'Nimbus Sans', 'Helvetica', 'Arial']:
        if candidate in available:
            return candidate
    return 'DejaVu Sans'


def calc_shares(df):
    df = df[df['Supply_Use'] == 'Supply'].copy()
    process_map = {
        'Mining':   'Mining',
        'Smelting': 'Smelting',
        'Refining': 'Refining',
        'Semi':     'Semis',
    }
    df['Process_label'] = df['Process'].map(process_map)
    df = df.dropna(subset=['Process_label'])

    results = {}
    for year in [1995, 2024]:
        sub   = df[df['Year'] == year]
        total = sub.groupby('Process_label')['Value'].sum()
        china = (sub[sub['Region'].str.strip() == 'China']
                 .groupby('Process_label')['Value'].sum())
        results[year] = (china / total * 100).round(1)
    return results


def plot_shares(shares: dict, save_path: str):
    plt.rcParams['font.family'] = _best_font()

    process_order = ['Mining', 'Smelting', 'Refining', 'Semis']

    china_95 = [shares[1995].get(p, 0) for p in process_order]
    china_24 = [shares[2024].get(p, 0) for p in process_order]
    rest_95  = [100 - v for v in china_95]
    rest_24  = [100 - v for v in china_24]

    C_CHINA_95 = '#B87333'
    C_CHINA_24 = '#7C2D12'
    C_REST     = '#D1C9C0'
    REST_ALPHA = 0.45
    LABEL_CLR  = '#F5F0EB'
    BG         = '#FDFAF7'
    GRID_CLR   = '#E8E0D8'

    n     = len(process_order)
    x     = np.arange(n)
    width = 0.35
    gap   = 0.06

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
    ax.set_facecolor(BG)

    # ── bars ──────────────────────────────────────────────────────────────
    x95 = x - width/2 - gap/2
    x24 = x + width/2 + gap/2

    bc95 = ax.bar(x95, china_95, width, color=C_CHINA_95, zorder=3, linewidth=0)
    ax.bar(x95, rest_95, width, bottom=china_95,
           color=C_REST, alpha=REST_ALPHA, zorder=3, linewidth=0)

    bc24 = ax.bar(x24, china_24, width, color=C_CHINA_24, zorder=3, linewidth=0)
    ax.bar(x24, rest_24, width, bottom=china_24,
           color=C_REST, alpha=REST_ALPHA, zorder=3, linewidth=0)

    # ── in-bar labels ─────────────────────────────────────────────────────
    def label_bars(bars, values):
        for bar, val in zip(bars, values):
            if val >= 4:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() / 2,
                        f'{val:.0f}%',
                        ha='center', va='center',
                        fontsize=9, fontweight='bold',
                        color=LABEL_CLR, zorder=4)

    label_bars(bc95, china_95)
    label_bars(bc24, china_24)

    # ── delta annotations ─────────────────────────────────────────────────
    for bar_95, bar_24, v95, v24 in zip(bc95, bc24, china_95, china_24):
        if abs(v24 - v95) < 1:
            continue
        xc_95 = bar_95.get_x() + bar_95.get_width() / 2
        xc_24 = bar_24.get_x() + bar_24.get_width() / 2
        y_mid = max(v95, v24) + 4
        ax.annotate('', xy=(xc_24, y_mid), xytext=(xc_95, y_mid),
                    arrowprops=dict(arrowstyle='->', color=C_CHINA_24,
                                    lw=1.4, connectionstyle='arc3,rad=0'),
                    zorder=5)
        delta = v24 - v95
        ax.text((xc_95 + xc_24) / 2, y_mid + 1.5,
                f'+{delta:.0f} pp' if delta > 0 else f'{delta:.0f} pp',
                ha='center', va='bottom', fontsize=8,
                color=C_CHINA_24, fontweight='bold')

    # ── axes & grid ───────────────────────────────────────────────────────
    ax.set_ylim(0, 115)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f'{int(v)}%'))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(25))
    ax.set_xticks(x)
    ax.set_xticklabels(process_order, fontsize=12)
    ax.tick_params(axis='both', length=0, labelcolor='#555')
    ax.set_ylabel('Share of global supply', fontsize=11,
                  color='#555', labelpad=10)
    ax.yaxis.grid(True, color=GRID_CLR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # ── legend ────────────────────────────────────────────────────────────
    legend_elements = [
        Patch(facecolor=C_CHINA_95, label='China 1995'),
        Patch(facecolor=C_CHINA_24, label='China 2024'),
        Patch(facecolor=C_REST, alpha=REST_ALPHA, label='Rest of world'),
    ]
    ax.legend(handles=legend_elements,
              loc='upper left', bbox_to_anchor=(1.01, 1.0),
              borderaxespad=0, frameon=False, fontsize=10,
              labelcolor='#444')

    # ── titles ────────────────────────────────────────────────────────────
    fig.text(0.13, 0.96, 'From minor player to global manufacturer',
             fontsize=15, fontweight='bold', color='#1a1a1a', va='top')
    fig.text(0.13, 0.91,
             "China's share of global copper supply by process stage, 1995 vs 2024",
             fontsize=10, color='#777', va='top')
    fig.text(0.13, 0.02, 'Source: ICSG — Supply flows only',
             fontsize=8, color='#aaa')

    plt.tight_layout(rect=[0, 0.04, 0.87, 0.89])

    # ── save ──────────────────────────────────────────────────────────────
    os.makedirs(save_path, exist_ok=True)
    out = os.path.join(save_path, 'china_copper_shares.png')
    fig.savefig(out, dpi=180, bbox_inches='tight', facecolor=BG)
    print(f'Figure saved → {out}')
    plt.show()
    return fig


if __name__ == "__main__":
    df = pd.read_csv(p)
    shares = calc_shares(df)
    plot_shares(shares, save_path=figp)