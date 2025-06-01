"""Reusable plotting helpers."""
import numpy as np
import matplotlib.pyplot as plt


def plot_traces_grid(t: np.ndarray, summary: dict, *,
                     title: str, out_path=None, N_runs=None):
    cond_order = ["baseline-intact", "baseline-cut", "shift-intact"]
    row_keys   = [("V1", "Ca1"), ("V2", "Ca2"), ("IPSC", None)]
    colors     = {"V1":"tab:blue", "Ca1":"tab:red",
                  "V2":"tab:blue", "Ca2":"tab:red",
                  "IPSC":"black"}

    fig, axs = plt.subplots(3, 3, figsize=(16, 10),
                            sharex=True, sharey='row')
    subtitle = f"{title}\n(mean ± 1 SD"
    if N_runs is not None:
        subtitle += f", N={N_runs})"
    fig.suptitle(subtitle, fontsize=12)

    for col, cond in enumerate(cond_order):
        for row, (primary, secondary) in enumerate(row_keys):
            mean, sd = summary[cond][primary]
            ax = axs[row, col]
            ax.plot(t, mean, color=colors[primary], label=primary)
            ax.fill_between(t, mean-sd, mean+sd, color=colors[primary], alpha=0.3)

            if secondary:
                mean2, sd2 = summary[cond][secondary]
                ax.plot(t, mean2, color=colors[secondary], label=secondary)
                ax.fill_between(t, mean2-sd2, mean2+sd2,
                                color=colors[secondary], alpha=0.3)

            if row == 0:
                ax.set_title(cond.replace("-", " – "))
            if row == 2:
                ax.set_xlabel("Time (ms)")
            if col == 0:
                ax.set_ylabel({"V1":"SAC1", "V2":"SAC2", "IPSC":"DSGC"}[primary])

            ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=300)
    return fig
