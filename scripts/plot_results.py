"""Render the paper's empirical plot data; no inference or selection."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
CHECKS=[]
COL={'Frozen':'#8A939B','Ranking':'#C77B2B','IS':'#007D7B','Published':'#667596','Ablation':'#719E9B'}
DISPLAY={'IS':'Mirror'}  # records keep the identifier IS; figures show the method name
INK,MUTED,GRID='#203345','#566570','#E8ECEF'
def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.2,
        "axes.labelsize": 6.2, "axes.titlesize": 6.8, "axes.linewidth": .45,
        "xtick.labelsize": 6.0, "ytick.labelsize": 6.0, "xtick.major.size": 2.,
        "ytick.major.size": 2., "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "svg.hashsalt": "interbind-v7-compact-results",
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED,
        "ytick.color": MUTED, "axes.edgecolor": "#82909A"})

def save(fig, name):
    (HERE / "figures").mkdir(exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        meta = {"CreationDate": None, "ModDate": None} if ext == "pdf" else {"Date": None} if ext == "svg" else None
        fig.savefig(HERE / "figures" / f"{name}.{ext}", dpi=300, metadata=meta)
    plt.close(fig)

def bare(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=.5, zorder=0)
    ax.set_axisbelow(True)

def legend(fig, y=.985):
    handles = [Line2D([], [], color=COL[n], lw=1.8,
                     ls="--" if n == "Frozen" else "-", label=DISPLAY.get(n, n))
               for n in ("Frozen", "Ranking", "IS")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, y),
               ncol=3, frameon=False, handlelength=1.9, columnspacing=1.8,
               handletextpad=.45, fontsize=7.2, borderaxespad=0)

FAIL='#B5473A'
def _ecdf_panel(ax, r, heading):
    arrays = {m: np.asarray(a) for m, a in r["values"].items()}
    lower = min(0., min(a.min() for a in arrays.values()))
    upper = max(a.max() for a in arrays.values())
    x = np.unique(np.concatenate([np.asarray([lower, upper]), *[a.ravel() for a in arrays.values()]]))
    for method in ("Frozen", "Ranking", "IS"):
        yy = np.stack([100 * np.searchsorted(np.sort(v), x, side="right") / len(v) for v in arrays[method]])
        ax.fill_between(x, yy.min(0), yy.max(0), step="post", color=COL[method], alpha=.17, lw=0)
        ax.step(x, yy.mean(0), where="post", color=COL[method], lw=1.05,
                ls=(0, (2.5, 1.7)) if method == "Frozen" else "-", zorder=4)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color=GRID, lw=.5, zorder=0); ax.set_axisbelow(True)
    ax.set_xlim(lower, upper * 1.02); ax.set_ylim(0, 101); ax.set_yticks([0, 50, 100])
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.set_title(heading, loc="left", pad=5, fontsize=6.7, fontweight="semibold")
    ax.set_xlabel("|Interaction| ↓\n" + r["unit"], fontsize=6.0, labelpad=2, linespacing=1.15)
    ax.set_ylabel("Sources (%)", fontsize=6.1, labelpad=2)

def _binding_change(ax, data):
    B = {m: np.asarray(v).mean(0) for m, v in data["routing_binding"]["values"].items()}   # seed mean per scene
    C = {m: np.asarray(v).mean(0) for m, v in data["routing_cross"]["values"].items()}
    d = {m: (B[m] - B["Frozen"], C[m] - C["Frozen"]) for m in ("Ranking", "IS")}
    both_up = 100 * np.mean((d["Ranking"][0] > 0) & (d["Ranking"][1] > 0))
    is_good = 100 * np.mean((d["IS"][0] > 0) & (d["IS"][1] < 0))
    CHECKS.append({"test": "color_binding_scene_directions", "ranking_both_up_percent": float(both_up),
                   "mirror_binding_up_cross_down_percent": float(is_good),
                   "passed": bool(round(both_up) == 100 and round(is_good) == 90)})
    xl, yl = (-0.08, 0.50), (-0.26, 0.42)
    ax.axhspan(0, yl[1], color=FAIL, alpha=0.06, lw=0, zorder=0)
    ax.add_patch(plt.Rectangle((0, yl[0]), xl[1], -yl[0], color=COL["IS"], alpha=0.07, lw=0, zorder=0))
    ax.axhline(0, color=MUTED, lw=0.5, zorder=1); ax.axvline(0, color=MUTED, lw=0.5, zorder=1)
    ax.scatter(*d["Ranking"], s=4, facecolors="none", edgecolors=COL["Ranking"], linewidths=0.45, alpha=0.6, zorder=3)
    ax.scatter(*d["IS"], s=4, color=COL["IS"], lw=0, alpha=0.55, zorder=3)
    ax.scatter([0], [0], s=18, marker="s", facecolors="white", edgecolors=INK, linewidths=0.9, zorder=6)
    ax.annotate("frozen", xy=(0, 0), xytext=(-0.07, 0.06), fontsize=5.8, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.4))
    ax.text(0.49, 0.155, f"ranking: both grow\n({both_up:.0f}% of scenes)", color=COL["Ranking"], fontsize=5.9,
            fontweight="semibold", ha="right", va="top")
    ax.text(0.49, -0.03, f"{DISPLAY['IS']}: binding ↑,\ncross-effect ↓\n({is_good:.0f}% of scenes)", color=COL["IS"],
            fontsize=5.9, fontweight="semibold", ha="right", va="top")
    ax.set_xlim(*xl); ax.set_ylim(*yl)
    ax.set_xticks([0, 0.2, 0.4]); ax.set_yticks([-0.2, 0, 0.2, 0.4])
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("b  Color binding, per scene", loc="left", pad=5, fontsize=6.7, fontweight="semibold")
    ax.set_xlabel("Change in binding ↑\n(own object's color word)", fontsize=6.0, labelpad=2, linespacing=1.15)
    ax.set_ylabel("Change in cross-effect ↓\n(other object's color word)", fontsize=6.0, labelpad=1, linespacing=1.15)

def distributions_figure(data):
    """Paper Figure 5: (a) typography and (c) striped-trigger interaction CDFs; (b) per-scene color-binding change."""
    fig = plt.figure(figsize=(5.5, 1.85))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.3, 1], left=0.075, right=0.99, top=0.80, bottom=0.235, wspace=0.5)
    axa, axb, axc = (fig.add_subplot(gs[0, i]) for i in range(3))
    _ecdf_panel(axa, data["typography"], "a  Typography")
    _binding_change(axb, data)
    _ecdf_panel(axc, data["backdoor"], "c  Striped trigger")
    handles = [Line2D([], [], color=COL[n], lw=1.2, ls="--" if n == "Frozen" else "-", label=DISPLAY.get(n, n))
               for n in ("Frozen", "Ranking", "IS")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, 1.0), ncol=3, frameon=False, fontsize=6.2,
               handlelength=1.6, columnspacing=1.7, borderaxespad=0)
    save(fig, "interaction_distributions")

def answer_figure(data):
    # Identical three-method comparison in each panel. External-method audits
    # remain in the source records and their comparisons remain in Table 4.
    styles = {
        "Frozen": (COL["Frozen"], "s", "white"),
        "Ranking": (COL["Ranking"], "o", "white"),
        "IS": (COL["IS"], "o", COL["IS"]),
        "Defense-Prefix": ("#536EA1", "^", "white"),
        "Dyslexify": ("#8A6596", "D", "white"),
        "PAR": ("#405375", "P", "white"),
    }
    panels = [
        ("a  Typography", "SCAM accuracy (%)", (65, 86), [65, 75, 85],
         (-5, 64), [0, 30, 60], "typography_SCAM", "Typography"),
        ("b  Color binding", "Exchange accuracy (%)", (49, 81), [50, 60, 70, 80],
         (-34, 13), [-30, -15, 0, 10], "routing_heldout", "Routing"),
        ("c  Backdoors: stripes", "Attacked accuracy (%)", (-3, 52), [0, 25, 50],
         (-6, 96), [0, 40, 80], "backdoor_accuracy", "Backdoor"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.76))
    fig.subplots_adjust(left=.095, right=.986, top=.72, bottom=.25, wspace=.37)
    for ax, (title, xlabel, xlim, xticks, ylim, yticks, key, family) in zip(axes, panels):
        ax.set(xlim=xlim, ylim=ylim, xticks=xticks, yticks=yticks)
        bare(ax)
        ax.axhline(0, color="#9CA7AE", lw=.5, ls=(0, (2, 2)), zorder=1)
        if family == "Routing":
            ax.axhspan(ylim[0], 0, color="#FBF2ED", zorder=0)
        ax.set_title(title, loc="left", fontsize=6.8, fontweight="semibold", pad=5)
        ax.set_xlabel(xlabel, fontsize=6.2, labelpad=3)
        for method in data["display"][key]:
            x, y = data[key][method]
            samples = data["per_seed"].get(family + "|" + method, [[x], [y]])
            color, marker, face = styles[method]
            ax.plot([min(samples[0]), max(samples[0])], [y, y], color=color, lw=.8)
            ax.plot([x, x], [min(samples[1]), max(samples[1])], color=color, lw=.8)
            ax.plot(x, y, marker=marker, ms=4.2, mfc=face, mec=color, mew=.9,
                    zorder=6 if method == "IS" else 5)
        CHECKS.append({"test": "consistent_core_methods", "panel": key,
                       "methods": data["display"][key],
                       "passed": all(m in data["display"][key]
                                     for m in ("Frozen", "Ranking", "IS"))})
    axes[0].set_ylabel("Interaction removed (%)", fontsize=6.1, labelpad=3)
    # The same three symbols apply to every subject.
    handles = [Line2D([], [], ls="", marker=styles[m][1], ms=4.0,
                      mec=styles[m][0], mfc=styles[m][2], mew=.8, label=DISPLAY.get(m, m))
               for m in ("Frozen", "Ranking", "IS")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.54, 1.0),
               ncol=3, frameon=False, fontsize=6.1, handlelength=.8,
               columnspacing=1.15, handletextpad=.4, borderaxespad=0)
    save(fig, "answer_dependency")

def retest_figure(data):
    """Compact v6.3 connected-dot design, with every estimate retained.

    Thin segments are seed ranges; translucent bars are the range of seed
    means across the five backdoor processing conditions. They are distinct.
    """
    fig = plt.figure(figsize=(5.5, 2.35))
    axes = [fig.add_axes([left, .18, .275, .60])
            for left in (.042, .370, .698)]
    headings = [("a  Typography", "Word interaction"),
                ("b  Color binding", "Cross-effect"),
                ("c  Backdoors", "Trigger interaction")]
    for ax, (title, subtitle) in zip(axes, headings):
        ax.text(0, 1.19, title, transform=ax.transAxes, fontsize=6.8,
                fontweight="semibold", ha="left", va="bottom")
        ax.text(0, 1.07, subtitle, transform=ax.transAxes, fontsize=6.0,
                color=MUTED, ha="left", va="bottom")
        ax.set_ylim(7.72, -.55)
        ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=6.0, length=2, pad=2)
        ax.xaxis.grid(True, color=GRID, linewidth=.45)
        ax.set_axisbelow(True)

    def pair(ax, y, ranking, steering, ranges=None):
        ax.plot([ranking[0], steering[0]], [y, y],
                color="#BDC5CA", lw=.65, zorder=2)
        if ranges is not None:
            for vals, color in zip(ranges, (COL["Ranking"], COL["IS"])):
                ax.plot([min(vals), max(vals)], [y, y], color=color,
                        lw=4.2, alpha=.22, solid_capstyle="butt", zorder=1)
        for point, method in ((ranking, "Ranking"), (steering, "IS")):
            ax.plot([point[1], point[2]], [y, y],
                    color=COL[method], lw=.9, zorder=3)
            ax.plot(point[0], y, "o", ms=3.0, color=COL[method],
                    mfc="white" if method == "Ranking" else COL[method],
                    mew=.85, zorder=4)

    def label(ax, y, name):
        ax.text(.014, y-.27, name, transform=ax.get_yaxis_transform(),
                fontsize=6.0, ha="left", va="bottom",
                bbox={"facecolor": "white", "edgecolor": "none",
                      "pad": .1, "alpha": .93})

    short = {"38 unseen classes": "Unseen classes",
             "Fresh images, seen classes": "Additional images / seen classes",
             "Fresh images, unseen classes": "Additional images / unseen classes",
             "Photographed attacks (SCAM)": "Photographed attacks (SCAM)",
             "New object size/position": "New size / position",
             "New color pairings": "New color pairs",
             "New caption wording": "New caption wording",
             "OpenAI L/14 repair": "OpenAI L/14",
             "SigLIP B/16 repair": "SigLIP B/16"}
    for ax, key in zip(axes[:2], ("typography", "routing")):
        for index, (name, ranking, steering) in enumerate(data[key]):
            y = index+.12
            label(ax, y, short.get(name, name))
            pair(ax, y, ranking, steering)

    axes[0].set_xlim(25, 67)
    axes[0].set_xticks([25, 35, 45, 55, 65])
    routing_points = [x for row in data['routing'] for triple in row[1:] for x in triple]
    routing_lo = min(-55., 10*np.floor((min(routing_points)-2)/10))
    routing_hi = max(15., 10*np.ceil((max(routing_points)+2)/10))
    axes[1].set_xlim(routing_lo, routing_hi)
    axes[1].xaxis.set_major_locator(MaxNLocator(4))
    axes[1].axvspan(routing_lo, 0, color="#FAF0E7", zorder=0)
    axes[1].axvline(0, color="#859099", lw=.65, zorder=1)
    CHECKS.append(dict(test='all_routing_means_and_seed_ranges_visible',
                       passed=all(routing_lo<=x<=routing_hi for x in routing_points),
                       axis_limits=[routing_lo,routing_hi]))
    ax = axes[2]
    for index, (name, ranking_proc, steering_proc, ranking, steering) in enumerate(data["backdoor"]):
        y = .25 + index*2.62
        label(ax, y, name + " trigger")
        pair(ax, y, ranking, steering, (ranking_proc, steering_proc))
    ax.set_xlim(55, 90)
    ax.set_xticks([60, 70, 80, 90])
    ax.text(.02, .035, "Pale bars: processing range", transform=ax.transAxes,
            fontsize=6.0, color=MUTED, ha="left", va="bottom")
    handles = [
        Line2D([], [], marker="o", ls="", ms=3.4, color=COL["Ranking"],
               mfc="white", mew=.85, label="Ranking"),
        Line2D([], [], marker="o", ls="", ms=3.4, color=COL["IS"], label=DISPLAY["IS"]),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.51, 1.0),
               ncol=2, frameon=False, fontsize=6.2, handlelength=.8,
               handletextpad=.45, columnspacing=1.9, borderaxespad=0)
    fig.text(.51, .025, "Reduction in unwanted interaction magnitude (%)",
             ha="center", va="bottom", fontsize=6.3)
    save(fig, "retest")

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('reproduced/figures'));a=p.parse_args()
    HERE=a.out.parent
    # save() writes HERE/figures; redirect its destination for arbitrary --out.
    def save(fig,name):
        a.out.mkdir(parents=True,exist_ok=True)
        for ext in ('pdf','svg','png'):
            metadata={'CreationDate':None,'ModDate':None} if ext=='pdf' else {'Date':None} if ext=='svg' else None
            fig.savefig(a.out/(name+'.'+ext),dpi=300,metadata=metadata)
        plt.close(fig)
    data=json.loads((ROOT/'results/plot_data.json').read_text());style()
    answer_figure(data['answer_dependency']);distributions_figure(data['interaction_distributions']);retest_figure(data['retest'])
    print('Rendered paper Figures 4 (answer_dependency), 5 (interaction_distributions), and 6 (retest) to',a.out)
    (a.out/'checks.json').write_text(json.dumps(CHECKS,indent=2)+'\n')
