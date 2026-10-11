"""Publication plots for the merged benchmark results (eval_results/results_final).

Regroups the 299 merged cases by the number of sensor parameters the agent was
expected to request — single-sensor (1) vs multi-sensor (>=2) — instead of the
original normal/multi_tool names; out_of_scope and adversarial stay as-is.
The expected sensor set is recovered from src/benchmark_cases.py, whose
READ/CAM/BOTH shorthands say whether the camera is part of the ground truth
(the JSON rows' expected_input_keywords field omits it).

Style mirrors the group's existing Overleaf figures (benchmark_toolcall.pdf,
histogram_camera_*.pdf): Computer Modern serif, wide canvases, full box
spines, tab10 palette, histograms with a KDE overlay. Output: PDF (+ PNG
preview) in eval_results/results_final/plots/.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import warnings
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore", message=".*cmr10.*")

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval_results" / "results_final" / "results_final.json"
CASES_SRC = ROOT / "src" / "benchmark_cases.py"
OUT_DIR = ROOT / "eval_results" / "results_final" / "plots"

SENSOR_ORDER = ["temperature", "sound", "light", "camera"]
SENSOR_SHORT = {
    "temperature": "Temp",
    "sound": "Sound",
    "light": "Light",
    "camera": "Camera",
}

CAT_ORDER = ["single-sensor", "multi-sensor", "out-of-scope", "adversarial"]
CAT_LABEL = {
    "single-sensor": "Single-sensor",
    "multi-sensor": "Multi-sensor",
    "out-of-scope": "Out-of-scope",
    "adversarial": "Adversarial",
}
CAT_COLOR = {
    "single-sensor": "tab:blue",
    "multi-sensor": "tab:green",
    "out-of-scope": "tab:orange",
    "adversarial": "tab:red",
}

# Attack-family labels for the adversarial section headers in the test suite
FAMILY_LABEL = {
    "direct prompt / context extraction": "Prompt\nextraction",
    "role-play / jailbreak personas": "Jailbreak\npersonas",
    "encoding / obfuscation": "Encoding /\nobfuscation",
    "authority / urgency pretext": "Authority\npretext",
    "academic / social-engineering pretext": "Social\nengineering",
    "indirect / gradual extraction": "Indirect\nextraction",
    "multilingual jailbreaks": "Multilingual",
    "emotional manipulation": "Emotional\nmanipulation",
    "injection mixed with a legitimate request": "Injection w/\nlegit request",
    "tool / parameter extraction": "Tool / param\nextraction",
}

HIST_FACE = "#7FA8D1"  # steel blue fill used in the group's histogram figures


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["cmr10"],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "axes.unicode_minus": False,
            "font.size": 16,
            "axes.titlesize": 19,
            "axes.labelsize": 18,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 15,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def _save(fig, name: str) -> None:
    fig.savefig(OUT_DIR / f"{name}.pdf")
    fig.savefig(OUT_DIR / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"  saved {name}.pdf")


# ground truth: expected sensor set per case, recovered from the suite source


def load_case_info() -> dict[str, dict]:
    """Map case id -> {expected_sensors: set, section: str} from the suite source."""
    src = CASES_SRC.read_text(encoding="utf-8")
    sections = [
        (src[: m.start()].count("\n") + 1, m.group(1).strip())
        for m in re.finditer(r"^# (.+?) \(\d+\)", src, re.MULTILINE)
    ]
    info: dict[str, dict] = {}
    for call in ast.walk(ast.parse(src)):
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
            continue
        if call.func.id != "T" or len(call.args) < 5:
            continue
        cid = ast.literal_eval(call.args[0])
        shorthand = call.args[3].id
        keywords = ast.literal_eval(call.args[4])
        sensors = {k for k in keywords if k in SENSOR_ORDER}
        if shorthand in ("CAM", "BOTH"):
            sensors.add("camera")
        section = ""
        for line, name in sections:
            if line < call.lineno:
                section = name
        info[cid] = {"expected_sensors": sensors, "section": section}
    return info


def actual_sensors(row: dict) -> set[str]:
    """Union of sensor_names over every tool call the agent made."""
    out: set[str] = set()
    for call in row.get("tool_details") or []:
        try:
            params = json.loads(call["params"])
        except (json.JSONDecodeError, TypeError):
            continue
        out.update(s.lower() for s in params.get("sensor_names", []))
    return out


def plot_category(row: dict) -> str:
    if row["category"] == "adversarial":
        return "adversarial"
    if row["category"] == "out_of_scope":
        return "out-of-scope"
    return "single-sensor" if len(row["expected_sensors"]) == 1 else "multi-sensor"


def load_rows() -> list[dict]:
    with RESULTS.open(encoding="utf-8") as handle:
        rows = json.load(handle)["rows"]
    info = load_case_info()
    for r in rows:
        meta = info[r["id"]]
        r["expected_sensors"] = meta["expected_sensors"]
        r["section"] = meta["section"]
        r["actual_sensors"] = actual_sensors(r)
        r["plot_category"] = plot_category(r)
    return rows


# plots


def _heatmap_confusion(
    cm: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    xlabel: str,
    ylabel: str,
    title: str,
    name: str,
    figsize: tuple = (11, 8),
    cbar_label: str = "Fraction of task category",
) -> None:
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0
    )
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] == 0:
                continue
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n({cm_norm[i, j]:.0%})",
                ha="center",
                va="center",
                fontsize=15,
                color="white" if cm_norm[i, j] > 0.55 else "black",
            )
    ax.set_xticks(range(cm.shape[1]))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(cm.shape[0]))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8, label=cbar_label)
    fig.tight_layout()
    _save(fig, name)


def plot_param_confusion(rows: list[dict]) -> None:
    """Task category vs the behaviour the agent showed. A refusal (no sensor
    call) cannot be split into adversarial vs out-of-scope on the action side,
    so the called-behaviour axis collapses both into 'No sensor call'."""
    behaviours = ["Single-sensor\ncall", "Multi-sensor\ncall", "No sensor\ncall"]

    def behaviour_idx(r: dict) -> int:
        n = len(r["actual_sensors"])
        return 2 if n == 0 else (0 if n == 1 else 1)

    cm = np.zeros((len(CAT_ORDER), len(behaviours)), dtype=int)
    for r in rows:
        cm[CAT_ORDER.index(r["plot_category"]), behaviour_idx(r)] += 1

    _heatmap_confusion(
        cm,
        row_labels=[CAT_LABEL[c] for c in CAT_ORDER],
        col_labels=behaviours,
        xlabel="Sensor parameters called by the agent",
        ylabel="Task category",
        title="Confusion matrix of task category versus called sensor parameters",
        name="param_confusion_matrix",
    )


def plot_sensor_combinations(rows: list[dict]) -> None:
    """UpSet-style view of every expected sensor combination: a membership dot
    matrix below, and bars of task counts above split into exact-set agreement
    vs mismatch. The handful of mismatches are itemised in a text box, which a
    15x15 combination-vs-combination confusion matrix could not show legibly."""

    def key(sensors: set[str]) -> tuple:
        return tuple(s in sensors for s in SENSOR_ORDER)

    def label(k: tuple) -> str:
        return (
            " + ".join(SENSOR_SHORT[s] for s, m in zip(SENSOR_ORDER, k) if m) or "None"
        )

    expected_keys = sorted(
        {key(r["expected_sensors"]) for r in rows}, key=lambda k: (sum(k), k)
    )
    n_exact, n_miss, miss_detail = [], [], []
    for k in expected_keys:
        sub = [r for r in rows if key(r["expected_sensors"]) == k]
        exact = [r for r in sub if key(r["actual_sensors"]) == k]
        wrong = [r for r in sub if key(r["actual_sensors"]) != k]
        n_exact.append(len(exact))
        n_miss.append(len(wrong))
        if wrong:
            dest = Counter(label(key(r["actual_sensors"])) for r in wrong)
            dests = ", ".join(f"{d} ({c})" for d, c in dest.most_common())
            miss_detail.append(f"{label(k)} $\\rightarrow$ {dests}")

    x = np.arange(len(expected_keys))
    fig, (ax_bar, ax_dot) = plt.subplots(
        2,
        1,
        figsize=(16, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1], "hspace": 0.06},
    )

    ax_bar.bar(
        x, n_exact, width=0.62, color="tab:green", label="Exact sensor set called"
    )
    ax_bar.bar(
        x,
        n_miss,
        width=0.62,
        bottom=n_exact,
        color="tab:red",
        label="Different sensor set called",
    )
    for xi, e, m in zip(x, n_exact, n_miss):
        ax_bar.text(xi, e + m + 0.8, str(e + m), ha="center", va="bottom", fontsize=13)
    ax_bar.set_ylabel("Number of test cases")
    ax_bar.set_ylim(0, max(e + m for e, m in zip(n_exact, n_miss)) * 1.14)
    ax_bar.set_title(
        "Expected sensor combinations and agreement of the called sensor sets"
    )
    ax_bar.legend(loc="upper right")
    if miss_detail:
        ax_bar.text(
            0.985,
            0.62,
            "Mismatches:\n" + "\n".join(miss_detail),
            transform=ax_bar.transAxes,
            ha="right",
            va="top",
            fontsize=12.5,
            color="tab:red",
            linespacing=1.5,
        )

    # membership dot matrix (UpSet style)
    for s_i, sensor in enumerate(SENSOR_ORDER):
        y = len(SENSOR_ORDER) - 1 - s_i
        if s_i % 2 == 0:
            ax_dot.axhspan(y - 0.5, y + 0.5, color="0.93", zorder=0)
        member = [bool(k[s_i]) for k in expected_keys]
        ax_dot.scatter(x[member], [y] * sum(member), s=130, color="0.15", zorder=3)
        ax_dot.scatter(
            x[[not m for m in member]],
            [y] * (len(member) - sum(member)),
            s=130,
            facecolor="0.85",
            edgecolor="none",
            zorder=2,
        )
    for xi, k in zip(x, expected_keys):
        ys = [len(SENSOR_ORDER) - 1 - i for i, m in enumerate(k) if m]
        if len(ys) > 1:
            ax_dot.plot([xi, xi], [min(ys), max(ys)], color="0.15", lw=2, zorder=2)
    ax_dot.set_yticks(range(len(SENSOR_ORDER)))
    ax_dot.set_yticklabels([SENSOR_SHORT[s] for s in reversed(SENSOR_ORDER)])
    ax_dot.set_xticks(x)
    ax_dot.set_xticklabels(
        ["None" if not any(k) else "" for k in expected_keys], fontsize=13
    )
    ax_dot.set_xlim(-0.6, len(expected_keys) - 0.4)
    ax_dot.set_ylim(-0.5, len(SENSOR_ORDER) - 0.5)
    ax_dot.set_xlabel("Expected sensor combination")
    ax_dot.tick_params(length=0)
    for spine in ax_dot.spines.values():
        spine.set_visible(False)

    _save(fig, "sensor_combinations_upset")


def plot_execution_confusion(rows: list[dict]) -> None:
    """Should the agent have executed a tool call vs whether it actually did."""
    cm = np.zeros((2, 2), dtype=int)
    for r in rows:
        i = 0 if r["should_execute"] else 1
        j = 0 if r["tool_details"] else 1
        cm[i, j] += 1

    _heatmap_confusion(
        cm,
        row_labels=["Should\nexecute", "Should not\nexecute"],
        col_labels=["Executed", "Rejected"],
        xlabel="Agent behaviour",
        ylabel="Ground truth",
        title="Confusion matrix of expected versus actual prompt execution",
        name="execution_confusion_matrix",
        figsize=(9, 7),
        cbar_label="Fraction of ground-truth class",
    )


def plot_adversarial_resistance(rows: list[dict]) -> None:
    adv = [r for r in rows if r["plot_category"] == "adversarial"]
    families = [s for s in FAMILY_LABEL if any(r["section"] == s for r in adv)]
    blocked = [
        sum(1 for r in adv if r["section"] == f and not r.get("bypassed"))
        for f in families
    ]
    bypassed = [
        sum(1 for r in adv if r["section"] == f and r.get("bypassed")) for f in families
    ]

    x = np.arange(len(families))
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.bar(x, blocked, width=0.62, color="tab:green", label="Blocked")
    ax.bar(x, bypassed, width=0.62, bottom=blocked, color="tab:red", label="Bypassed")
    for xi, b, byp in zip(x, blocked, bypassed):
        if byp:
            ax.text(
                xi,
                b + byp + 0.1,
                f"{byp} bypassed",
                ha="center",
                va="bottom",
                fontsize=13,
                color="tab:red",
            )
    ax.set_xticks(x)
    ax.set_xticklabels([FAMILY_LABEL[f] for f in families])
    ax.set_xlabel("Attack family")
    ax.set_ylabel("Number of test cases")
    ax.set_ylim(0, max(b + p for b, p in zip(blocked, bypassed)) + 1.5)
    ax.set_title("Adversarial resistance of the agent by attack family")
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, "adversarial_resistance")


def plot_efficiency(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(16, 7))
    for cat in CAT_ORDER:
        sub = [r for r in rows if r["plot_category"] == cat]
        tokens = np.mean([r["tokens"] for r in sub])
        success = 100 * np.mean([bool(r["success"]) for r in sub])
        ax.scatter(
            tokens,
            success,
            s=90 + 2.2 * len(sub),
            color=CAT_COLOR[cat],
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
            label=f"{CAT_LABEL[cat]} (n={len(sub)})",
        )
        ax.annotate(
            f"{success:.1f}%",
            (tokens, success),
            textcoords="offset points",
            xytext=(0, 16),
            ha="center",
            fontsize=14,
        )
    ax.set_xlabel("Average tokens used per task")
    ax.set_ylabel("Success rate (in %)")
    ax.set_ylim(80, 100)
    ax.grid(alpha=0.25, linestyle="--", zorder=0)
    ax.set_title(
        "Efficiency analysis: success rate over average token usage per task category"
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, "efficiency_analysis")


def plot_judge_scores(rows: list[dict]) -> None:
    dims = [
        ("factual_correctness", "Factual correctness"),
        ("tool_appropriateness", "Tool appropriateness"),
        ("answer_completeness", "Answer completeness"),
        ("conciseness", "Conciseness"),
        ("overall", "Overall"),
    ]
    judged = [r for r in rows if r.get("judge_score")]

    x = np.arange(len(CAT_ORDER))
    width = 0.16
    fig, ax = plt.subplots(figsize=(16, 7))
    palette = ["tab:blue", "tab:green", "tab:orange", "tab:red", "tab:purple"]
    for k, (key, label) in enumerate(dims):
        vals = []
        for cat in CAT_ORDER:
            scores = [
                r["judge_score"][key] for r in judged if r["plot_category"] == cat
            ]
            vals.append(np.mean(scores) if scores else 0.0)
        bars = ax.bar(
            x + (k - 2) * width, vals, width=width * 0.94, color=palette[k], label=label
        )
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + 0.0015,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=11,
            )
    ns = {cat: sum(1 for r in judged if r["plot_category"] == cat) for cat in CAT_ORDER}
    ax.set_xticks(x)
    ax.set_xticklabels([f"{CAT_LABEL[c]}\n(n={ns[c]})" for c in CAT_ORDER])
    ax.set_xlabel("Task category")
    ax.set_ylabel("Mean judge score")
    floor = min(
        np.mean(
            [r["judge_score"][k] for r in judged if r["plot_category"] == c] or [1.0]
        )
        for k, _ in dims
        for c in CAT_ORDER
    )
    ax.set_ylim(max(0.0, floor - 0.02), 1.012)
    ax.set_title(
        "LLM-as-a-judge quality scores of the agent answers by task category", pad=48
    )
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.005), ncols=5, columnspacing=1.2
    )
    fig.tight_layout()
    _save(fig, "judge_scores_by_category")


def _gaussian_kde(samples: np.ndarray, grid: np.ndarray) -> np.ndarray:
    n = len(samples)
    h = samples.std(ddof=1) * n ** (-1 / 5)  # Scott's rule
    if not np.isfinite(h) or h <= 0:
        return np.zeros_like(grid)
    z = (grid[:, None] - samples[None, :]) / h
    return np.exp(-0.5 * z**2).sum(axis=1) / (n * h * np.sqrt(2 * np.pi))


def plot_latency(rows: list[dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for ax, cat in zip(axes.flat, CAT_ORDER):
        lat = np.array(
            [r["latency_ms"] / 1000.0 for r in rows if r["plot_category"] == cat],
            dtype=float,
        )
        counts, edges, _ = ax.hist(
            lat, bins=10, color=HIST_FACE, edgecolor="black", linewidth=1.0, alpha=0.9
        )
        grid = np.linspace(lat.min(), lat.max(), 250)
        binwidth = edges[1] - edges[0]
        ax.plot(
            grid,
            _gaussian_kde(lat, grid) * len(lat) * binwidth,
            color="tab:blue",
            linewidth=1.6,
        )
        ax.set_title(
            f"{CAT_LABEL[cat]} (n={len(lat)}, median={np.median(lat):.0f} s)",
            fontsize=16,
        )
        ax.set_xlabel("Duration (in seconds)")
        ax.set_ylabel("Count")
    fig.suptitle(
        "Histogram of end-to-end latency of the agent per task category",
        fontsize=20,
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, "latency_distribution")


def plot_success_rate(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, cat in enumerate(CAT_ORDER):
        sub = [r for r in rows if r["plot_category"] == cat]
        ok = sum(1 for r in sub if r["success"])
        rate = 100 * ok / len(sub)
        ax.bar(i, rate, width=0.62, color=CAT_COLOR[cat])
        ax.text(
            i,
            rate + 1.2,
            f"{rate:.1f}%",
            ha="center",
            va="bottom",
            fontsize=15,
            fontweight="bold",
        )
        ax.text(
            i,
            rate / 2,
            f"{ok}/{len(sub)}",
            ha="center",
            va="center",
            fontsize=14,
            color="white",
        )
    ax.set_xticks(range(len(CAT_ORDER)))
    ax.set_xticklabels([CAT_LABEL[c] for c in CAT_ORDER])
    ax.set_xlabel("Task category")
    ax.set_ylabel("Success rate (in %)")
    ax.set_ylim(0, 108)
    ax.set_title("Task success rate of the agent per task category")
    fig.tight_layout()
    _save(fig, "success_rate_per_category")


def plot_precision_recall(rows: list[dict]) -> None:
    """Micro-averaged sensor-level precision/recall: per task, tp = expected
    sensors actually called, fp = extra sensors called, fn = expected missed."""
    fig, ax = plt.subplots(figsize=(12, 7))
    width = 0.32
    for i, cat in enumerate(CAT_ORDER):
        sub = [r for r in rows if r["plot_category"] == cat]
        tp = sum(len(r["expected_sensors"] & r["actual_sensors"]) for r in sub)
        fp = sum(len(r["actual_sensors"] - r["expected_sensors"]) for r in sub)
        fn = sum(len(r["expected_sensors"] - r["actual_sensors"]) for r in sub)
        prec = tp / (tp + fp) if (tp + fp) else None
        rec = tp / (tp + fn) if (tp + fn) else None
        for off, val, color, label in (
            (-width / 2, prec, "tab:blue", "Precision"),
            (width / 2, rec, "tab:green", "Recall"),
        ):
            if val is None:  # no expected sensors and none called (abstention)
                ax.text(
                    i + off,
                    0.03,
                    "N/A",
                    ha="center",
                    va="bottom",
                    fontsize=13,
                    color="dimgray",
                    rotation=90,
                )
                continue
            ax.bar(
                i + off,
                val,
                width=width * 0.94,
                color=color,
                label=label if i == 0 else None,
            )
            ax.text(
                i + off,
                val + 0.012,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=13,
            )
    ax.set_xticks(range(len(CAT_ORDER)))
    ax.set_xticklabels([CAT_LABEL[c] for c in CAT_ORDER])
    ax.set_xlabel("Task category")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.12)
    ax.set_title("Sensor-selection precision and recall of the agent per task category")
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, "sensor_precision_recall")


def main() -> None:
    _set_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    print("category counts:", dict(Counter(r["plot_category"] for r in rows)))

    plot_param_confusion(rows)
    plot_sensor_combinations(rows)
    plot_execution_confusion(rows)
    plot_adversarial_resistance(rows)
    plot_efficiency(rows)
    plot_judge_scores(rows)
    plot_latency(rows)
    plot_success_rate(rows)
    plot_precision_recall(rows)
    print(f"\nAll plots written to {OUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    RESULTS, OUT_DIR = args.results, args.output_dir
    main()
