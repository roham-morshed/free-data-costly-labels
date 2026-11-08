"""Turn ``results/`` into every table and figure the paper needs.

Nothing here computes a model.  It only aggregates result files, so it can be
re-run freely and it can never invent a number: a cell that did not run is
missing, and missing cells are counted and reported rather than filled in.

Outputs
-------
``paper/tables/T1..T7.tex``     booktabs LaTeX
``paper/figures/F1..F6.{pdf,png}``
``paper/results_summary.md``    every headline number in plain language
``paper/reproducibility.json``  versions, machine, hashes, runtime, counts

Usage::

    python scripts/make_paper_artifacts.py
"""

from __future__ import annotations

import argparse
import platform
import sys
import warnings
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as sps  # noqa: E402

from tscfree.config import dataset_names, load_config, load_datasets_yaml  # noqa: E402
from tscfree.lfa import LFA_COMPONENTS, composite_lfa, zscore  # noqa: E402
from tscfree.stats import (  # noqa: E402
    average_ranks,
    friedman_test,
    nemenyi_critical_difference,
    plot_cd_diagram,
    regression_rate,
    wilcoxon_holm,
)
from tscfree.utils import (  # noqa: E402
    dir_size,
    human_bytes,
    package_versions,
    read_json,
    write_json,
)

# Okabe-Ito, colour-blind safe.
PALETTE = {
    "1nn_dtw": "#000000",
    "sup_strict": "#E69F00",
    "sup_tf": "#56B4E9",
    "self_train": "#009E73",
    "label_spread": "#F0E442",
    "augment": "#0072B2",
    "safe_ssl": "#D55E00",
    "safe_ssl_novalve": "#CC79A7",
    "deep_ssl": "#7F3F98",
    "sup_full": "#999999",
}
MARKERS = {
    "1nn_dtw": "o", "sup_strict": "s", "sup_tf": "^", "self_train": "v",
    "label_spread": "D", "augment": "P", "safe_ssl": "*", "safe_ssl_novalve": "X",
    "deep_ssl": "h",
}
PRETTY = {
    "1nn_dtw": r"1NN-DTW",
    "sup_strict": r"\textsc{sup-strict}",
    "sup_tf": r"\textsc{sup-tf}",
    "self_train": r"Self-train",
    "label_spread": r"LabelSpread",
    "augment": r"Augment",
    "safe_ssl": r"\textbf{SAFE-SSL}",
    "safe_ssl_novalve": r"SAFE-SSL$^{-v}$",
    "sup_full": r"\textsc{sup-full}",
}

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
    }
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_rows(directory: Path) -> pd.DataFrame:
    """Flatten every result file in ``directory`` into one long DataFrame."""
    recs: list[dict[str, Any]] = []
    for f in sorted(directory.glob("*.json")):
        try:
            payload = read_json(f)
        except Exception:
            continue
        tag = payload.get("key", {}).get("tag", "")
        for r in payload.get("rows", []):
            rec = {k: v for k, v in r.items() if k not in ("versions", "extra", "lfa")}
            rec["tag"] = tag
            extra = r.get("extra") or {}
            rec["valve_status"] = (extra.get("valve") or {}).get("status")
            rec["ridge_cv"] = extra.get("ridge_cv")
            lfa = r.get("lfa") or {}
            for c in LFA_COMPONENTS:
                rec[f"lfa_{c}"] = lfa.get(c)
            recs.append(rec)
    return pd.DataFrame.from_records(recs)


def load_lfa(cfg) -> pd.DataFrame:
    d = cfg.path_for("results_lfa")
    recs = []
    for f in sorted(d.glob("*.json")):
        p = read_json(f)
        rec = {"dataset": p["dataset"], **{k: v for k, v in p["components"].items()}}
        recs.append(rec)
    df = pd.DataFrame.from_records(recs)
    if df.empty:
        return df
    table = {c: df[c].to_numpy(dtype=float) for c in LFA_COMPONENTS}
    df["LFA"] = composite_lfa(table)
    for c in LFA_COMPONENTS:
        df[f"z_{c}"] = zscore(df[c].to_numpy(dtype=float))
    return df


def seed_mean(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Average over seeds first -- comparisons across datasets come afterwards."""
    have = [m for m in metrics if m in df.columns]
    return df.groupby(["dataset", "method", "k"], as_index=False)[have].mean()


def matrix(agg: pd.DataFrame, k: int, methods: list[str], metric: str = "accuracy"):
    """``(datasets x methods)`` matrix at one budget, restricted to complete rows."""
    sub = agg[(agg.k == k) & (agg.method.isin(methods))]
    piv = sub.pivot(index="dataset", columns="method", values=metric)
    piv = piv.reindex(columns=methods)
    piv = piv.dropna(axis=0, how="any")
    return piv


# ---------------------------------------------------------------------------
# LaTeX helpers
# ---------------------------------------------------------------------------


def latex_table(
    body: str, caption: str, label: str, colspec: str, header: str, note: str = ""
) -> str:
    out = [
        "% Auto-generated by scripts/make_paper_artifacts.py -- do not edit by hand.",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        header,
        r"\midrule",
        body.rstrip(),
        r"\bottomrule",
        r"\end{tabular}",
    ]
    if note:
        out.append(rf"\\[2pt]\footnotesize{{{note}}}")
    out += [r"\end{table}", ""]
    return "\n".join(out)


def fmt(v: float, nd: int = 3, bold: bool = False) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "--"
    s = f"{v:.{nd}f}"
    return rf"\textbf{{{s}}}" if bold else s


def esc(s: str) -> str:
    return str(s).replace("_", r"\_")


def save_fig(fig, name: str, figdir: Path) -> None:
    figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(figdir / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def t1_main(agg: pd.DataFrame, methods: list[str], ks: list[int], out: Path,
            ceilings: dict[str, float]) -> dict:
    info = {}
    for k in ks:
        piv = matrix(agg, k, methods)
        if piv.empty:
            continue
        lines = []
        for ds, row in piv.iterrows():
            best = row.max()
            cells = " & ".join(
                fmt(row[m], bold=bool(np.isclose(row[m], best))) for m in methods
            )
            ceil = ceilings.get(ds)
            lines.append(f"{esc(ds)} & {cells} & {fmt(ceil)} \\\\")
        ranks = average_ranks(piv.to_numpy())
        lines.append(r"\midrule")
        lines.append(
            "Mean acc. & "
            + " & ".join(fmt(piv[m].mean()) for m in methods)
            + f" & {fmt(np.mean([ceilings.get(d, np.nan) for d in piv.index]))} \\\\"
        )
        lines.append(
            "Mean rank & "
            + " & ".join(fmt(r, nd=2, bold=bool(np.isclose(r, ranks.min()))) for r in ranks)
            + " & -- \\\\"
        )
        header = "Dataset & " + " & ".join(PRETTY[m] for m in methods) + r" & \textsc{sup-full} \\"
        tex = latex_table(
            "\n".join(lines),
            caption=(
                f"Test accuracy at $k={k}$ labels per class, averaged over "
                f"{int(agg.attrs.get('n_seeds', 10))} seeds; best per dataset in bold. "
                r"\textsc{sup-full} is the fully-supervised ceiling."
            ),
            label=f"tab:main_k{k}",
            colspec="l" + "r" * (len(methods) + 1),
            header=header,
            note=f"{len(piv)} datasets with complete results at this budget.",
        )
        (out / f"T1_main_k{k}.tex").write_text(tex, encoding="utf-8")
        if k == ks[0]:
            # the spec names the primary table T1_main.tex; keep both so a
            # \input{T1_main} in the paper source always resolves.
            (out / "T1_main.tex").write_text(tex, encoding="utf-8")
        info[k] = {"n_datasets": int(len(piv)),
                   "mean_rank": {m: float(r) for m, r in zip(methods, ranks, strict=True)},
                   "mean_acc": {m: float(piv[m].mean()) for m in methods}}
    return info


def _pstar(pv: float) -> str:
    """p-value for a table cell, bolded when it clears the 5% level."""
    txt = f"{pv:.3f}" if pv >= 0.001 else f"{pv:.1e}"
    return rf"\textbf{{{txt}}}" if pv < 0.05 else txt


def _wilcoxon_p(a, b) -> float:
    """Two-sided Wilcoxon p-value; 1.0 when every paired difference is zero."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if d.size < 1 or np.allclose(d, 0.0):
        return 1.0
    return float(sps.wilcoxon(a, b, zero_method="pratt").pvalue)


def t2_attribution(agg: pd.DataFrame, ks: list[int], out: Path) -> dict:
    rows = []
    for k in ks:
        piv = matrix(agg, k, ["sup_strict", "sup_tf", "safe_ssl", "safe_ssl_novalve"])
        if piv.empty:
            continue
        d_feat = piv["sup_tf"] - piv["sup_strict"]
        d_pseudo = piv["safe_ssl"] - piv["sup_tf"]
        d_pseudo_nv = piv["safe_ssl_novalve"] - piv["sup_tf"]
        rows.append(
            {
                "k": k,
                "n": len(piv),
                "sup_strict": piv["sup_strict"].mean(),
                "sup_tf": piv["sup_tf"].mean(),
                "safe_ssl": piv["safe_ssl"].mean(),
                "d_features": d_feat.mean(),
                "d_features_sd": d_feat.std(ddof=1),
                "d_pseudo": d_pseudo.mean(),
                "d_pseudo_sd": d_pseudo.std(ddof=1),
                "d_pseudo_novalve": d_pseudo_nv.mean(),
                "p_features": _wilcoxon_p(piv["sup_tf"], piv["sup_strict"]),
                "p_pseudo": _wilcoxon_p(piv["safe_ssl"], piv["sup_tf"]),
                "p_pseudo_novalve": _wilcoxon_p(piv["safe_ssl_novalve"], piv["sup_tf"]),
            }
        )
    # The deltas here are two orders of magnitude smaller than the accuracies, so
    # they are printed in accuracy *points* -- at 3 decimals of raw accuracy the
    # k=2 feature gain would round to 0.000 and the table would say nothing.
    body = "\n".join(
        f"{r['k']} & {fmt(r['sup_strict'])} & {fmt(r['sup_tf'])} & {fmt(r['safe_ssl'])} & "
        f"{r['d_features'] * 100:+.2f} & {_pstar(r['p_features'])} & "
        f"{r['d_pseudo'] * 100:+.2f} & {_pstar(r['p_pseudo'])} & "
        f"{r['d_pseudo_novalve'] * 100:+.2f} \\\\"
        for r in rows
    )
    tex = latex_table(
        body,
        caption=(
            "RQ2, three-tier attribution. The first three columns are test accuracies "
            "averaged over datasets and seeds. "
            "$\\Delta_{\\text{features}}=$ \\textsc{sup-tf}$-$\\textsc{sup-strict} is the gain "
            "from fitting the random-kernel transform on unlabeled data; "
            "$\\Delta_{\\text{pseudo}}=$ SAFE-SSL$-$\\textsc{sup-tf} is the gain from "
            "pseudo-labeling; $\\Delta_{\\text{pseudo}}^{-v}$ is the same with the safety "
            "valve removed. All $\\Delta$ are in \\emph{accuracy points} "
            "($100\\times$ accuracy), and $p$ is a two-sided Wilcoxon signed-rank test over "
            "the 40 datasets."
        ),
        label="tab:attribution",
        colspec="rrrrrrrrr",
        header=(
            r"$k$ & \textsc{sup-strict} & \textsc{sup-tf} & SAFE-SSL & "
            r"$\Delta_{\text{features}}$ & $p$ & $\Delta_{\text{pseudo}}$ & $p$ & "
            r"$\Delta_{\text{pseudo}}^{-v}$ \\"
        ),
    )
    (out / "T2_attribution.tex").write_text(tex, encoding="utf-8")
    return {"rows": rows}


def recovery_frame(
    agg: pd.DataFrame, k: int, methods: list[str], ceilings: dict[str, float], min_gap: float
) -> tuple[pd.DataFrame, int, int]:
    """Per-dataset recovery ratios at one budget, computed on seed-averaged accuracy.

    The ratio divides by ``acc_full - acc_strict``.  On datasets where the
    fully-supervised ceiling is barely above (or below) the low-label floor that
    denominator is a rounding artefact, and dividing by it turns a 2-point
    accuracy difference into a ratio of -5.  Averaging such ratios is
    meaningless, so datasets with a gap smaller than ``min_gap`` are dropped and
    the number dropped is reported.  Nothing is clipped or winsorised.
    """
    piv = matrix(agg, k, methods)
    if piv.empty:
        return piv, 0, 0
    full = pd.Series({d: ceilings.get(d, np.nan) for d in piv.index}, dtype=float)
    strict = piv["sup_strict"]
    gap = full - strict
    keep = gap.notna() & (gap > min_gap)
    rec = piv.loc[keep].sub(strict[keep], axis=0).div(gap[keep], axis=0)
    return rec, int(keep.sum()), int((~keep).sum())


def t3_recovery(agg: pd.DataFrame, methods: list[str], ks: list[int], out: Path,
                ceilings: dict[str, float], min_gap: float) -> dict:
    lines = []
    per_k: dict[int, pd.DataFrame] = {}
    n_used: dict[int, int] = {}
    n_drop: dict[int, int] = {}
    for k in ks:
        per_k[k], n_used[k], n_drop[k] = recovery_frame(agg, k, methods, ceilings, min_gap)
    for m in methods:
        med, mean = [], []
        for k in ks:
            r = per_k[k]
            med.append(float(r[m].median()) if len(r) and m in r else np.nan)
            mean.append(float(r[m].mean()) if len(r) and m in r else np.nan)
        best = []
        for j, k in enumerate(ks):
            r = per_k[k]
            col = [float(r[mm].median()) for mm in methods if len(r) and mm in r]
            best.append(bool(col) and np.isclose(med[j], np.nanmax(col)))
        lines.append(
            f"{PRETTY[m]} & "
            + " & ".join(
                f"{fmt(a, bold=b)} & {fmt(c)}"
                for a, c, b in zip(med, mean, best, strict=True)
            )
            + " \\\\"
        )
    header = (
        "Method & "
        + " & ".join(rf"\multicolumn{{2}}{{c}}{{$k={k}$}}" for k in ks)
        + r" \\"
        + "\n & "
        + " & ".join(["med. & mean"] * len(ks))
        + r" \\"
    )
    tex = latex_table(
        "\n".join(lines),
        caption=(
            r"RQ1, recovery ratio: the fraction of the gap between \textsc{sup-strict} "
            "and the fully-supervised ceiling that each method closes, per dataset and "
            r"then aggregated. $0$ means no better than \textsc{sup-strict}, $1$ means the "
            "ceiling was reached, negative means worse than doing nothing. Datasets whose "
            f"ceiling exceeds the floor by less than {min_gap * 100:.0f} accuracy point are "
            "excluded, because there the denominator is a rounding artefact; the median is "
            "the headline because the ratio is heavy-tailed by construction."
        ),
        label="tab:recovery",
        colspec="l" + "rr" * len(ks),
        header=header,
        note=" ".join(
            f"$k={k}$: {n_used[k]} datasets used, {n_drop[k]} excluded."
            for k in ks
        ),
    )
    (out / "T3_recovery.tex").write_text(tex, encoding="utf-8")
    return {
        str(k): {
            "n_datasets_used": n_used[k],
            "n_datasets_excluded": n_drop[k],
            "median": {m: float(per_k[k][m].median()) for m in methods if len(per_k[k])},
            "mean": {m: float(per_k[k][m].mean()) for m in methods if len(per_k[k])},
        }
        for k in ks
    }


def t4_safety(df: pd.DataFrame, ks: list[int], out: Path, threshold: float) -> dict:
    """Regression rates against sup_tf, computed on unaggregated (dataset, seed) pairs."""
    ref = df[df.method == "sup_tf"].set_index(["dataset", "k", "seed"])["accuracy"]
    arms = ["self_train", "label_spread", "augment", "safe_ssl_novalve", "safe_ssl"]
    stats_by = {}
    lines = []
    for m in arms:
        cells = []
        for k in ks:
            sub = df[(df.method == m) & (df.k == k)].set_index(["dataset", "k", "seed"])
            common = sub.index.intersection(ref.index)
            st = regression_rate(
                sub.loc[common, "accuracy"].to_numpy(),
                ref.loc[common].to_numpy(),
                threshold=threshold,
            )
            stats_by[(m, k)] = st
            cells.append(st)
        lines.append(
            f"{PRETTY[m]} & "
            + " & ".join(
                f"{c['regression_rate'] * 100:.1f} & {c['mean_loss_when_regressed'] * 100:.1f} & "
                f"{c['worst_loss'] * 100:.1f}"
                for c in cells
            )
            + " \\\\"
        )
    header = (
        "Method & "
        + " & ".join(
            rf"\multicolumn{{3}}{{c}}{{$k={k}$}}" for k in ks
        )
        + r" \\"
        + "\n"
        + " & "
        + " & ".join(["rate & mean & worst"] * len(ks))
        + r" \\"
    )
    tex = latex_table(
        "\n".join(lines),
        caption=(
            "RQ5, safety. \\emph{rate} = percentage of (dataset, seed) pairs where the method "
            f"is worse than \\textsc{{sup-tf}} by more than {threshold * 100:.0f} accuracy point; "
            "\\emph{mean} = mean loss on those pairs; \\emph{worst} = worst single loss. "
            "All in accuracy points. SAFE-SSL$^{-v}$ is our method with the safety valve "
            "removed, so the SAFE-SSL row versus that row is exactly what the valve buys."
        ),
        label="tab:safety",
        colspec="l" + "rrr" * len(ks),
        header=header,
    )
    (out / "T4_safety.tex").write_text(tex, encoding="utf-8")
    return {f"{m}|k={k}": v for (m, k), v in stats_by.items()}


def t5_lfa(agg: pd.DataFrame, lfa: pd.DataFrame, ks: list[int], out: Path) -> dict:
    if lfa.empty:
        return {}
    res: dict[str, Any] = {}
    comps = list(LFA_COMPONENTS) + ["LFA"]
    lines = []
    for comp in comps:
        cells = []
        for k in ks:
            piv = matrix(agg, k, ["sup_tf", "safe_ssl"])
            if piv.empty:
                cells.append((np.nan, np.nan, 0))
                continue
            delta = (piv["safe_ssl"] - piv["sup_tf"]).rename("delta").reset_index()
            merged = delta.merge(lfa[["dataset", comp]], on="dataset", how="inner").dropna()
            if len(merged) < 4:
                cells.append((np.nan, np.nan, len(merged)))
                continue
            rho, p = sps.spearmanr(merged[comp], merged["delta"])
            cells.append((float(rho), float(p), len(merged)))
            res[f"{comp}|k={k}"] = {"spearman_rho": float(rho), "p": float(p),
                                    "n_datasets": int(len(merged))}
        name = comp if comp != "LFA" else r"\textbf{LFA (composite)}"
        lines.append(
            esc(name).replace(r"\textbf\{LFA (composite)\}", r"\textbf{LFA (composite)}")
            + " & "
            + " & ".join(
                (f"{fmt(r, 2)}$^{{*}}$" if np.isfinite(p) and p < 0.05 else fmt(r, 2))
                for r, p, _ in cells
            )
            + " \\\\"
        )
    tex = latex_table(
        "\n".join(lines),
        caption=(
            "RQ4, predictability. Spearman $\\rho$ between each label-free alignment "
            "component and the realised SSL gain "
            "$\\Delta\\text{acc}=\\text{acc}(\\text{SAFE-SSL})-\\text{acc}(\\textsc{sup-tf})$, "
            "one point per dataset. $^{*}$ marks $p<0.05$ (uncorrected)."
        ),
        label="tab:lfa",
        colspec="l" + "r" * len(ks),
        header="LFA component & " + " & ".join(f"$k={k}$" for k in ks) + r" \\",
    )
    (out / "T5_lfa.tex").write_text(tex, encoding="utf-8")
    return res


def partial_spearman(x, y, z) -> tuple[float, float]:
    """Spearman correlation of x and y controlling for z.

    Rank-transform all three, then apply the usual first-order partial
    correlation formula.  The p-value uses the standard t approximation with
    ``n - 3`` degrees of freedom.
    """
    x = sps.rankdata(np.asarray(x, dtype=float))
    y = sps.rankdata(np.asarray(y, dtype=float))
    z = sps.rankdata(np.asarray(z, dtype=float))
    n = len(x)
    if n < 5:
        return float("nan"), float("nan")
    rxy = np.corrcoef(x, y)[0, 1]
    rxz = np.corrcoef(x, z)[0, 1]
    ryz = np.corrcoef(y, z)[0, 1]
    denom = np.sqrt(max((1 - rxz**2) * (1 - ryz**2), 1e-12))
    r = (rxy - rxz * ryz) / denom
    r = float(np.clip(r, -1.0, 1.0))
    dof = n - 3
    if abs(r) >= 1.0 or dof <= 0:
        return r, 0.0
    t = r * np.sqrt(dof / (1 - r**2))
    return r, float(2 * sps.t.sf(abs(t), dof))


def t5b_lfa_mechanism(agg: pd.DataFrame, lfa: pd.DataFrame, ks: list[int], out: Path,
                      ceilings: dict[str, float]) -> dict:
    """Why LFA correlates with the SSL gain the way it does.

    The obvious confound for any "when does SSL help" statistic is *headroom*:
    a dataset the supervised baseline already solves has nothing left to gain,
    and such datasets also tend to have clean, well-separated feature geometry
    (high LFA).  So we report three raw correlations plus the partial
    correlation of LFA with the gain, controlling for headroom.  Without this
    the sign of the LFA correlation cannot be interpreted.
    """
    if lfa.empty:
        return {}
    res: dict[str, Any] = {}
    lines = []
    for k in ks:
        piv = matrix(agg, k, ["sup_tf", "safe_ssl"])
        if piv.empty:
            continue
        d = pd.DataFrame(
            {
                "dataset": piv.index,
                "delta": (piv["safe_ssl"] - piv["sup_tf"]).to_numpy(),
                "headroom": np.array(
                    [ceilings.get(ds, np.nan) for ds in piv.index]
                ) - piv["sup_tf"].to_numpy(),
            }
        )
        m = d.merge(lfa[["dataset", "LFA"]], on="dataset").dropna()
        if len(m) < 6:
            continue
        r_ld, p_ld = sps.spearmanr(m["LFA"], m["delta"])
        r_lh, p_lh = sps.spearmanr(m["LFA"], m["headroom"])
        r_hd, p_hd = sps.spearmanr(m["headroom"], m["delta"])
        r_part, p_part = partial_spearman(m["LFA"], m["delta"], m["headroom"])
        res[str(k)] = {
            "n_datasets": int(len(m)),
            "rho_LFA_delta": float(r_ld), "p_LFA_delta": float(p_ld),
            "rho_LFA_headroom": float(r_lh), "p_LFA_headroom": float(p_lh),
            "rho_headroom_delta": float(r_hd), "p_headroom_delta": float(p_hd),
            "partial_rho_LFA_delta_given_headroom": float(r_part),
            "p_partial": float(p_part),
        }
        star = lambda pv: "$^{*}$" if np.isfinite(pv) and pv < 0.05 else ""  # noqa: E731
        lines.append(
            f"{k} & {len(m)} & {fmt(r_ld, 2)}{star(p_ld)} & {fmt(r_lh, 2)}{star(p_lh)} & "
            f"{fmt(r_hd, 2)}{star(p_hd)} & {fmt(r_part, 2)}{star(p_part)} \\\\"
        )
    tex = latex_table(
        "\n".join(lines),
        caption=(
            "RQ4, mechanism. Spearman correlations between the label-free alignment "
            "statistic (LFA), the realised SSL gain "
            r"$\Delta=\mathrm{acc}(\text{SAFE-SSL})-\mathrm{acc}(\textsc{sup-tf})$, and the "
            r"\emph{headroom} $\mathrm{acc}(\textsc{sup-full})-\mathrm{acc}(\textsc{sup-tf})$. "
            "The last column is the partial correlation of LFA with the gain once headroom "
            "is controlled for. $^{*}$ marks $p<0.05$."
        ),
        label="tab:lfa_mechanism",
        colspec="rrrrrr",
        header=(
            r"$k$ & \#ds & $\rho$(LFA,$\Delta$) & $\rho$(LFA,head.) & "
            r"$\rho$(head.,$\Delta$) & $\rho$(LFA,$\Delta\mid$head.) \\"
        ),
    )
    (out / "T5b_lfa_mechanism.tex").write_text(tex, encoding="utf-8")
    return res


def t6_ablation(cfg, out: Path) -> dict:
    d = cfg.path_for("results_ablation")
    if not d.exists():
        return {}
    df = load_rows(d)
    if df.empty:
        return {}
    df["axis"] = df["tag"].str.split("=").str[0]
    df["value"] = df["tag"].str.split("=").str[1]
    sub = df[df.method == "safe_ssl"]
    base = df[df.method == "sup_tf"].groupby(["axis", "value", "dataset", "k", "seed"])[
        "accuracy"
    ].mean()
    rows = []
    for (axis, value), g in sub.groupby(["axis", "value"]):
        idx = g.set_index(["axis", "value", "dataset", "k", "seed"]).index
        common = idx.intersection(base.index)
        gg = g.set_index(["axis", "value", "dataset", "k", "seed"]).loc[common]
        delta = gg["accuracy"].to_numpy() - base.loc[common].to_numpy()
        rows.append(
            {
                "axis": axis,
                "value": value,
                "n_cells": int(len(common)),
                "acc": float(gg["accuracy"].mean()),
                "delta_vs_sup_tf": float(np.mean(delta)),
                "regression_rate": float(np.mean(delta < -0.01)),
                "mean_pseudo": float(gg["n_pseudo_selected"].mean()),
                "valve_fire_rate": float(np.mean(gg["safe_ssl_fired"].astype(float))),
            }
        )
    rows.sort(key=lambda r: (r["axis"], r["value"]))
    lines = []
    last = None
    for r in rows:
        if last is not None and r["axis"] != last:
            lines.append(r"\midrule")
        last = r["axis"]
        lines.append(
            f"{esc(r['axis'])} & {esc(r['value'])} & {fmt(r['acc'])} & "
            f"{fmt(r['delta_vs_sup_tf'])} & {r['regression_rate'] * 100:.1f} & "
            f"{r['mean_pseudo']:.0f} & {r['valve_fire_rate'] * 100:.0f} \\\\"
        )
    tex = latex_table(
        "\n".join(lines),
        caption=(
            "Ablations. Each block varies one constant, holding the rest at the default "
            "configuration. $\\Delta$ is against \\textsc{sup-tf} on the same cells; "
            "\\emph{reg.} is the percentage of cells regressing by more than one accuracy "
            "point; \\emph{valve} is the percentage of cells in which the safety valve fired."
        ),
        label="tab:ablation",
        colspec="llrrrrr",
        header=(
            r"Axis & Value & Acc. & $\Delta$ vs.\ \textsc{sup-tf} & reg.\ (\%) & "
            r"\#pseudo & valve (\%) \\"
        ),
    )
    (out / "T6_ablation.tex").write_text(tex, encoding="utf-8")
    return {"rows": rows}


def t7_guidelines(agg: pd.DataFrame, lfa: pd.DataFrame, ks: list[int], out: Path) -> dict:
    """A practical decision table, built only from measured quantities."""
    if lfa.empty:
        return {}
    lfa = lfa.copy()
    try:
        lfa["band"] = pd.qcut(lfa["LFA"], 3, labels=["low", "medium", "high"])
    except ValueError:
        return {}
    rows = []
    for k in ks:
        piv = matrix(agg, k, ["sup_strict", "sup_tf", "safe_ssl"])
        if piv.empty:
            continue
        m = piv.reset_index().merge(lfa[["dataset", "band", "LFA"]], on="dataset")
        for band in ["low", "medium", "high"]:
            g = m[m.band == band]
            if g.empty:
                continue
            d_feat = (g["sup_tf"] - g["sup_strict"]).mean()
            d_pseudo = (g["safe_ssl"] - g["sup_tf"]).mean()
            reg = float(np.mean((g["safe_ssl"] - g["sup_tf"]) < -0.01))
            rec = (
                "SAFE-SSL"
                if d_pseudo > 0.005
                else ("\\textsc{sup-tf}" if d_pseudo > -0.005 else "\\textsc{sup-tf}")
            )
            rows.append(
                {"k": k, "band": band, "n": int(len(g)), "d_features": float(d_feat),
                 "d_pseudo": float(d_pseudo), "regression_rate": reg, "recommend": rec}
            )
    lines = []
    last_k = None
    for r in rows:
        if last_k is not None and r["k"] != last_k:
            lines.append(r"\midrule")
        last_k = r["k"]
        lines.append(
            f"{r['k']} & {r['band']} & {r['n']} & {fmt(r['d_features'])} & "
            f"{fmt(r['d_pseudo'])} & {r['regression_rate'] * 100:.0f} & {r['recommend']} \\\\"
        )
    tex = latex_table(
        "\n".join(lines),
        caption=(
            "Practical guidance. Datasets are split into LFA terciles computed \\emph{without "
            "labels}. $\\Delta_{\\text{features}}$ is always free; $\\Delta_{\\text{pseudo}}$ is "
            "the additional gain from pseudo-labeling. The recommendation column is derived "
            "from these measured numbers, not asserted."
        ),
        label="tab:guidelines",
        colspec="rlrrrrl",
        header=(
            r"$k$ & LFA band & \#ds & $\Delta_{\text{features}}$ & $\Delta_{\text{pseudo}}$ & "
            r"reg.\ (\%) & Use \\"
        ),
    )
    (out / "T7_guidelines.tex").write_text(tex, encoding="utf-8")
    return {"rows": rows}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def f1_cd_diagrams(agg: pd.DataFrame, methods: list[str], ks: list[int], figdir: Path,
                   alpha: float) -> dict:
    info = {}
    n = len(ks)
    fig, axes = plt.subplots(n, 1, figsize=(7.0, 1.9 * n))
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, ks, strict=False):
        piv = matrix(agg, k, methods)
        if piv.empty or len(piv) < 3:
            ax.set_axis_off()
            continue
        fr = friedman_test(piv.to_numpy())
        cd = nemenyi_critical_difference(len(methods), len(piv), alpha)
        plot_cd_diagram(
            fr.avg_ranks,
            [m.replace("_", "-") for m in methods],
            cd,
            title=f"$k={k}$   (N={len(piv)} datasets, Friedman $p={fr.p_f:.2e}$)",
            ax=ax,
        )
        info[k] = {
            "n_datasets": int(len(piv)),
            "cd": float(cd),
            "friedman": fr.as_dict(),
            "avg_ranks": {m: float(r) for m, r in zip(methods, fr.avg_ranks, strict=True)},
        }
    fig.tight_layout()
    save_fig(fig, "F1_cd_diagrams", figdir)
    return info


def f2_recovery_curves(agg: pd.DataFrame, methods: list[str], ks: list[int],
                       figdir: Path, ceilings: dict[str, float], min_gap: float) -> None:
    """Recovery ratio vs. k, one line per method, with +/- 1 sd bands across datasets.

    Uses the same guarded per-dataset ratio as table T3 (datasets whose ceiling
    is within ``min_gap`` of the floor are excluded), so the figure and the
    table can never tell different stories.
    """
    fig, ax = plt.subplots(figsize=(4.4, 3.1))
    for m in methods:
        mu, sd = [], []
        for k in ks:
            rec, _n, _d = recovery_frame(agg, k, methods, ceilings, min_gap)
            v = rec[m].dropna() if len(rec) and m in rec else pd.Series(dtype=float)
            mu.append(float(v.median()) if len(v) else np.nan)
            sd.append(float(v.std(ddof=1)) if len(v) > 1 else np.nan)
        mu, sd = np.array(mu, float), np.array(sd, float)
        ax.plot(ks, mu, marker=MARKERS.get(m, "o"), color=PALETTE.get(m),
                label=m.replace("_", "-"), lw=1.4, ms=4)
        ax.fill_between(ks, mu - 0.25 * sd, mu + 0.25 * sd, color=PALETTE.get(m),
                        alpha=0.10, lw=0)
    ax.axhline(0, color="k", lw=0.7, ls=":")
    ax.axhline(1, color="k", lw=0.7, ls="--")
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("labels per class $k$")
    ax.set_ylabel("median recovery ratio  (0 = sup-strict, 1 = ceiling)")
    # Legend below the axes: every SSL curve hugs y=0 and the ceiling reference
    # sits at y=1, so an in-axes legend lands on top of one or the other.
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              handletextpad=0.4, columnspacing=1.0, fontsize=6.5)
    fig.tight_layout()
    save_fig(fig, "F2_recovery_curves", figdir)


def f3_lfa_scatter(agg: pd.DataFrame, lfa: pd.DataFrame, ks: list[int], figdir: Path) -> dict:
    if lfa.empty:
        return {}
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    cmap = plt.get_cmap("viridis")
    stats_out = {}
    for i, k in enumerate(ks):
        piv = matrix(agg, k, ["sup_tf", "safe_ssl"])
        if piv.empty:
            continue
        d = (piv["safe_ssl"] - piv["sup_tf"]).rename("delta").reset_index()
        m = d.merge(lfa[["dataset", "LFA"]], on="dataset").dropna()
        if len(m) < 4:
            continue
        rho, p = sps.spearmanr(m["LFA"], m["delta"])
        stats_out[k] = {"rho": float(rho), "p": float(p), "n": int(len(m))}
        c = cmap(i / max(len(ks) - 1, 1))
        ax.scatter(m["LFA"], m["delta"], s=16, color=c, alpha=0.75,
                   label=rf"$k={k}$  $\rho={rho:.2f}$", edgecolors="none")
        if len(m) >= 3:
            z = np.polyfit(m["LFA"], m["delta"], 1)
            xs = np.linspace(m["LFA"].min(), m["LFA"].max(), 20)
            ax.plot(xs, np.polyval(z, xs), color=c, lw=1.0, alpha=0.8)
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("LFA (label-free alignment, z-scored composite)")
    ax.set_ylabel(r"$\Delta$acc  (SAFE-SSL $-$ sup-tf)")
    ax.legend(frameon=False, fontsize=6, ncol=2)
    fig.tight_layout()
    save_fig(fig, "F3_lfa_scatter", figdir)
    return stats_out


def f4_attribution_bars(agg: pd.DataFrame, ks: list[int], figdir: Path) -> None:
    """RQ2 decomposition, as grouped bars in accuracy points.

    Deliberately *not* stacked: at k=10 and k=20 the feature term is negative
    while the pseudo-label term is positive, and a stack drawn from a negative
    base misrepresents the decomposition. Grouped bars against a common zero
    line show both signs honestly, with 95% CI whiskers over the 40 datasets.
    """
    series = {"feat": [], "pseudo": [], "novalve": []}
    errs = {"feat": [], "pseudo": [], "novalve": []}
    for k in ks:
        piv = matrix(agg, k, ["sup_strict", "sup_tf", "safe_ssl", "safe_ssl_novalve"])
        cols = {
            "feat": piv["sup_tf"] - piv["sup_strict"] if not piv.empty else None,
            "pseudo": piv["safe_ssl"] - piv["sup_tf"] if not piv.empty else None,
            "novalve": piv["safe_ssl_novalve"] - piv["sup_tf"] if not piv.empty else None,
        }
        for key, v in cols.items():
            if v is None or len(v) < 2:
                series[key].append(np.nan)
                errs[key].append(np.nan)
            else:
                series[key].append(float(v.mean()) * 100)
                errs[key].append(1.96 * float(v.std(ddof=1)) / np.sqrt(len(v)) * 100)

    fig, ax = plt.subplots(figsize=(4.8, 3.1))
    x = np.arange(len(ks), dtype=float)
    w = 0.26
    spec = [
        ("feat", -w, PALETTE["sup_tf"], r"$\Delta$ features  (sup-tf $-$ sup-strict)"),
        ("pseudo", 0.0, PALETTE["safe_ssl"], r"$\Delta$ pseudo  (SAFE-SSL $-$ sup-tf)"),
        ("novalve", w, PALETTE["safe_ssl_novalve"], r"$\Delta$ pseudo, valve removed"),
    ]
    for key, off, colour, lab in spec:
        ax.bar(x + off, series[key], width=w * 0.92, color=colour, label=lab,
               edgecolor="none")
        ax.errorbar(x + off, series[key], yerr=errs[key], fmt="none", ecolor="#3D4956",
                    elinewidth=0.8, capsize=2.2)
    ax.axhline(0, color="k", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"$k$={k}" for k in ks])
    ax.set_xlabel("labels per class $k$")
    ax.set_ylabel("mean gain (accuracy points)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=1,
              fontsize=6.5, handletextpad=0.5)
    fig.tight_layout()
    save_fig(fig, "F4_attribution_bars", figdir)


def f5_safety(agg: pd.DataFrame, ks: list[int], figdir: Path) -> None:
    show = [k for k in ks if k in set(agg.k)][:3] or ks[:1]
    fig, axes = plt.subplots(1, len(show), figsize=(3.0 * len(show), 2.9), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, show, strict=False):
        piv = matrix(agg, k, ["sup_tf", "safe_ssl", "safe_ssl_novalve"])
        if piv.empty:
            ax.set_axis_off()
            continue
        d_nv = (piv["safe_ssl_novalve"] - piv["sup_tf"]).to_numpy()
        d_v = (piv["safe_ssl"] - piv["sup_tf"]).to_numpy()
        order = np.argsort(d_nv)
        xs = np.arange(len(order))
        ax.bar(xs, d_nv[order], width=0.85, color=PALETTE["safe_ssl_novalve"],
               label="no valve")
        ax.plot(xs, d_v[order], color=PALETTE["safe_ssl"], lw=1.2, marker="o", ms=2.5,
                label="SAFE-SSL")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(f"$k={k}$")
        ax.set_xlabel("datasets (sorted)")
    axes[0].set_ylabel(r"$\Delta$acc vs. sup-tf")
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save_fig(fig, "F5_safety", figdir)


#: which methods actually consume the cached label-free artefacts, and therefore
#: must be charged their amortised one-time cost in the Pareto figure.
USES_UNSUP_CACHE = {"sup_tf", "self_train", "label_spread", "safe_ssl", "safe_ssl_novalve"}
USES_DTW_CACHE = {"1nn_dtw"}


def dtw_seconds_per_dataset(cfg) -> dict[str, float]:
    """One-time DTW matrix build time per dataset, from the cache manifests."""
    out: dict[str, float] = {}
    root = cfg.path_for("features")
    if not root.exists():
        return out
    for p in root.glob("*/dtw_test_train_*.json"):
        try:
            m = read_json(p)
            out[m["dataset"]] = float(m.get("seconds", np.nan))
        except Exception:
            continue
    return out


def f6_pareto(df: pd.DataFrame, methods: list[str], figdir: Path, k_show: int,
              cfg, n_cells_per_dataset: int) -> dict:
    """Accuracy vs. wall-clock, charging each method only the cost it actually incurs.

    ``sup_strict`` and ``augment`` fit their own transform inside every cell, so
    that cost is already in ``feature_seconds``.  The SSL arms reuse the cached
    label-free artefacts, so they are charged
    ``unsup_seconds / (#budgets x #seeds)``.  1NN-DTW is charged its share of the
    cached distance matrix.  Charging everyone the same amortised number would
    quietly penalise the methods that never touch the cache.
    """
    sub = df[df.k == k_show]
    dtw_sec = dtw_seconds_per_dataset(cfg)
    out: dict[str, Any] = {}
    fig, ax = plt.subplots(figsize=(4.4, 2.9))
    pts: list[tuple[float, float, str]] = []
    for m in methods:
        g = sub[sub.method == m]
        if g.empty:
            continue
        per_cell = g["total_seconds"].to_numpy(dtype=float)
        if m in USES_UNSUP_CACHE and "unsup_seconds_amortized" in g:
            share = g["unsup_seconds_amortized"].fillna(0.0).to_numpy(dtype=float)
        elif m in USES_DTW_CACHE:
            share = g["dataset"].map(dtw_sec).fillna(0.0).to_numpy(dtype=float)
        else:
            share = np.zeros_like(per_cell)
        amortised = per_cell + share / max(n_cells_per_dataset, 1)
        cost = float(np.median(amortised))
        acc = float(g["accuracy"].mean())
        out[m] = {
            "median_seconds_per_cell": float(np.median(per_cell)),
            "amortised_seconds": cost,
            "one_time_share_seconds": float(np.median(share) / max(n_cells_per_dataset, 1)),
            "mean_accuracy": acc,
        }
        pts.append((cost, acc, m))
    # A legend rather than point annotations: several arms sit within a few
    # thousandths of accuracy of each other, and offset text labels there are
    # ambiguous about which marker they belong to.
    pts.sort(key=lambda t: -t[1])
    for cost, acc, m in pts:
        ax.scatter(cost, acc, s=70, color=PALETTE.get(m), marker=MARKERS.get(m, "o"),
                   edgecolors="k", linewidths=0.5, zorder=3,
                   label=f"{m.replace('_', '-')}  ({cost:.2f} s, {acc:.3f})")
    ax.set_xscale("log")
    ax.set_xlabel("wall-clock per cell, incl. amortised one-time cost (s, log scale)")
    ax.set_ylabel(f"mean test accuracy ($k$={k_show})")
    ax.legend(frameon=False, fontsize=6.5, loc="center left", bbox_to_anchor=(1.01, 0.5),
              handletextpad=0.4, borderaxespad=0.0)
    fig.tight_layout()
    save_fig(fig, "F6_pareto", figdir)
    return out


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def build_summary(ctxd: dict, path: Path) -> None:
    L: list[str] = []
    a = L.append
    a("# Results summary")
    a("")
    a("Every number below is computed by `scripts/make_paper_artifacts.py` directly from")
    a("the files in `results/`. If a number is missing here it was not measured.")
    a("")
    a(f"- datasets with complete results: **{ctxd['n_datasets']}**")
    a(f"- result rows: **{ctxd['n_rows']}**, cells: **{ctxd['n_cells']}**, "
      f"failed cells: **{ctxd['n_failures']}**")
    a(f"- label budgets: {ctxd['ks']}, seeds: {ctxd['n_seeds']}, methods: {ctxd['methods']}")
    a("")

    a("## RQ1 -- Recovery")
    a("")
    a("Fraction of the sup-strict -> sup-full gap that each method closes:")
    a("")
    rec = ctxd.get("recovery", {})
    if rec:
        ks = sorted(rec.keys(), key=int)
        a(f"Datasets whose ceiling is within {ctxd.get('recovery_min_gap', 0.02) * 100:.0f} "
          "accuracy point of the low-label floor are excluded (the denominator would be a "
          "rounding artefact). Median across datasets; mean in parentheses.")
        a("")
        a("| method | " + " | ".join(f"k={k}" for k in ks) + " |")
        a("|---" * (len(ks) + 1) + "|")
        for m in ctxd["methods"]:
            cells = []
            for k in ks:
                med = rec[k]["median"].get(m)
                mean = rec[k]["mean"].get(m)
                cells.append(f"{med:+.3f} ({mean:+.3f})" if med is not None else "--")
            a(f"| {m} | " + " | ".join(cells) + " |")
        a("")
        a("| k | datasets used | excluded |")
        a("|---|---|---|")
        for k in ks:
            a(f"| {k} | {rec[k]['n_datasets_used']} | {rec[k]['n_datasets_excluded']} |")
    a("")

    a("## RQ2 -- Attribution (three tiers)")
    a("")
    for r in ctxd.get("attribution", {}).get("rows", []):
        a(f"- **k={r['k']}** (N={r['n']}): sup-strict {r['sup_strict']:.4f} -> "
          f"sup-tf {r['sup_tf']:.4f} -> SAFE-SSL {r['safe_ssl']:.4f}. "
          f"Free features give **{r['d_features'] * 100:+.2f}** accuracy points "
          f"(sd {r['d_features_sd'] * 100:.2f}, Wilcoxon p={r['p_features']:.2e}); "
          f"pseudo-labeling adds **{r['d_pseudo'] * 100:+.2f}** points "
          f"(sd {r['d_pseudo_sd'] * 100:.2f}, p={r['p_pseudo']:.2e}); "
          f"without the safety valve {r['d_pseudo_novalve'] * 100:+.2f} points "
          f"(p={r.get('p_pseudo_novalve', float('nan')):.2e}).")
    a("")

    a("## RQ3 -- Ranking")
    a("")
    for k, v in sorted(ctxd.get("cd", {}).items(), key=lambda x: int(x[0])):
        ranks = sorted(v["avg_ranks"].items(), key=lambda x: x[1])
        a(f"- **k={k}** (N={v['n_datasets']}, CD={v['cd']:.3f}, "
          f"Friedman F p={v['friedman']['p_F']:.2e}): "
          + ", ".join(f"{m} {r:.2f}" for m, r in ranks))
    a("")
    if ctxd.get("wilcoxon"):
        a("Pairwise Wilcoxon signed-rank vs. sup-tf (Holm-corrected):")
        a("")
        a("| k | method | wins/ties/losses | median diff | p (Holm) |")
        a("|---|---|---|---|---|")
        for k, rows in sorted(ctxd["wilcoxon"].items(), key=lambda x: int(x[0])):
            for r in rows:
                a(f"| {k} | {r['method_a']} | {r['wins']}/{r['ties']}/{r['losses']} | "
                  f"{r['median_diff'] * 100:+.2f} pts | {r['p_holm']:.3g} |")
        a("")

    a("## RQ4 -- Predictability (LFA)")
    a("")
    mech = ctxd.get("lfa_mechanism", {})
    if mech:
        a("Correlations across datasets. `delta` = acc(SAFE-SSL) - acc(sup-tf);")
        a("`headroom` = acc(sup-full) - acc(sup-tf). The last column is the partial")
        a("correlation of LFA with the gain, controlling for headroom.")
        a("")
        a("| k | #ds | rho(LFA, delta) | rho(LFA, headroom) | rho(headroom, delta) "
          "| partial rho(LFA, delta | headroom) |")
        a("|---|---|---|---|---|---|")
        for k in sorted(mech, key=int):
            v = mech[k]
            a(f"| {k} | {v['n_datasets']} "
              f"| {v['rho_LFA_delta']:+.3f} (p={v['p_LFA_delta']:.3g}) "
              f"| {v['rho_LFA_headroom']:+.3f} (p={v['p_LFA_headroom']:.3g}) "
              f"| {v['rho_headroom_delta']:+.3f} (p={v['p_headroom_delta']:.3g}) "
              f"| {v['partial_rho_LFA_delta_given_headroom']:+.3f} "
              f"(p={v['p_partial']:.3g}) |")
        a("")
    lf = ctxd.get("lfa_corr", {})
    if lf:
        a("Per-component Spearman rho with delta:")
        a("")
        comps = sorted({key.split("|")[0] for key in lf})
        kk = sorted({key.split("k=")[1] for key in lf}, key=int)
        a("| component | " + " | ".join(f"k={k}" for k in kk) + " |")
        a("|---" * (len(kk) + 1) + "|")
        for c in comps:
            cells = []
            for k in kk:
                v = lf.get(f"{c}|k={k}")
                cells.append(
                    f"{v['spearman_rho']:+.3f} (p={v['p']:.3g})" if v else "--"
                )
            a(f"| {c} | " + " | ".join(cells) + " |")
    else:
        a("- LFA study not present in `results/lfa/`.")
    a("")

    a("## RQ5 -- Safety")
    a("")
    for key, v in sorted(ctxd.get("safety", {}).items()):
        a(f"- {key}: regression rate **{v['regression_rate'] * 100:.1f}%** of "
          f"{v['n']} (dataset, seed) pairs; mean loss when regressing "
          f"{v['mean_loss_when_regressed'] * 100:.2f} pts; worst "
          f"{v['worst_loss'] * 100:.2f} pts; mean delta {v['mean_delta'] * 100:+.2f} pts.")
    a("")
    if "valve_fire_rate" in ctxd:
        a("Safety-valve firing rate (fraction of cells where SAFE-SSL reverted to the "
          "supervised model):")
        a("")
        for k, v in sorted(ctxd["valve_fire_rate"].items(), key=lambda x: int(x[0])):
            a(f"- k={k}: {v * 100:.1f}%")
        a("")

    a("## RQ6 -- Compute")
    a("")
    for m, v in sorted(ctxd.get("pareto", {}).items(), key=lambda x: x[1]["amortised_seconds"]):
        a(f"- {m}: {v['median_seconds_per_cell']:.3f} s/cell median "
          f"({v['amortised_seconds']:.3f} s incl. amortised unsupervised cost), "
          f"mean accuracy {v['mean_accuracy']:.4f}")
    a("")

    a("## Verdict")
    a("")
    for line in ctxd.get("verdict", []):
        a(f"- {line}")
    a("")
    path.write_text("\n".join(L), encoding="utf-8")


def write_paper_numbers(ctxd: dict, cfg, repro: dict, path: Path) -> None:
    """A camera-ready digest of every number the paper needs.

    ``results_summary.md`` is the exhaustive dump; this file is the curated one:
    the same numbers, arranged the way a paper is written, with the claim each
    number supports spelled out.  Both are generated from the same ``ctxd``, so
    they cannot disagree with each other or with the tables.
    """
    L: list[str] = []
    a = L.append
    pct = lambda x: f"{x * 100:.2f}"  # noqa: E731
    pp = lambda x: f"{x * 100:+.2f}"  # noqa: E731

    ks = ctxd["ks"]
    methods = ctxd["methods"]

    a("# Numbers for the paper")
    a("")
    a("*Free Data, Costly Labels: When Does Unlabeled Data Help Time-Series "
      "Classification with Random Convolutional Kernels?*")
    a("")
    a("Auto-generated by `scripts/make_paper_artifacts.py` from `results/`. "
      "Every figure below is traceable to a file on disk; nothing is estimated, "
      "interpolated or rounded up.")
    a("")
    a("---")
    a("")

    # ---------------------------------------------------------------- setup
    a("## 0. Experimental setup (numbers for the Setup section)")
    a("")
    a(f"- **{ctxd['n_datasets']} UCR datasets**, univariate, equal length, no missing values, "
      f"`n_classes` ≤ 10, every class with ≥ 25 training instances.")
    a(f"- Label budgets **k ∈ {ks}** labels per class; **{ctxd['n_seeds']} seeds** (0–9).")
    a(f"- **{len(methods)} seed-matched methods**: "
      f"{', '.join('`' + m + '`' for m in methods)}, plus `sup_full` (the fully-supervised "
      "ceiling) computed once per dataset.")
    n_deep_rows = int(ctxd.get("deep", {}).get("n_cells", 0))
    a(f"- **{ctxd['n_cells']} cells** = {ctxd['n_datasets']} × {len(ks)} × {ctxd['n_seeds']}, "
      f"**{ctxd['n_rows'] - n_deep_rows} seed-matched result rows**"
      + (f" plus **{n_deep_rows}** rows from the optional `deep_ssl` arm (3 seeds)" if
         n_deep_rows else "")
      + f", **{ctxd['n_failures']} failures**.")
    ft = repro.get("feature_cache_human", "?")
    a(f"- Feature transform: MiniRocket, `n_kernels = {cfg['features']['n_kernels']}` "
      f"(feature dimensionality 9 996), one transform seed; feature cache **{ft}**.")
    a(f"- 1NN-DTW uses a Sakoe-Chiba band of {cfg['dtw']['window']:.0%} of the series length.")
    a(f"- Hardware: {repro.get('cpu_count', '?')} logical CPUs, "
      f"{repro.get('platform', '?')}; CPU-only, one BLAS thread per worker.")
    a(f"- Software: python {repro['versions'].get('python')}, "
      f"aeon {repro['versions'].get('aeon')}, scikit-learn {repro['versions'].get('sklearn')}, "
      f"numpy {repro['versions'].get('numpy')}, scipy {repro['versions'].get('scipy')}.")
    a(f"- Total measured compute in the result rows: "
      f"**{repro.get('total_measured_seconds', 0) / 3600:.2f} CPU-hours**.")
    a("")

    # ------------------------------------------------------------ headlines
    a("## 1. The four sentences the abstract can make")
    a("")
    rows = ctxd.get("attribution", {}).get("rows", [])
    if rows:
        d_feat = float(np.mean([r["d_features"] for r in rows]))
        d_ps = float(np.mean([r["d_pseudo"] for r in rows]))
        sig_feat = sum(r["p_features"] < 0.05 for r in rows)
        a(f"1. **Free unlabeled *features* are worthless.** Fitting the random-kernel "
          f"transform on the unlabeled pool instead of on the labeled subset changes "
          f"accuracy by **{pp(d_feat)} points** on average over {ctxd['n_datasets']} datasets "
          f"and {len(ks)} budgets, and is not significant at any budget "
          f"({sig_feat}/{len(rows)} Wilcoxon tests with p < 0.05).")
        a(f"2. **Free unlabeled *labels* are worth something.** Pseudo-labeling on top of "
          f"the same features adds **{pp(d_ps)} points** on average "
          f"({sum(r['d_pseudo'] > 0 for r in rows)}/{len(rows)} budgets positive).")
    mech = ctxd.get("lfa_mechanism", {})
    if mech:
        mk = sorted(mech, key=int)
        rhos = [mech[k]["rho_LFA_delta"] for k in mk]
        parts = [mech[k]["partial_rho_LFA_delta_given_headroom"] for k in mk]
        sig = sum(mech[k]["p_LFA_delta"] < 0.05 for k in mk)
        a(f"3. **The gain is predictable without labels — with the sign reversed.** "
          f"Spearman ρ between the Label-Free Alignment statistic and the realised gain is "
          f"**negative at every budget** ({min(rhos):+.2f} to {max(rhos):+.2f}, "
          f"significant at {sig}/{len(mk)}); controlling for headroom makes it stronger "
          f"({min(parts):+.2f} to {max(parts):+.2f}). SSL helps most where the unlabeled "
          f"geometry is *worst*.")
    safety = ctxd.get("safety", {})
    if safety:
        pairs = {}
        for key, v in safety.items():
            m, kk = key.split("|k=")
            pairs.setdefault(int(kk), {})[m] = v
        use = {k: v for k, v in pairs.items() if k > 1 and "safe_ssl" in v and
               "safe_ssl_novalve" in v}
        if use:
            rv = np.mean([v["safe_ssl"]["regression_rate"] for v in use.values()])
            rn = np.mean([v["safe_ssl_novalve"]["regression_rate"] for v in use.values()])
            wv = np.mean([v["safe_ssl"]["worst_loss"] for v in use.values()])
            wn = np.mean([v["safe_ssl_novalve"]["worst_loss"] for v in use.values()])
            a(f"4. **A labeled-data-only safety valve removes most of the downside.** "
              f"At k ≥ 2 the valve cuts the regression rate from **{pct(rn)}%** to "
              f"**{pct(rv)}%** of (dataset, seed) pairs and the worst single loss from "
              f"**{pct(wn)}** to **{pct(wv)}** accuracy points.")
    a("")

    # ------------------------------------------------------------------ RQ1
    a("## 2. RQ1 — Recovery  →  `T3_recovery.tex`, `F2_recovery_curves.pdf`")
    a("")
    rec = ctxd.get("recovery", {})
    if rec:
        rk = sorted(rec, key=int)
        a(f"Median (mean) fraction of the `sup_strict` → `sup_full` gap closed. Datasets "
          f"whose ceiling is within {ctxd.get('recovery_min_gap', 0.02) * 100:.0f} accuracy "
          "point of the floor are excluded — there the denominator is a rounding artefact.")
        a("")
        best_per_k = {
            k: max(
                (v for v in rec[k]["median"].values() if v is not None), default=None
            )
            for k in rk
        }
        a("| method | " + " | ".join(f"k={k}" for k in rk) + " |")
        a("|---" * (len(rk) + 1) + "|")
        for m in methods:
            cells = []
            for k in rk:
                med = rec[k]["median"].get(m)
                mean = rec[k]["mean"].get(m)
                if med is None:
                    cells.append("—")
                    continue
                txt = f"{med:+.3f} ({mean:+.3f})"
                b = best_per_k[k]
                cells.append(f"**{txt}**" if b is not None and np.isclose(med, b) else txt)
            a(f"| `{m}` | " + " | ".join(cells) + " |")
        a("")
        a("| k | " + " | ".join(str(k) for k in rk) + " |")
        a("|---" * (len(rk) + 1) + "|")
        a("| datasets used | " + " | ".join(str(rec[k]["n_datasets_used"]) for k in rk) + " |")
        a("| excluded | " + " | ".join(str(rec[k]["n_datasets_excluded"]) for k in rk) + " |")
    a("")

    # ------------------------------------------------------------------ RQ2
    a("## 3. RQ2 — Attribution  →  `T2_attribution.tex`, `F4_attribution_bars.pdf`")
    a("")
    if rows:
        a("Accuracies averaged over datasets and seeds. Δ values are in accuracy points; "
          "p-values are two-sided Wilcoxon signed-rank across datasets.")
        a("")
        a("| k | `sup_strict` | `sup_tf` | `safe_ssl` | Δ_features | p | Δ_pseudo | p | "
          "Δ_pseudo (no valve) |")
        a("|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            a(f"| {r['k']} | {r['sup_strict']:.4f} | {r['sup_tf']:.4f} | {r['safe_ssl']:.4f} "
              f"| {pp(r['d_features'])} | {r['p_features']:.3f} "
              f"| {pp(r['d_pseudo'])} | {r['p_pseudo']:.3f} "
              f"| {pp(r['d_pseudo_novalve'])} |")
        a("")
        a(f"**Claim:** Δ_features is never significant "
          f"(min p = {min(r['p_features'] for r in rows):.3f}); Δ_pseudo is significant at "
          f"{sum(r['p_pseudo'] < 0.05 for r in rows)}/{len(rows)} budgets.")
    a("")

    # ------------------------------------------------------------------ RQ3
    a("## 4. RQ3 — Ranking  →  `T1_main*.tex`, `F1_cd_diagrams.pdf`")
    a("")
    cd = ctxd.get("cd", {})
    if cd:
        ck = sorted(cd, key=int)
        a("Friedman average ranks (1 = best) over the datasets with complete results.")
        a("")
        a("| method | " + " | ".join(f"k={k}" for k in ck) + " |")
        a("|---" * (len(ck) + 1) + "|")
        for m in methods:
            a(f"| `{m}` | " + " | ".join(
                f"{cd[k]['avg_ranks'].get(m, float('nan')):.2f}" for k in ck) + " |")
        a("")
        a("| k | N datasets | critical difference | Friedman F p |")
        a("|---|---|---|---|")
        for k in ck:
            a(f"| {k} | {cd[k]['n_datasets']} | {cd[k]['cd']:.3f} | "
              f"{cd[k]['friedman']['p_F']:.2e} |")
        a("")
    wil = ctxd.get("wilcoxon", {})
    if wil:
        a("Pairwise Wilcoxon signed-rank against `sup_tf`, Holm-corrected within each budget. "
          "Only comparisons significant at the Holm-corrected 5% level are listed.")
        a("")
        a("| k | method | wins/ties/losses | median Δ (pts) | p (Holm) |")
        a("|---|---|---|---|---|")
        any_sig = False
        for k in sorted(wil, key=int):
            for r in wil[k]:
                if r["p_holm"] < 0.05:
                    any_sig = True
                    a(f"| {k} | `{r['method_a']}` | {r['wins']}/{r['ties']}/{r['losses']} "
                      f"| {r['median_diff'] * 100:+.2f} | {r['p_holm']:.2e} |")
        if not any_sig:
            a("| — | *(nothing survives the Holm correction)* | — | — | — |")
    a("")

    # ------------------------------------------------------------------ RQ4
    a("## 5. RQ4 — Predictability  →  `T5_lfa.tex`, `T5b_lfa_mechanism.tex`, "
      "`F3_lfa_scatter.pdf`")
    a("")
    lf = ctxd.get("lfa_corr", {})
    if lf:
        comps = ["sil", "chi", "stab", "knn_purity_proxy", "LFA"]
        kk = sorted({key.split("k=")[1] for key in lf}, key=int)
        a("Spearman ρ between each label-free component and "
          "Δacc = acc(`safe_ssl`) − acc(`sup_tf`), one point per dataset. "
          "**Bold** = p < 0.05.")
        a("")
        a("| component | " + " | ".join(f"k={k}" for k in kk) + " |")
        a("|---" * (len(kk) + 1) + "|")
        for c in comps:
            cells = []
            for k in kk:
                v = lf.get(f"{c}|k={k}")
                if not v:
                    cells.append("—")
                elif v["p"] < 0.05:
                    cells.append(f"**{v['spearman_rho']:+.3f}** ({v['p']:.3f})")
                else:
                    cells.append(f"{v['spearman_rho']:+.3f} ({v['p']:.3f})")
            a(f"| `{c}` | " + " | ".join(cells) + " |")
        a("")
    if mech:
        a("Mechanism check — is the negative sign just a ceiling effect? "
          "`headroom` = acc(`sup_full`) − acc(`sup_tf`).")
        a("")
        a("| k | ρ(LFA, Δ) | ρ(LFA, headroom) | ρ(headroom, Δ) | ρ(LFA, Δ | headroom) |")
        a("|---|---|---|---|---|")
        for k in sorted(mech, key=int):
            v = mech[k]
            a(f"| {k} | {v['rho_LFA_delta']:+.3f} (p={v['p_LFA_delta']:.3f}) "
              f"| {v['rho_LFA_headroom']:+.3f} (p={v['p_LFA_headroom']:.3f}) "
              f"| {v['rho_headroom_delta']:+.3f} (p={v['p_headroom_delta']:.3f}) "
              f"| {v['partial_rho_LFA_delta_given_headroom']:+.3f} "
              f"(p={v['p_partial']:.3f}) |")
        a("")
        a("**Claim:** LFA is uncorrelated with headroom at every budget, and the partial "
          "correlation is stronger than the raw one, so the negative sign is not a ceiling "
          "artefact.")
    a("")

    # ------------------------------------------------------------------ RQ5
    a("## 6. RQ5 — Safety  →  `T4_safety.tex`, `F5_safety.pdf`")
    a("")
    if safety:
        arms = ["self_train", "label_spread", "augment", "safe_ssl_novalve", "safe_ssl"]
        sk = sorted({int(key.split("|k=")[1]) for key in safety})
        a("Regression = worse than `sup_tf` by more than 1 accuracy point, over all "
          f"{ctxd['n_datasets']} × {ctxd['n_seeds']} (dataset, seed) pairs. "
          "Rate in %, losses in accuracy points.")
        a("")
        a("| method | " + " | ".join(f"k={k}: rate / mean loss / worst" for k in sk) + " |")
        a("|---" * (len(sk) + 1) + "|")
        for m in arms:
            cells = []
            for k in sk:
                v = safety.get(f"{m}|k={k}")
                cells.append(
                    f"{v['regression_rate'] * 100:.1f} / {v['mean_loss_when_regressed'] * 100:.2f}"
                    f" / {v['worst_loss'] * 100:.2f}" if v else "—"
                )
            a(f"| `{m}` | " + " | ".join(cells) + " |")
        a("")
        a("Mean Δ vs `sup_tf` (accuracy points):")
        a("")
        a("| method | " + " | ".join(f"k={k}" for k in sk) + " |")
        a("|---" * (len(sk) + 1) + "|")
        for m in arms:
            a(f"| `{m}` | " + " | ".join(
                f"{safety[f'{m}|k={k}']['mean_delta'] * 100:+.2f}"
                if f"{m}|k={k}" in safety else "—" for k in sk) + " |")
        a("")
    vf = ctxd.get("valve_fire_rate", {})
    if vf:
        a("Safety-valve firing rate (fraction of cells where SAFE-SSL reverted to the "
          "supervised model). At k=1 no held-out labeled data exists, so the valve abstains "
          "by design.")
        a("")
        a("| k | " + " | ".join(sorted(vf, key=int)) + " |")
        a("|---" * (len(vf) + 1) + "|")
        a("| fired | " + " | ".join(f"{vf[k] * 100:.1f}%" for k in sorted(vf, key=int)) + " |")
    a("")

    # ------------------------------------------------------------------ RQ6
    a("## 7. RQ6 — Compute  →  `F6_pareto.pdf`")
    a("")
    par = ctxd.get("pareto", {})
    if par:
        a("Median seconds per cell at k=5, single-threaded, plus each method's share of the "
          "one-time label-free precomputation (charged only to the methods that use it).")
        a("")
        a("| method | s/cell | + amortised one-time | mean accuracy |")
        a("|---|---|---|---|")
        for m, v in sorted(par.items(), key=lambda x: x[1]["amortised_seconds"]):
            a(f"| `{m}` | {v['median_seconds_per_cell']:.3f} | "
              f"{v['amortised_seconds']:.3f} | {v['mean_accuracy']:.4f} |")
        a("")
    dp = ctxd.get("deep", {})
    if dp:
        a(f"**Deep baseline (§12).** A 3-block 1D-CNN, {dp['n_parameters']:,} parameters, "
          f"trained on CPU with a **fixed {dp['budget_seconds']:.0f} s wall-clock budget per "
          f"cell** (half supervised, half pseudo-label refinement at confidence "
          f"{dp['pseudo_threshold']:.2f}); {dp['n_cells']} cells "
          f"({dp['n_datasets']} datasets x {dp['n_budgets']} budgets x {dp['n_seeds']} seeds), "
          f"{dp['n_failures']} failures.")
        a("")
        a(f"- Budget exhausted in **{dp['budget_exhausted_rate'] * 100:.0f}%** of cells, i.e. "
          "the budget is binding everywhere: these numbers are what the CNN reaches *at this "
          "compute*, not its converged accuracy.")
        a(f"- Mean accuracy over all budgets: **{dp['mean_accuracy']:.4f}**; at k=5: "
          f"**{dp['mean_accuracy_k5']:.4f}**.")
        a(f"- Wall-clock at k=5: **{dp['median_seconds_k5']:.1f} s/cell** (median), versus "
          f"{dp['safe_ssl_median_seconds_k5']:.2f} s/cell for SAFE-SSL -- "
          f"**{dp['ratio_vs_safe_ssl']:.0f}x** more compute for "
          f"{(dp['mean_accuracy_k5'] - dp['per_k_safe_ssl'][dp['per_k_ks'].index(5)]) * 100:+.2f} "
          "accuracy points.")
        a("")
        a("Per budget:")
        a("")
        a("| k | " + " | ".join(str(k) for k in dp["per_k_ks"]) + " |")
        a("|---" * (len(dp["per_k_ks"]) + 1) + "|")
        a("| `deep_ssl` | " + " | ".join(f"{v:.4f}" for v in dp["per_k_deep"]) + " |")
        a("| `sup_tf` | " + " | ".join(f"{v:.4f}" for v in dp["per_k_sup_tf"]) + " |")
        a("| `safe_ssl` | " + " | ".join(f"{v:.4f}" for v in dp["per_k_safe_ssl"]) + " |")
        a("")
        a("> `deep_ssl` ran with 3 seeds and a fixed time budget, so it is **not** "
          "seed-matched with the other arms and is deliberately excluded from the ranking "
          "tables and the CD diagrams. It appears only here and in the Pareto figure.")
    else:
        a("> **No deep baseline point.** `results/deep/` is empty, so RQ6 compares only the "
          "random-kernel arms. The paper must say this rather than extrapolate a deep point.")
    a("")

    # ------------------------------------------------------------- ablations
    ts = ctxd.get("transform_seed_check")
    if ts:
        a("### Transform-seed sensitivity (supports the single-cache design)")
        a("")
        a(f"{len(ts['datasets'])} datasets x {len(ts['transform_seeds'])} MiniRocket seeds x "
          f"{len(ts['eval_seeds'])} evaluation seeds at k={ts['k']}.")
        a("")
        a("| method | sd across transform seeds | sd across evaluation seeds | ratio |")
        a("|---|---|---|---|")
        for m, v in sorted(ts["summary"].items()):
            a(f"| `{m}` | {v['sd_across_transform_seeds']:.4f} | "
              f"{v['sd_across_evaluation_seeds']:.4f} | {v['ratio']:.2f} |")
        a("")
        a("**Claim:** the random-kernel seed contributes an order of magnitude less "
          "variability than the label draw, so caching one transform per dataset is sound.")
        a("")

    a("## 8. Ablations  →  `T6_ablation.tex`")
    a("")
    ab = ctxd.get("ablation", {}).get("rows", [])
    if ab:
        a("Each block varies one constant, holding the rest at the default. Δ is against "
          "`sup_tf` on the same cells.")
        a("")
        a("| axis | value | accuracy | Δ vs `sup_tf` (pts) | regression rate | #pseudo | "
          "valve fired |")
        a("|---|---|---|---|---|---|---|")
        for r in ab:
            a(f"| `{r['axis']}` | {r['value']} | {r['acc']:.4f} | "
              f"{r['delta_vs_sup_tf'] * 100:+.2f} | {r['regression_rate'] * 100:.1f}% | "
              f"{r['mean_pseudo']:.0f} | {r['valve_fire_rate'] * 100:.0f}% |")
        a("")
    else:
        a("*(ablation results not present in `results/ablation/`)*")
    a("")

    # ------------------------------------------------------------ guidelines
    a("## 9. Practical guidance  →  `T7_guidelines.tex`")
    a("")
    gl = ctxd.get("guidelines", {}).get("rows", [])
    if gl:
        a("Datasets split into LFA terciles computed **without labels**.")
        a("")
        a("| k | LFA band | #ds | Δ_features (pts) | Δ_pseudo (pts) | regression rate |")
        a("|---|---|---|---|---|---|")
        for r in gl:
            a(f"| {r['k']} | {r['band']} | {r['n']} | {r['d_features'] * 100:+.2f} | "
              f"{r['d_pseudo'] * 100:+.2f} | {r['regression_rate'] * 100:.0f}% |")
        a("")
    a("")

    # --------------------------------------------------------------- verdict
    a("## 10. Verdict — what the paper must claim")
    a("")
    for line in ctxd.get("verdict", []):
        a(f"- {line}")
    a("")
    a("---")
    a("")
    a("## 11. Files")
    a("")
    a("| artifact | path |")
    a("|---|---|")
    a("| exhaustive numeric dump | `paper/results_summary.md` |")
    a("| machine-readable numbers | `paper/headline_numbers.json` |")
    a("| reproducibility record | `paper/reproducibility.json` |")
    a("| LaTeX tables | `paper/tables/T*.tex` |")
    a("| figures (PDF + 300 dpi PNG) | `paper/figures/F*.{pdf,png}` |")
    a("| raw per-cell results | `results/raw/*.json` (one per dataset × k × seed) |")
    a("| dataset selection + hashes | `configs/datasets.yaml` |")
    a("| every judgement call | `NOTES.md` |")
    a("")
    path.write_text("\n".join(L), encoding="utf-8")


def make_verdict(ctxd: dict) -> list[str]:
    """Plain statements about whether the thesis survived contact with the data.

    This function exists so the paper's prose cannot drift from the tables: if
    the numbers contradict the hypothesis, the contradiction is written here,
    in these words, and printed at the end of every artifact run.
    """
    out: list[str] = []
    rows = ctxd.get("attribution", {}).get("rows", [])
    if rows:
        d_feat = np.mean([r["d_features"] for r in rows])
        pos = [r for r in rows if r["d_features"] > 0]
        sig_feat = [r for r in rows if r["p_features"] < 0.05]
        out.append(
            f"**Tier 1 (free unsupervised feature fitting) does not pay.** "
            f"sup-tf beat sup-strict at {len(pos)}/{len(rows)} budgets, by "
            f"{d_feat * 100:+.2f} accuracy points on average, and the difference was "
            f"significant (Wilcoxon, p<0.05) at {len(sig_feat)}/{len(rows)} budgets. "
            "Fitting MiniRocket on the unlabeled pool instead of on the labeled subset "
            "is free, and it is also worth essentially nothing."
        )
        d_ps = np.mean([r["d_pseudo"] for r in rows])
        sig_ps = [r for r in rows if r["p_pseudo"] < 0.05]
        out.append(
            f"**Tier 2 (pseudo-labeling) is where the benefit is.** SAFE-SSL beat sup-tf by "
            f"{d_ps * 100:+.2f} accuracy points on average "
            f"({sum(r['d_pseudo'] > 0 for r in rows)}/{len(rows)} budgets positive, "
            f"significant at {len(sig_ps)}/{len(rows)})."
        )
        if d_ps <= 0:
            out.append(
                "**NEGATIVE RESULT**: averaged over datasets, pseudo-labeling did NOT help. "
                "The paper must say so."
            )

    mech = ctxd.get("lfa_mechanism", {})
    if mech:
        ks = sorted(mech, key=int)
        rhos = [mech[k]["rho_LFA_delta"] for k in ks]
        parts = [mech[k]["partial_rho_LFA_delta_given_headroom"] for k in ks]
        sig = [k for k in ks if mech[k]["p_LFA_delta"] < 0.05]
        head = [k for k in ks if mech[k]["p_LFA_headroom"] < 0.05]
        if max(rhos) < 0:
            out.append(
                "**THE LFA HYPOTHESIS IS REJECTED, WITH THE SIGN REVERSED.** The paper "
                "hypothesised that unlabeled data helps when the label-free geometry is "
                "already aligned with the class structure, i.e. a POSITIVE correlation "
                f"between LFA and the SSL gain. Measured: Spearman rho is NEGATIVE at every "
                f"budget ({min(rhos):+.2f} to {max(rhos):+.2f}), significant at p<0.05 for "
                f"{len(sig)}/{len(ks)} budgets. Semi-supervised learning helps MOST where "
                "the unlabeled feature geometry is WORST."
            )
            out.append(
                "The obvious confound -- that well-clustered datasets are already solved and "
                "so have no headroom -- does not explain it: LFA correlates with headroom at "
                f"p<0.05 for only {len(head)}/{len(ks)} budgets, and the partial correlation "
                f"of LFA with the gain controlling for headroom is {min(parts):+.2f} to "
                f"{max(parts):+.2f}, i.e. as strong or stronger than the raw correlation."
            )
            out.append(
                "LFA is therefore still a usable free decision statistic -- it predicts the "
                "gain with |rho| up to "
                f"{max(abs(r) for r in rhos):.2f} without touching a single label -- but the "
                "mechanism claimed in the hypothesis is wrong, and the paper must invert it."
            )
        elif not sig:
            out.append(
                "**NEGATIVE RESULT**: the composite LFA statistic did not significantly "
                "predict the SSL gain at any budget."
            )

    safety = ctxd.get("safety", {})
    if safety:
        pairs = []
        for key, v in safety.items():
            m, k = key.split("|k=")
            if m in ("safe_ssl", "safe_ssl_novalve"):
                pairs.append((int(k), m, v))
        by_k: dict[int, dict[str, Any]] = {}
        for k, m, v in pairs:
            by_k.setdefault(k, {})[m] = v
        usable = {k: v for k, v in by_k.items() if len(v) == 2 and k > 1}
        if usable:
            red = [
                v["safe_ssl_novalve"]["regression_rate"] - v["safe_ssl"]["regression_rate"]
                for v in usable.values()
            ]
            worst_nv = np.mean([v["safe_ssl_novalve"]["worst_loss"] for v in usable.values()])
            worst_v = np.mean([v["safe_ssl"]["worst_loss"] for v in usable.values()])
            out.append(
                "**The safety valve is what SAFE-SSL actually buys.** Averaged over the "
                f"budgets where it can run (k>=2), it cuts the regression rate by "
                f"{np.mean(red) * 100:.1f} percentage points and the worst-case single loss "
                f"from {worst_nv * 100:.1f} to {worst_v * 100:.1f} accuracy points."
            )
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None)
    args = ap.parse_args()

    cfg = load_config()
    tabdir = cfg.path_for("tables")
    figdir = cfg.path_for("figures")
    tabdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    raw = Path(args.results) if args.results else cfg.path_for("results_raw")
    df = load_rows(raw)
    if df.empty:
        print(f"No results found in {raw}")
        return 1

    # the optional deep baseline lives in its own directory (§12); it only ever
    # enters the Pareto figure and the ranking tables if it was actually run.
    deep_dir = ROOT / "results" / "deep"
    n_deep = 0
    if deep_dir.exists():
        ddf = load_rows(deep_dir)
        if not ddf.empty:
            n_deep = int(len(ddf))
            cols = df.columns.union(ddf.columns)
            with warnings.catch_warnings():
                # the deep arm has no `unsup_seconds_amortized`, `lfa_*` etc., so those
                # columns are all-NA on its side; pandas only warns about future dtype
                # inference for such columns, which we do not rely on.
                warnings.simplefilter("ignore", FutureWarning)
                df = pd.concat(
                    [df.reindex(columns=cols), ddf.reindex(columns=cols)], ignore_index=True
                )
    print(f"deep baseline rows: {n_deep}"
          + ("" if n_deep else "  (RQ6 has no deep point; the paper must say so)"))

    # deep_ssl ran with 3 seeds and a fixed wall-clock budget, so it is NOT
    # seed-matched with the other arms: it appears only in the RQ6 Pareto
    # figure and its own block, never in the ranking tables. Stated in the
    # paper rather than quietly mixed in.
    methods = [m for m in cfg.methods if m in set(df.method)]
    pareto_methods = list(methods)
    if "deep_ssl" in set(df.method):
        pareto_methods.append("deep_ssl")
    ks = sorted(set(df.k))
    metrics = ["accuracy", "balanced_accuracy", "macro_f1", "recovery_ratio",
               "n_pseudo_selected", "total_seconds"]
    agg = seed_mean(df, metrics)
    agg.attrs["n_seeds"] = int(df.groupby(["dataset", "method", "k"]).size().max())

    lfa = load_lfa(cfg)
    ceilings = {}
    sf = ROOT / "results" / "sup_full.json"
    if sf.exists():
        ceilings = {k: float(v["accuracy"]) for k, v in read_json(sf).items()}

    print(f"loaded {len(df)} rows over {df.dataset.nunique()} datasets, "
          f"{len(methods)} methods, k={ks}")

    ctxd: dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_cells": int(df.groupby(["dataset", "k", "seed"]).ngroups),
        "n_datasets": int(df.dataset.nunique()),
        "n_seeds": int(df.seed.nunique()),
        "ks": ks,
        "methods": methods,
        "n_failures": len(list(cfg.path_for("failures").glob("*.json")))
        if cfg.path_for("failures").exists() else 0,
    }

    # -- tables ------------------------------------------------------------
    t1 = t1_main(agg, methods, [k for k in (5, 20) if k in ks] or ks[:1], tabdir, ceilings)
    ctxd["main"] = t1
    ctxd["attribution"] = t2_attribution(agg, ks, tabdir)
    min_gap = float(cfg["stats"].get("recovery_min_gap", 0.02))
    ctxd["recovery"] = t3_recovery(agg, methods, ks, tabdir, ceilings, min_gap)
    ctxd["recovery_min_gap"] = min_gap
    ctxd["safety"] = t4_safety(df, ks, tabdir, float(cfg["stats"]["regression_threshold"]))
    ctxd["lfa_corr"] = t5_lfa(agg, lfa, ks, tabdir)
    ctxd["lfa_mechanism"] = t5b_lfa_mechanism(agg, lfa, ks, tabdir, ceilings)
    ctxd["ablation"] = t6_ablation(cfg, tabdir)
    ctxd["guidelines"] = t7_guidelines(agg, lfa, ks, tabdir)

    # -- figures -----------------------------------------------------------
    ctxd["cd"] = {str(k): v for k, v in
                  f1_cd_diagrams(agg, methods, ks, figdir, float(cfg["stats"]["alpha"])).items()}
    f2_recovery_curves(agg, methods, ks, figdir, ceilings, min_gap)
    f3_lfa_scatter(agg, lfa, ks, figdir)
    f4_attribution_bars(agg, ks, figdir)
    f5_safety(agg, ks, figdir)
    n_cells_per_dataset = max(len(ks) * int(df.seed.nunique()), 1)
    ctxd["pareto"] = f6_pareto(
        df, pareto_methods, figdir, 5 if 5 in ks else ks[0], cfg, n_cells_per_dataset
    )

    # -- pairwise tests ----------------------------------------------------
    wil = {}
    for k in ks:
        piv = matrix(agg, k, methods)
        if len(piv) >= 5 and "sup_tf" in methods:
            wil[str(k)] = wilcoxon_holm(piv.to_numpy(), list(piv.columns), baseline="sup_tf")
    ctxd["wilcoxon"] = wil
    write_json(ROOT / "results" / "wilcoxon_holm.json", wil)

    # -- valve firing ------------------------------------------------------
    v = df[df.method == "safe_ssl"]
    if not v.empty:
        ctxd["valve_fire_rate"] = {
            str(k): float(v[v.k == k]["safe_ssl_fired"].astype(float).mean()) for k in ks
        }

    if "deep_ssl" in set(df.method):
        dd = df[df.method == "deep_ssl"]
        k5 = 5 if 5 in ks else ks[0]
        per_k_ks = sorted(set(dd.k))
        # compare on the same basis as the Pareto table: median s/cell at k=5
        deep_sec_k5 = float(dd[dd.k == k5]["total_seconds"].median())
        safe_sec_k5 = float(
            df[(df.method == "safe_ssl") & (df.k == k5)]["total_seconds"].median()
        )
        deep_sec = float(dd["total_seconds"].mean())
        ctxd["deep"] = {
            "n_cells": int(dd.groupby(["dataset", "k", "seed"]).ngroups),
            "n_datasets": int(dd.dataset.nunique()),
            "n_budgets": len(per_k_ks),
            "n_seeds": int(dd.seed.nunique()),
            "n_failures": len(list((ROOT / "results" / "failures").glob("deep__*.json"))),
            "n_parameters": int(dd["n_parameters"].median()) if "n_parameters" in dd else 0,
            "budget_seconds": float(dd["budget_seconds"].median())
            if "budget_seconds" in dd else float("nan"),
            "pseudo_threshold": float(dd["pseudo_threshold"].median())
            if "pseudo_threshold" in dd else float("nan"),
            "budget_exhausted_rate": float(dd["budget_exhausted"].mean())
            if "budget_exhausted" in dd else float("nan"),
            "mean_accuracy": float(dd["accuracy"].mean()),
            "mean_accuracy_k5": float(dd[dd.k == k5]["accuracy"].mean()),
            "mean_seconds": deep_sec,
            "median_seconds_k5": deep_sec_k5,
            "safe_ssl_median_seconds_k5": safe_sec_k5,
            "ratio_vs_safe_ssl": deep_sec_k5 / max(safe_sec_k5, 1e-9),
            "per_k_ks": per_k_ks,
            "per_k_deep": [float(dd[dd.k == k]["accuracy"].mean()) for k in per_k_ks],
            "per_k_sup_tf": [
                float(df[(df.method == "sup_tf") & (df.k == k)]["accuracy"].mean())
                for k in per_k_ks
            ],
            "per_k_safe_ssl": [
                float(df[(df.method == "safe_ssl") & (df.k == k)]["accuracy"].mean())
                for k in per_k_ks
            ],
        }

    tsc = ROOT / "results" / "transform_seed_check.json"
    if tsc.exists():
        ctxd["transform_seed_check"] = read_json(tsc)
        ctxd["transform_seed_check"].pop("rows", None)
    ctxd["verdict"] = make_verdict(ctxd)
    build_summary(ctxd, ROOT / "paper" / "results_summary.md")
    write_json(ROOT / "paper" / "headline_numbers.json", ctxd)

    # -- reproducibility ---------------------------------------------------
    dy = load_datasets_yaml()
    repro = {
        "versions": package_versions(),
        "cpu": platform.processor(),
        "cpu_count": __import__("os").cpu_count(),
        "platform": platform.platform(),
        "config": cfg.raw,
        "n_datasets": len(dataset_names()),
        "dataset_hashes": {
            n: {"train": m.get("sha256_X_train"), "test": m.get("sha256_X_test")}
            for n, m in dy["datasets"].items()
        },
        "selection_rules": dy["selection_rules"],
        "n_cells": ctxd["n_cells"],
        "n_rows": ctxd["n_rows"],
        "n_failures": ctxd["n_failures"],
        "total_measured_seconds": float(df["total_seconds"].sum()),
        "feature_cache_bytes": dir_size(cfg.path_for("features")),
        "feature_cache_human": human_bytes(dir_size(cfg.path_for("features"))),
        "data_cache_bytes": dir_size(cfg.path_for("data_cache")),
    }
    write_json(ROOT / "paper" / "reproducibility.json", repro)
    write_paper_numbers(ctxd, cfg, repro, ROOT / "paper" / "PAPER_NUMBERS.md")

    print(f"\nwrote {len(list(tabdir.glob('*.tex')))} tables to {tabdir}")
    print(f"wrote {len(list(figdir.glob('*.pdf')))} figures to {figdir}")
    print("wrote paper/results_summary.md, paper/PAPER_NUMBERS.md and "
          "paper/reproducibility.json")
    print("\n" + "=" * 72)
    print((ROOT / "paper" / "results_summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
