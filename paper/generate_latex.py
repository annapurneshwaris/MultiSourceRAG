"""STEP 6: Generate LaTeX tables from JSON table files.

Reads: paper/tables/table{1..8}_*.json
Outputs: paper/tables/latex/table{1..8}.tex
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_tex(content, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  Saved: {path}")


def bold_if(val, is_best):
    s = f"{val}"
    if is_best:
        return f"\\textbf{{{s}}}"
    return s


# ===================================================================
# TABLE 1
# ===================================================================
def latex_table1():
    print("\n--- LaTeX TABLE 1 ---")
    data = load_json("paper/tables/table1_dataset_stats.json")

    lines = [
        r"\begin{table}[t]",
        r"\caption{Dataset statistics. Token counts computed via whitespace tokenization.}",
        r"\label{tab:dataset}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llrrrrrrl}",
        r"\toprule",
        r"Source & Repository & Raw & Chunks & Min & P25 & Median & P75 & Max \\",
        r"\midrule",
    ]

    for row in data["rows"]:
        lines.append(
            f"{row['source']} & {row['repository']} & "
            f"{row['raw_items']:,} & {row['chunks']:,} & "
            f"{row['min_tokens']} & {row['p25_tokens']} & {row['median_tokens']} & "
            f"{row['p75_tokens']} & {row['max_tokens']} \\\\"
        )

    lines.append(r"\midrule")
    lines.append(
        f"\\textbf{{Total}} & & "
        f"{data['total']['raw_items']:,} & {data['total']['chunks']:,} & "
        f"& & & & \\\\"
    )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    save_tex("\n".join(lines), "paper/tables/latex/table1.tex")


# ===================================================================
# TABLE 2
# ===================================================================
def latex_table2():
    print("\n--- LaTeX TABLE 2 ---")
    data = load_json("paper/tables/table2_ablation.json")
    best = data["best_values"]

    lines = [
        r"\begin{table}[t]",
        r"\caption{Source ablation results. CSAS and RA averaged across 3 LLM judges.",
        r"DBW uses heuristic router ($n$=250). Best values in \textbf{bold}.",
        r"$^*$\,\textit{p}\,<\,.05, $^{**}$\,\textit{p}\,<\,.01, $^{***}$\,\textit{p}\,<\,.001 vs.\ D (paired bootstrap).}",
        r"\label{tab:ablation}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Config & $n$ & CSAS$\uparrow$ & RA$\uparrow$ & MSUR & Retrieved & Latency (ms) \\",
        r"\midrule",
    ]

    for row in data["rows"]:
        cfg = row["config"]
        sig = row.get("ra_sig_vs_d", "")
        sig_tex = f"$^{{{sig}}}$" if sig else ""

        csas_str = bold_if(f"{row['csas']:.3f}", cfg == best["csas"])
        ra_str = bold_if(f"{row['ra']:.3f}", cfg == best["ra"]) + sig_tex
        msur_str = bold_if(f"{row['msur']:.3f}", cfg == best["msur"])

        # Add separator before baselines
        if cfg == "BM25":
            lines.append(r"\midrule")

        lines.append(
            f"{cfg} & {row['n']} & {csas_str} & {ra_str} & {msur_str} & "
            f"{row['retrieved']:.0f} & {row['latency_ms']:.0f} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    save_tex("\n".join(lines), "paper/tables/latex/table2.tex")


# ===================================================================
# TABLE 3
# ===================================================================
def latex_table3():
    print("\n--- LaTeX TABLE 3 ---")
    data = load_json("paper/tables/table3_significance.json")

    lines = [
        r"\begin{table}[t]",
        r"\caption{Paired bootstrap significance tests (10{,}000 resamples). $\dagger$ = significant after Holm--Bonferroni correction.}",
        r"\label{tab:significance}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"& \multicolumn{3}{c}{RA} & \multicolumn{3}{c}{CSAS} \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}",
        r"Comparison & $\Delta$ & $p$ & $d$ & $\Delta$ & $p$ & $d$ \\",
        r"\midrule",
    ]

    for comp in data["comparisons"]:
        label = comp["comparison"]
        ra = comp.get("ra", {})
        csas = comp.get("csas", {})

        def fmt_p(p_val):
            if p_val < 0.001:
                return "<.001"
            elif p_val < 0.01:
                return f"{p_val:.3f}"
            else:
                return f"{p_val:.3f}"

        def sig_mark(entry):
            if not isinstance(entry, dict) or "bootstrap_p" not in entry:
                return ""
            p = entry["bootstrap_p"]
            holm = entry.get("significant_after_holm", False)
            mark = ""
            if p < 0.001:
                mark = "***"
            elif p < 0.01:
                mark = "**"
            elif p < 0.05:
                mark = "*"
            if holm:
                mark += r"$\dagger$"
            return mark

        if isinstance(ra, dict) and "delta" in ra:
            ra_delta = f"{ra['delta']:+.3f}"
            ra_p = fmt_p(ra["bootstrap_p"]) + sig_mark(ra)
            ra_d = f"{ra['effect_size_cohens_d']:.2f}"
        else:
            ra_delta = ra_p = ra_d = "--"

        if isinstance(csas, dict) and "delta" in csas:
            csas_delta = f"{csas['delta']:+.3f}"
            csas_p = fmt_p(csas["bootstrap_p"]) + sig_mark(csas)
            csas_d = f"{csas['effect_size_cohens_d']:.2f}"
        else:
            csas_delta = csas_p = csas_d = "--"

        lines.append(
            f"{label} & {ra_delta} & {ra_p} & {ra_d} & "
            f"{csas_delta} & {csas_p} & {csas_d} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    save_tex("\n".join(lines), "paper/tables/latex/table3.tex")


# ===================================================================
# TABLE 4
# ===================================================================
def latex_table4():
    print("\n--- LaTeX TABLE 4 ---")
    data = load_json("paper/tables/table4_router.json")

    lines = [
        r"\begin{table}[t]",
        r"\caption{Router comparison on DBW configuration. Note: adaptive and LLM-zero-shot",
        r"use pre-fix reranker; heuristic uses fixed reranker with hybrid BM25. Direct RA",
        r"comparison is confounded by reranker version.}",
        r"\label{tab:router}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
        r"Router & $n$ & RA & CSAS & MSUR & Route (ms) & Total (ms) & Doc & Bug & WI \\",
        r"\midrule",
    ]

    for row in data["rows"]:
        boosts = row.get("avg_source_boosts", {})
        lines.append(
            f"{row['router_type']} & {row['n']} & {row['ra']:.3f} & {row['csas']:.3f} & "
            f"{row['msur']:.3f} & {row['route_ms_mean']:.1f} & {row['total_ms_mean']:.0f} & "
            f"{boosts.get('doc', 0):.2f} & {boosts.get('bug', 0):.2f} & {boosts.get('work_item', 0):.2f} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    save_tex("\n".join(lines), "paper/tables/latex/table4.tex")


# ===================================================================
# TABLE 5
# ===================================================================
def latex_table5():
    print("\n--- LaTeX TABLE 5 ---")
    data = load_json("paper/tables/table5_per_category.json")

    # 5a: DBW per category
    lines = [
        r"\begin{table}[t]",
        r"\caption{Per-category performance (DBW heuristic). Dominant source shows \% of",
        r"retrieved chunks from each source type.}",
        r"\label{tab:per-category}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lcccccl}",
        r"\toprule",
        r"Category & $n$ & RA & CSAS & MSUR & Dominant & Dom.\% \\",
        r"\midrule",
    ]

    for row in data["table5a_dbw_per_category"]:
        cat = row["category"].replace("_", r"\_")
        lines.append(
            f"{cat} & {row['n']} & {row['ra']:.3f} & {row['csas']:.3f} & "
            f"{row['msur']:.3f} & {row['dominant_source']} & {row['dominant_source_pct']:.0f}\\% \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    # 5b: Cross-config
    lines.append(r"\vspace{1em}")
    lines.append(r"\begin{tabular}{lcccccr}")
    lines.append(r"\toprule")
    lines.append(r"Category & DBW & D & B & W & BM25 & $\Delta$ \\")
    lines.append(r"\midrule")

    for row in data["table5b_cross_config"]:
        cat = row["category"].replace("_", r"\_")
        vals = []
        for cfg in ["DBW", "D", "B", "W", "BM25"]:
            v = row.get(f"{cfg}_ra")
            vals.append(f"{v:.3f}" if v is not None else "--")
        delta = row.get("delta_vs_best_single", 0)
        sign = "+" if delta >= 0 else ""
        lines.append(
            f"{cat} & {vals[0]} & {vals[1]} & {vals[2]} & {vals[3]} & {vals[4]} & {sign}{delta:.3f} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    # 5c: Significance
    lines.append(r"\vspace{1em}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"Category & $n$ & $\Delta$(DBW$-$D) & $p$ & Sig. \\")
    lines.append(r"\midrule")

    for row in data["table5c_per_category_significance"]:
        cat = row["category"].replace("_", r"\_")
        if "error" in row:
            lines.append(f"{cat} & {row['n_paired']} & -- & -- & -- \\\\")
            continue
        p = row["bootstrap_p"]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        lines.append(
            f"{cat} & {row['n_paired']} & {row['delta']:+.3f} & {p:.3f} & {sig} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    save_tex("\n".join(lines), "paper/tables/latex/table5.tex")


# ===================================================================
# TABLE 6
# ===================================================================
def latex_table6():
    print("\n--- LaTeX TABLE 6 ---")
    data = load_json("paper/tables/table6_inter_judge.json")

    lines = [
        r"\begin{table}[t]",
        r"\caption{Inter-judge agreement. Pearson $r$ and Spearman $\rho$ on RA;",
        r"quadratic weighted $\kappa$ on individual dimensions.}",
        r"\label{tab:agreement}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Judge Pair & $n$ & $r$ (RA) & $\rho$ (RA) & $\kappa$ (RCI) & $\kappa$ (AS) \\",
        r"\midrule",
    ]

    for pair in data["pairs"]:
        lines.append(
            f"{pair['pair']} & {pair['n_paired']} & "
            f"{pair['pearson_r']:.3f} & {pair['spearman_rho']:.3f} & "
            f"{pair['kappa_rci']:.3f} & {pair['kappa_as']:.3f} \\\\"
        )

    lines.append(r"\midrule")
    avg = data["averages"]
    lines.append(
        f"\\textit{{Average}} & & "
        f"{avg['pearson_r']:.3f} & {avg['spearman_rho']:.3f} & "
        f"{avg['kappa_rci']:.3f} & {avg['kappa_as']:.3f} \\\\"
    )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    save_tex("\n".join(lines), "paper/tables/latex/table6.tex")


# ===================================================================
# TABLE 7
# ===================================================================
def latex_table7():
    print("\n--- LaTeX TABLE 7 ---")

    lines = [
        r"\begin{table}[t]",
        r"\caption{Evaluation metrics and LLM judge rubric.}",
        r"\label{tab:metrics}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llp{6cm}}",
        r"\toprule",
        r"Metric & Range & Description \\",
        r"\midrule",
        r"RA & [0,1] & (RCI + AS + VM) / 6, averaged across 3 judges \\",
        r"\ \ RCI & 0--2 & Technical depth: 0=none, 1=correct area, 2=specific mechanism \\",
        r"\ \ AS & 0--2 & Actionable steps: 0=none, 1=vague, 2=specific commands \\",
        r"\ \ VM & 0--2 & Version match: 0=wrong, 1=general, 2=version-specific \\",
        r"\midrule",
        r"CSAS & [0,1] & F1 of source-type citations vs.\ expected sources \\",
        r"MSUR & [0,1] & Fraction of \{doc, bug, work\_item\} in top-$k$ \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    save_tex("\n".join(lines), "paper/tables/latex/table7.tex")


# ===================================================================
# TABLE 8
# ===================================================================
def latex_table8():
    print("\n--- LaTeX TABLE 8 ---")
    data = load_json("paper/tables/table8_latency.json")

    lines = [
        r"\begin{table}[t]",
        r"\caption{Latency breakdown for DBW heuristic configuration ($n$=" + str(data["n"]) + r").}",
        r"\label{tab:latency}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Stage & Mean (ms) & P50 (ms) & P95 (ms) \\",
        r"\midrule",
    ]

    for row in data["rows"]:
        stage = row["stage"]
        if stage == "Total":
            lines.append(r"\midrule")
        lines.append(
            f"{stage} & {row['mean_ms']:.0f} & {row['p50_ms']:.0f} & {row['p95_ms']:.0f} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    save_tex("\n".join(lines), "paper/tables/latex/table8.tex")


# ===================================================================
# MAIN
# ===================================================================
def main():
    print("=" * 60)
    print("STEP 6: Generating LaTeX Tables")
    print("=" * 60)

    latex_table1()
    latex_table2()
    latex_table3()
    latex_table4()
    latex_table5()
    latex_table6()
    latex_table7()
    latex_table8()

    print("\n" + "=" * 60)
    print("All 8 LaTeX tables generated!")
    print("=" * 60)


if __name__ == "__main__":
    main()
