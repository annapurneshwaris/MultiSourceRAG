"""Test RCI fix: run GPT-4o and Claude on 50-query subset with depth-based prompt.

Compares old vs new RCI agreement to validate the fix before full re-run.
"""

from __future__ import annotations

import json
import os
import sys
import time

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def run_test():
    from evaluation.llm_judge import LLMJudge
    from evaluation.judge_runner import _build_chunks_for_judge
    from evaluation.significance import spearman_correlation, quadratic_weighted_kappa

    BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    SUBSET_PATH = os.path.join(BASE, "data", "evaluation", "test_subset_50.json")
    OUT_GPT = os.path.join(BASE, "data", "evaluation", "judge_scores_v2_gpt4o.json")
    OUT_CLAUDE = os.path.join(BASE, "data", "evaluation", "judge_scores_v2_claude.json")

    with open(SUBSET_PATH, "r", encoding="utf-8") as f:
        subset = json.load(f)

    print(f"Loaded {len(subset)} test queries")

    # --- Run GPT-4o ---
    gpt_scores = []
    if os.path.exists(OUT_GPT):
        with open(OUT_GPT, "r", encoding="utf-8") as f:
            gpt_scores = json.load(f)
        print(f"Loaded {len(gpt_scores)} existing GPT-4o v2 scores")

    scored_keys = {(s["query_id"], s["config"]) for s in gpt_scores}
    remaining_gpt = [r for r in subset if (r.get("query_id"), r.get("config")) not in scored_keys]

    if remaining_gpt:
        print(f"\nRunning GPT-4o on {len(remaining_gpt)} remaining queries...")
        judge_gpt = LLMJudge(judge_model="gpt-4o-2024-11-20")

        for i, r in enumerate(remaining_gpt):
            qid = r.get("query_id", "")
            cfg = r.get("config", "DBW")
            print(f"  GPT [{i+1}/{len(remaining_gpt)}] {cfg}|{qid}")
            try:
                result = judge_gpt.evaluate(
                    query_id=qid,
                    query_text=r.get("query", ""),
                    expected_sources=r.get("expected_sources", []),
                    answer=r.get("answer", ""),
                    chunks=_build_chunks_for_judge(r),
                    config=cfg,
                )
                d = result.to_dict()
                d["router_type"] = r.get("router_type", "heuristic")
                gpt_scores.append(d)

                if (i + 1) % 10 == 0:
                    with open(OUT_GPT, "w", encoding="utf-8") as f:
                        json.dump(gpt_scores, f, indent=2, ensure_ascii=False)
                    print(f"    Saved checkpoint ({len(gpt_scores)} total)")

                time.sleep(0.3)
            except Exception as e:
                print(f"    ERROR: {e}")

        with open(OUT_GPT, "w", encoding="utf-8") as f:
            json.dump(gpt_scores, f, indent=2, ensure_ascii=False)
        print(f"GPT-4o done: {len(gpt_scores)} scores saved")

    # --- Run Claude ---
    claude_scores = []
    if os.path.exists(OUT_CLAUDE):
        with open(OUT_CLAUDE, "r", encoding="utf-8") as f:
            claude_scores = json.load(f)
        print(f"Loaded {len(claude_scores)} existing Claude v2 scores")

    scored_keys_c = {(s["query_id"], s["config"]) for s in claude_scores}
    remaining_claude = [r for r in subset if (r.get("query_id"), r.get("config")) not in scored_keys_c]

    if remaining_claude:
        print(f"\nRunning Claude on {len(remaining_claude)} remaining queries...")
        from generation.providers.anthropic_gen import AnthropicProvider
        judge_claude = LLMJudge(
            llm_provider=AnthropicProvider(),
            judge_model="claude-sonnet-4-20250514",
        )

        for i, r in enumerate(remaining_claude):
            qid = r.get("query_id", "")
            cfg = r.get("config", "DBW")
            print(f"  Claude [{i+1}/{len(remaining_claude)}] {cfg}|{qid}")
            try:
                result = judge_claude.evaluate(
                    query_id=qid,
                    query_text=r.get("query", ""),
                    expected_sources=r.get("expected_sources", []),
                    answer=r.get("answer", ""),
                    chunks=_build_chunks_for_judge(r),
                    config=cfg,
                )
                d = result.to_dict()
                d["router_type"] = r.get("router_type", "heuristic")
                claude_scores.append(d)

                if (i + 1) % 10 == 0:
                    with open(OUT_CLAUDE, "w", encoding="utf-8") as f:
                        json.dump(claude_scores, f, indent=2, ensure_ascii=False)
                    print(f"    Saved checkpoint ({len(claude_scores)} total)")

                time.sleep(0.3)
            except Exception as e:
                print(f"    ERROR: {e}")

        with open(OUT_CLAUDE, "w", encoding="utf-8") as f:
            json.dump(claude_scores, f, indent=2, ensure_ascii=False)
        print(f"Claude done: {len(claude_scores)} scores saved")

    # --- Compare ---
    print("\n" + "=" * 60)
    print("AGREEMENT ANALYSIS (v2 depth-based RCI)")
    print("=" * 60)

    # Build maps
    gpt_map = {(s["query_id"], s["config"]): s for s in gpt_scores}
    claude_map = {(s["query_id"], s["config"]): s for s in claude_scores}
    common = set(gpt_map.keys()) & set(claude_map.keys())
    print(f"\nCommon scored pairs: {len(common)}")

    for dim in ["rci", "as", "vm"]:
        key = "as" if dim == "as" else dim
        vals_g, vals_c = [], []
        for k in common:
            g_val = gpt_map[k].get(key)
            c_val = claude_map[k].get(key)
            if g_val is not None and c_val is not None:
                vals_g.append(float(g_val))
                vals_c.append(float(c_val))

        if len(vals_g) < 3:
            print(f"\n{dim.upper()}: insufficient data ({len(vals_g)} pairs)")
            continue

        rho = spearman_correlation(vals_g, vals_c, bootstrap_ci=True)
        int_g = [int(round(v)) for v in vals_g]
        int_c = [int(round(v)) for v in vals_c]

        kappa = None
        if len(set(int_g)) > 1 and len(set(int_c)) > 1:
            kappa = quadratic_weighted_kappa(int_g, int_c)

        # Distribution
        from collections import Counter
        dist_g = Counter(int_g)
        dist_c = Counter(int_c)

        print(f"\n{dim.upper()} (n={len(vals_g)}):")
        print(f"  GPT-4o distribution:  {dict(sorted(dist_g.items()))}")
        print(f"  Claude distribution:  {dict(sorted(dist_c.items()))}")
        print(f"  Spearman rho: {rho['correlation']:.3f} (p={rho['p_value']:.4f})"
              if rho['correlation'] is not None else f"  Spearman rho: N/A")
        if rho.get('ci_lower') is not None:
            print(f"  95% CI: [{rho['ci_lower']:.3f}, {rho['ci_upper']:.3f}]")
        print(f"  Weighted kappa: {kappa:.3f}" if kappa is not None else "  Weighted kappa: N/A")

        # Disagreement analysis
        disagreements = sum(1 for g, c in zip(int_g, int_c) if g != c)
        big_disagree = sum(1 for g, c in zip(int_g, int_c) if abs(g - c) >= 2)
        print(f"  Disagreements: {disagreements}/{len(int_g)} ({100*disagreements/len(int_g):.0f}%)")
        print(f"  2-point disagreements: {big_disagree}/{len(int_g)} ({100*big_disagree/len(int_g):.0f}%)")

    # RA composite
    ra_g, ra_c = [], []
    for k in common:
        g, c = gpt_map[k], claude_map[k]
        ra_gv = g.get("ra")
        ra_cv = c.get("ra")
        if ra_gv is not None and ra_cv is not None:
            ra_g.append(float(ra_gv))
            ra_c.append(float(ra_cv))

    if len(ra_g) >= 3:
        rho_ra = spearman_correlation(ra_g, ra_c, bootstrap_ci=True)
        print(f"\nRA composite (n={len(ra_g)}):")
        print(f"  Spearman rho: {rho_ra['correlation']:.3f} (p={rho_ra['p_value']:.4f})"
              if rho_ra['correlation'] is not None else "  Spearman rho: N/A")
        if rho_ra.get('ci_lower') is not None:
            print(f"  95% CI: [{rho_ra['ci_lower']:.3f}, {rho_ra['ci_upper']:.3f}]")

    # --- Compare with v1 scores ---
    OLD_GPT = os.path.join(BASE, "data", "evaluation", "judge_scores_test_gpt4o.json")
    OLD_CLAUDE = os.path.join(BASE, "data", "evaluation", "judge_scores_test_claude.json")
    if os.path.exists(OLD_GPT) and os.path.exists(OLD_CLAUDE):
        with open(OLD_GPT) as f:
            old_gpt = json.load(f)
        with open(OLD_CLAUDE) as f:
            old_claude = json.load(f)

        old_gpt_map = {(s["query_id"], s["config"]): s for s in old_gpt}
        old_claude_map = {(s["query_id"], s["config"]): s for s in old_claude}
        old_common = set(old_gpt_map.keys()) & set(old_claude_map.keys())

        print("\n" + "=" * 60)
        print("COMPARISON: v1 (old prompt) vs v2 (depth-based)")
        print("=" * 60)

        for dim in ["rci", "as", "vm"]:
            key = "as" if dim == "as" else dim

            # v1
            v1_g, v1_c = [], []
            for k in old_common:
                g_val = old_gpt_map[k].get(key)
                c_val = old_claude_map[k].get(key)
                if g_val is not None and c_val is not None:
                    v1_g.append(float(g_val))
                    v1_c.append(float(c_val))

            # v2
            v2_g, v2_c = [], []
            for k in common:
                g_val = gpt_map[k].get(key)
                c_val = claude_map[k].get(key)
                if g_val is not None and c_val is not None:
                    v2_g.append(float(g_val))
                    v2_c.append(float(c_val))

            rho_v1 = spearman_correlation(v1_g, v1_c, bootstrap_ci=False) if len(v1_g) >= 3 else {}
            rho_v2 = spearman_correlation(v2_g, v2_c, bootstrap_ci=False) if len(v2_g) >= 3 else {}

            r1 = rho_v1.get('correlation')
            r2 = rho_v2.get('correlation')
            r1_str = f"{r1:.3f}" if r1 is not None else "N/A"
            r2_str = f"{r2:.3f}" if r2 is not None else "N/A"
            delta = ""
            if r1 is not None and r2 is not None:
                d = r2 - r1
                delta = f" (delta: {'+' if d >= 0 else ''}{d:.3f})"

            print(f"  {dim.upper()}: v1={r1_str} -> v2={r2_str}{delta}")

        # RA
        v1_ra_g, v1_ra_c = [], []
        for k in old_common:
            g, c = old_gpt_map[k], old_claude_map[k]
            if g.get("ra") is not None and c.get("ra") is not None:
                v1_ra_g.append(float(g["ra"]))
                v1_ra_c.append(float(c["ra"]))

        rho_v1_ra = spearman_correlation(v1_ra_g, v1_ra_c, bootstrap_ci=False) if len(v1_ra_g) >= 3 else {}
        rho_v2_ra = spearman_correlation(ra_g, ra_c, bootstrap_ci=False) if len(ra_g) >= 3 else {}

        r1 = rho_v1_ra.get('correlation')
        r2 = rho_v2_ra.get('correlation')
        r1_str = f"{r1:.3f}" if r1 is not None else "N/A"
        r2_str = f"{r2:.3f}" if r2 is not None else "N/A"
        delta = ""
        if r1 is not None and r2 is not None:
            d = r2 - r1
            delta = f" (delta: {'+' if d >= 0 else ''}{d:.3f})"
        print(f"  RA:  v1={r1_str} -> v2={r2_str}{delta}")


if __name__ == "__main__":
    run_test()
