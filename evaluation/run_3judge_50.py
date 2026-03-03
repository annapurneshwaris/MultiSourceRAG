"""Run 3-judge validation on 50-query subset.

GPT-4o and Claude v2 scores already exist. This runs Gemini, then
computes the full 3-judge agreement matrix.
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SUBSET_PATH = os.path.join(BASE, "data", "evaluation", "test_subset_50.json")
OUT_GPT = os.path.join(BASE, "data", "evaluation", "judge_scores_v2_gpt4o.json")
OUT_CLAUDE = os.path.join(BASE, "data", "evaluation", "judge_scores_v2_claude.json")
OUT_GEMINI = os.path.join(BASE, "data", "evaluation", "judge_scores_v2_gemini.json")


def run_gemini():
    from evaluation.llm_judge import LLMJudge
    from evaluation.judge_runner import _build_chunks_for_judge
    from generation.providers.gemini_gen import GeminiProvider

    with open(SUBSET_PATH, "r", encoding="utf-8") as f:
        subset = json.load(f)

    scores = []
    if os.path.exists(OUT_GEMINI):
        with open(OUT_GEMINI, "r", encoding="utf-8") as f:
            scores = json.load(f)
        print(f"Loaded {len(scores)} existing Gemini v2 scores")

    scored_keys = {(s["query_id"], s["config"]) for s in scores}
    remaining = [r for r in subset if (r.get("query_id"), r.get("config")) not in scored_keys]

    if not remaining:
        print("Gemini: all 50 already scored")
        return scores

    print(f"Running Gemini on {len(remaining)} remaining queries...")
    judge = LLMJudge(
        llm_provider=GeminiProvider(),
        judge_model="gemini-2.5-flash",
    )

    for i, r in enumerate(remaining):
        qid = r.get("query_id", "")
        cfg = r.get("config", "DBW")
        print(f"  Gemini [{i+1}/{len(remaining)}] {cfg}|{qid}")
        try:
            result = judge.evaluate(
                query_id=qid,
                query_text=r.get("query", ""),
                expected_sources=r.get("expected_sources", []),
                answer=r.get("answer", ""),
                chunks=_build_chunks_for_judge(r),
                config=cfg,
            )
            d = result.to_dict()
            d["router_type"] = r.get("router_type", "heuristic")
            scores.append(d)

            if (i + 1) % 10 == 0:
                with open(OUT_GEMINI, "w", encoding="utf-8") as f:
                    json.dump(scores, f, indent=2, ensure_ascii=False)
                print(f"    Saved checkpoint ({len(scores)} total)")

            time.sleep(0.5)
        except Exception as e:
            print(f"    ERROR: {e}")

    with open(OUT_GEMINI, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)
    print(f"Gemini done: {len(scores)} scores saved")
    return scores


def compute_agreement():
    from evaluation.significance import spearman_correlation, quadratic_weighted_kappa
    from collections import Counter

    # Load all 3 judges
    judge_data = {}
    for name, path in [("gpt-4o", OUT_GPT), ("claude", OUT_CLAUDE), ("gemini", OUT_GEMINI)]:
        if not os.path.exists(path):
            print(f"Missing {name} scores at {path}")
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        judge_data[name] = {(s["query_id"], s["config"]): s for s in data}
        print(f"Loaded {name}: {len(data)} scores")

    judges = list(judge_data.keys())
    pairs = [(judges[i], judges[j]) for i in range(len(judges)) for j in range(i+1, len(judges))]

    print("\n" + "=" * 70)
    print("3-JUDGE AGREEMENT MATRIX (v2 depth-based RCI, 50 queries)")
    print("=" * 70)

    all_pair_results = {}

    for ja, jb in pairs:
        map_a, map_b = judge_data[ja], judge_data[jb]
        common = set(map_a.keys()) & set(map_b.keys())

        # Filter parse errors
        clean = []
        for k in common:
            a, b = map_a[k], map_b[k]
            if a.get("reasoning") == "Parse error" or b.get("reasoning") == "Parse error":
                continue
            clean.append(k)

        print(f"\n--- {ja} vs {jb} (n={len(clean)}) ---")
        pair_metrics = {}

        for dim in ["rci", "as", "vm"]:
            key = "as" if dim == "as" else dim
            va, vb = [], []
            for k in clean:
                a_val = map_a[k].get(key)
                b_val = map_b[k].get(key)
                if a_val is not None and b_val is not None:
                    va.append(float(a_val))
                    vb.append(float(b_val))

            if len(va) < 3:
                print(f"  {dim.upper()}: insufficient data ({len(va)})")
                continue

            rho = spearman_correlation(va, vb, bootstrap_ci=True)
            int_a = [int(round(v)) for v in va]
            int_b = [int(round(v)) for v in vb]
            kappa = None
            if len(set(int_a)) > 1 and len(set(int_b)) > 1:
                kappa = quadratic_weighted_kappa(int_a, int_b)

            dist_a = dict(sorted(Counter(int_a).items()))
            dist_b = dict(sorted(Counter(int_b).items()))
            disagree = sum(1 for a, b in zip(int_a, int_b) if a != b)
            big_dis = sum(1 for a, b in zip(int_a, int_b) if abs(a-b) >= 2)

            rho_val = rho.get('correlation')
            ci_lo = rho.get('ci_lower')
            ci_hi = rho.get('ci_upper')

            print(f"  {dim.upper()}: rho={rho_val:.3f}" + (f" [{ci_lo:.3f},{ci_hi:.3f}]" if ci_lo is not None else "") +
                  (f" kappa={kappa:.3f}" if kappa is not None else " kappa=N/A") +
                  f" disagree={disagree}/{len(int_a)} ({100*disagree/len(int_a):.0f}%)" +
                  f" 2pt={big_dis}")

            pair_metrics[dim] = {"rho": rho_val, "kappa": kappa}

        # RA composite
        ra_a, ra_b = [], []
        for k in clean:
            a_ra = map_a[k].get("ra")
            b_ra = map_b[k].get("ra")
            if a_ra is not None and b_ra is not None:
                ra_a.append(float(a_ra))
                ra_b.append(float(b_ra))

        if len(ra_a) >= 3:
            rho_ra = spearman_correlation(ra_a, ra_b, bootstrap_ci=True)
            rv = rho_ra.get('correlation')
            ci_lo = rho_ra.get('ci_lower')
            ci_hi = rho_ra.get('ci_upper')
            print(f"  RA:  rho={rv:.3f}" + (f" [{ci_lo:.3f},{ci_hi:.3f}]" if ci_lo is not None else ""))
            pair_metrics["ra"] = {"rho": rv}

        all_pair_results[f"{ja}_vs_{jb}"] = pair_metrics

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for dim in ["rci", "as", "vm", "ra"]:
        rhos = []
        kappas = []
        for pair_name, metrics in all_pair_results.items():
            if dim in metrics:
                r = metrics[dim].get("rho")
                if r is not None:
                    rhos.append(r)
                k = metrics[dim].get("kappa")
                if k is not None:
                    kappas.append(k)
        avg_rho = sum(rhos) / len(rhos) if rhos else None
        avg_kappa = sum(kappas) / len(kappas) if kappas else None
        print(f"  {dim.upper()}: avg_rho={avg_rho:.3f}" if avg_rho is not None else f"  {dim.upper()}: avg_rho=N/A", end="")
        print(f"  avg_kappa={avg_kappa:.3f}" if avg_kappa is not None else "  avg_kappa=N/A")

    print("\nVERDICT:", end=" ")
    ra_rhos = [m["ra"]["rho"] for m in all_pair_results.values() if "ra" in m and m["ra"].get("rho") is not None]
    if ra_rhos and min(ra_rhos) >= 0.70:
        print("PASS - All RA rho >= 0.70. Ready for full run.")
    elif ra_rhos and min(ra_rhos) >= 0.50:
        print(f"MARGINAL - Min RA rho = {min(ra_rhos):.3f}. Consider prompt tweaks.")
    else:
        print(f"FAIL - Min RA rho = {min(ra_rhos):.3f}. Fix prompt before full run." if ra_rhos else "FAIL - insufficient data")


if __name__ == "__main__":
    run_gemini()
    compute_agreement()
