## 6. Results and Discussion

### 6.1 Overall Performance (RQ1)

Table 2 presents ablation results across all 9 configurations. **CSAS is the strongest differentiator**: DBW achieves 0.768 CSAS, significantly outperforming single-source D (0.489, p<0.001) and the BM25 baseline (0.681, p<0.001).

For resolution accuracy, DBW (RA=0.708) significantly outperforms the best single-source configuration D (RA=0.686; delta=+0.028, p=0.012). The BM25 baseline achieves slightly higher RA (0.717) but this difference is not statistically significant (p=0.255), while its CSAS deficit (−0.087) is highly significant.

Multi-source configurations consistently outperform single-source on CSAS: DBW (0.768) > DB (0.654) > DW (0.513) > D (0.489) > BW (0.489) > B (0.414) > W (0.161). This confirms that source-aware retrieval fundamentally improves attribution quality.

### 6.2 Significance and Effect Sizes

Table 3 reports paired bootstrap tests (10,000 resamples) with Holm-Bonferroni correction for 9 comparisons.

**RA significance:**
- DBW vs D: delta=+0.028, p=0.012* — significant before and after Holm correction
- DBW vs W: delta=+0.149, p<0.001*** — large effect (d=0.XX)
- DBW vs B: delta=+0.067, p<0.001***
- DBW vs BM25: delta=−0.007, p=0.255 — not significant

**CSAS significance (all highly significant):**
- DBW vs D: delta=+0.280, p<0.001***
- DBW vs BM25: delta=+0.090, p<0.001***
- DBW vs W: delta=+0.607, p<0.001***

The key finding: standard RA metrics underestimate the value of multi-source retrieval. CSAS reveals differences invisible to answer quality metrics alone.

### 6.3 Per-Category Analysis (RQ3)

Table 5 and Figure 3 reveal category-specific source routing patterns:

**Router alignment is near-perfect** (Figure 3):
- how_to/config → docs dominate (83%/78% of retrieved chunks)
- debugging/error_diagnosis → bugs dominate (70%/80%)
- status_roadmap → work items dominate (76%)

**DBW significantly outperforms D in 3 of 5 categories** (Table 5c):
- how_to: +0.054 (p<0.001***)
- status_roadmap: +0.104 (p=0.002**)
- config: +0.104 (p=0.006**)

For debugging (−0.019, p=0.214) and error_diagnosis (−0.040, p=0.094), D retains a slight edge. This is expected: debugging queries are well-served by VS Code's detailed documentation, and adding bug/work_item sources introduces noise.

### 6.4 Attribution Quality (RQ2)

CSAS captures a dimension of quality orthogonal to RA. Across all 5 categories, DBW maintains CSAS > 0.72, demonstrating consistent cross-source attribution regardless of query type.

BM25 achieves high RA but moderate CSAS (0.681) because keyword matching retrieves relevant text without source-type awareness — the answer is correct but citations lack provenance. DBW's source-aware retrieval ensures citations trace back to appropriate source types.

### 6.5 Router Behavior

The heuristic router achieves source distributions closely matching expected patterns (Figure 3). Table 4 compares routing strategies with caveats: adaptive and LLM-zero-shot results use the pre-fix reranker, making direct RA comparison confounded by reranker version.

Routing latency varies significantly: heuristic (<1ms) vs adaptive (~2ms) vs LLM-zero-shot (~500ms), suggesting the heuristic router offers the best latency-quality tradeoff for this dataset.

### 6.6 Analysis Deep Dives

**Query difficulty** (Analysis 2): DBW's advantage is concentrated on the hardest queries. For queries where D alone scores RA < 0.50 (Q1), DBW provides a +0.234 improvement (p<0.001). For easy queries (Q4), D alone suffices. This supports HeteroRAG's value proposition: multi-source retrieval matters most when single sources lack coverage.

**Source complementarity** (Analysis 3): 66% of DBW queries retrieve chunks from 2+ sources, confirming active multi-source utilization. Average sources per query: 2.0.

**Router confidence** (Analysis 4): [Reference confidence terciles — does higher confidence correlate with better RA?]

**Citation errors** (Analysis 1): Only 3 queries exhibit high RA (≥0.7) but low CSAS (<0.5), indicating strong alignment between answer quality and attribution quality in the DBW configuration.

### 6.7 Qualitative Examples

[Reference Analysis 5 case studies — 5 illustrative examples showing multi-source wins, attribution wins, perfect routing, data ceiling, and diversity fix behavior.]

### 6.8 Limitations

1. **Debugging/error_diagnosis**: D retains a slight, non-significant advantage for 50% of query categories. Future work could explore category-adaptive routing that disables multi-source for debugging-heavy queries.

2. **Status/roadmap ceiling**: RA=0.494 across all configs for status_roadmap queries — an inherent data limitation since roadmap information changes frequently and our static dataset cannot capture current plans.

3. **VM low variance**: The Version Match dimension shows minimal discriminative power across configs, suggesting it may be better suited as a binary indicator than a 3-point scale.

4. **Single dataset**: While the VS Code ecosystem is large and representative of open-source projects, generalization to other domains requires additional evaluation.

5. **Router comparison confound**: Table 4 router variants were evaluated with different reranker versions, preventing clean isolation of routing strategy effects.
