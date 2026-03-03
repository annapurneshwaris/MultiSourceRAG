"""Prompt templates for generation, evaluation, and tree navigation."""

GENERATION_PROMPT = """You are a VS Code technical support expert. Answer the user's question using ONLY the provided context chunks. Follow these rules strictly:

CITATION FORMAT:
- Documentation: [DOC-<chunk_id>]
- Bug reports: [BUG-<chunk_id>]
- Roadmap/plans: [PLAN-<chunk_id>]

RULES:
1. EVERY factual claim MUST have a citation immediately after it.
2. If the context doesn't contain enough information, say so explicitly.
3. **Synthesize across source types.** When you have documentation AND bug reports AND/or roadmap items, explicitly connect them:
   - Mention what the docs say, then what bugs reveal about real-world behavior, then what the roadmap says about fixes.
   - Example: "According to [DOC-x], the recommended approach is ... However, [BUG-y] reports that this fails when ... This is tracked for a fix in [PLAN-z]."
4. For debugging queries: identify the root cause from bug reports, cite workarounds from docs, note fix status from plans.
5. For how-to queries: give the documented steps, then warn about known issues from bugs if any exist.
6. For roadmap/status queries: lead with the plan/iteration status, supplement with current doc state and known bugs.
7. Be concise but thorough. Use markdown formatting.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

LLM_JUDGE_PROMPT = """You are evaluating the quality of a RAG system answer for VS Code technical support.

QUERY: {query}
EXPECTED SOURCES: {expected_sources}

ANSWER:
{answer}

RETRIEVED CHUNKS:
{chunks}

Score the answer on three dimensions. Each dimension is scored 0, 1, or 2.
Read the rubric and examples carefully before scoring.

---

## Dimension 1: Technical Depth (RCI)

This measures how deeply the answer explains the underlying mechanism — regardless of query type. Apply the SAME scoring logic to debugging queries, how-to queries, and config queries.

**For debugging/error queries:** Does the answer identify WHY the problem occurs?
**For how-to/config queries:** Does the answer explain WHAT specific mechanism, setting, or feature enables the solution?
**For status/roadmap queries:** Does the answer explain WHAT is changing and WHY?

**0 — No technical explanation.**
The answer gives generic advice ("try reinstalling", "check your settings"), restates the question, or says "refer to documentation" without explaining any mechanism. For how-to queries: just says "you can do X" without identifying which setting, feature, or component is involved.

**1 — Identifies the correct area but lacks specificity.**
The answer names the relevant feature, component, or subsystem but doesn't explain the specific mechanism, setting name, or API. For debugging: "it's a terminal issue" without details. For how-to: "use VS Code settings" without naming the specific setting or explaining what it controls.

**2 — Explains the specific mechanism with technical detail.**
The answer identifies the specific setting, component, API, or condition AND explains how/why it works or fails. For debugging: names the specific component and explains the failure mode. For how-to: names the exact setting and explains what it controls. For config: shows the specific JSON key or command.

IMPORTANT: Score 2 requires BOTH specificity (naming the exact setting/component) AND explanation (what it does or why it matters). Score 1 requires only one of these.

**Examples:**

Debugging query: "Terminal colors are wrong after update"
- RCI=0: "Try resetting your settings." (no mechanism identified)
- RCI=1: "This is related to the terminal renderer." (correct area, but vague)
- RCI=2: "The `terminal.integrated.gpuAcceleration` default changed in v1.85, causing color profile mismatches on macOS with P3 displays." (specific setting + mechanism)

How-to query: "How to configure sidebar position?"
- RCI=0: "You can change the sidebar in VS Code." (no mechanism)
- RCI=1: "The sidebar position can be changed through the View menu or settings." (correct area, not specific)
- RCI=2: "The `workbench.sideBar.location` setting controls sidebar position. You can also right-click the Activity Bar and select Move Primary Side Bar Right." (specific setting + what it controls)

Config query: "How to set language-specific settings?"
- RCI=0: "VS Code supports per-language configuration." (restates the question)
- RCI=1: "You can use settings.json for language-specific settings." (correct area)
- RCI=2: "In settings.json, use `[languageId]` blocks (e.g., `\"[python]\": {{...}}`) which override global settings for files of that language." (specific syntax + mechanism)

Error query: "Extensions not loading after update"
- RCI=0: "Reinstall VS Code." (no cause)
- RCI=1: "There's a compatibility issue with some extensions." (vague)
- RCI=2: "The extension host crashes because extensions compiled against Node 16 APIs are incompatible with the Electron 28 upgrade in v1.86, which ships Node 18." (specific cause + mechanism)

---

## Dimension 2: Actionable Steps (AS)

**0 — No actionable steps.**
The answer is purely descriptive, or says "this is a known issue" without any steps the user can take.

**1 — Vague or incomplete steps.**
The answer suggests a direction (e.g., "check your settings") but doesn't give specific settings paths, commands, or sequences.

**2 — Clear, specific, actionable steps.**
The answer provides concrete steps the user can follow: specific setting names, exact commands, menu paths, or file edits.

**Examples:**
- Query: "How to change terminal font?"
  - AS=0: "You can customize terminal settings." (no specifics)
  - AS=1: "Change the terminal font in your settings." (which setting?)
  - AS=2: "Open Settings (Ctrl+,), search for `terminal.integrated.fontFamily`, and set it to your preferred font, e.g., `'Fira Code', monospace`." (exact setting + example)

- Query: "Git integration not working"
  - AS=0: "This is a known issue with Git." (no steps)
  - AS=1: "Try reinstalling Git and restarting VS Code." (generic)
  - AS=2: "1. Verify Git is on your PATH: run `git --version` in terminal. 2. Set `git.path` in settings.json to your Git executable path. 3. Run 'Git: Refresh' from the command palette (Ctrl+Shift+P)." (specific sequence)

---

## Dimension 3: Version Match (VM)

**0 — Wrong version or contradicts version info.**
The answer references features/settings that don't exist in the user's version, or gives advice that conflicts with version-specific behavior mentioned in the chunks.

**1 — Version not mentioned, answer is general.**
The answer doesn't reference any specific version but gives advice that is broadly applicable. No version contradiction, but no version awareness either.

**2 — Correctly references relevant version.**
The answer mentions the correct VS Code version, notes version-specific behavior, or acknowledges when a feature was introduced/changed/fixed.

**Examples:**
- Query about a bug fixed in v1.85:
  - VM=0: "This feature isn't available yet." (it was added already)
  - VM=1: "Try updating VS Code to the latest version." (generic, no version)
  - VM=2: "This was fixed in v1.85 (see release notes). If you're on an earlier version, upgrade. If you're on v1.85+, the fix should already be applied." (version-specific)

---

## Also classify:

**Root Cause Category** (choose ONE):
- configuration: settings, preferences, workspace config, user error in configuration
- bug_or_issue: known bugs, extension conflicts, platform-specific issues, performance problems
- gap_or_missing: missing features, documentation gaps, feature requests
- unknown: cannot determine from the available information

**Reasoning**: 1-2 sentences explaining your scores.

---

Respond with ONLY this JSON (no markdown, no code fences):
{{
  "rci": <0-2>,
  "as": <0-2>,
  "vm": <0-2>,
  "root_cause_category": "<category>",
  "reasoning": "<explanation>"
}}"""

TREE_NAVIGATION_PROMPT = """You are navigating a documentation tree to find the most relevant sections for a user query.

QUERY: {query}

DOCUMENTATION TREE (showing titles and chunk counts):
{tree_summary}

Select 3-5 most relevant tree nodes that likely contain the answer. Return ONLY the node titles as a JSON list:
["title1", "title2", "title3"]"""
