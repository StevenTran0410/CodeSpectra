# Repo CodeSpectra 

## Product Benchmark Research and Copy / Improve Strategy

This document does not repeat the full PRD. It focuses on the question:
**what already exists in the market, what they do well, and what Repo CodeSpectra should copy, avoid, or improve?**

---

## 1) Quick Summary

### What the market already does very well

* chat with repository context,
* code completion and editing,
* semantic retrieval,
* repository-aware prompts,
* local model connectors.

### What the market has not optimized for this use case

* onboarding reports with fixed sections,
* feature-to-file mapping that is clear enough for reading a repository,
* extracting hidden conventions / anti-patterns,
* “what order should I read this repository in,”
* “which files should not be touched casually,”
* “what does this repository actually do” in an evidence-first format rather than just chat output.

### Strategic conclusion

Repo CodeSpectra should not compete on “better AI coding.”
Repo CodeSpectra should lock onto the problem of:
**repository comprehension, onboarding, memory refresh, and architecture storytelling.**

---

## 2) Condensed benchmark table

| Tool / inspiration source | What it does well                                                              | What to borrow                                                | What is still insufficient for Repo CodeSpectra                            | Improvement direction                                                            |
| ------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------- | -------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| Sourcegraph Cody          | Context retrieval, repo awareness, code graph, multi-repo context              | Hybrid retrieval, context window strategy, repo-aware answers | Does not make onboarding reports the product center                  | Use hybrid retrieval, but make the output a dossier/report rather than just chat |
| Aider                     | Compact repo map, signature-first context                                      | Precomputed repo map, symbol-centric summary                  | More focused on editing flow than reading and understanding the repo | Expand the repo map into a structured report with evidence                       |
| Continue                  | Separated model roles, embeddings, reranker, context providers                 | Provider abstraction, local embeddings, context roles         | Not optimized specifically for onboarding + feature mapping          | Keep the adapter approach, but add a section-specific analysis pipeline          |
| Repomix                   | Packs repos into an AI-friendly format, token awareness, ignore/secret hygiene | Canonical packed summary, token budget awareness              | More packaging-oriented, does not generate deep insight              | Use the packed summary as raw material for analysis                              |
| Tree-sitter / ast-grep    | Multi-language parsing, structural analysis                                    | Syntax-aware extraction, convention mining, structural rules  | Only a building block, not a finished product                        | Use them to build an intelligence layer that produces developer-usable output    |

---

## 3) Analysis of each benchmark

## 3.1 Sourcegraph Cody

### Strengths

* treats context as the center of the system,
* uses multiple context sources such as keyword search, code graph, and repository-based context,
* has the right mental model for retrieval quality.[R1]

### What to borrow

* each question type needs a different context strategy,
* do not blindly stuff the entire repository into the model,
* there should be repository-aware retrieval specifically for architecture / conventions / feature mapping.

### What is not fully aligned with Repo CodeSpectra

* Cody is still centered around being a coding assistant,
* onboarding reports are not its primary artifact.

### What Repo CodeSpectra should do differently

* make the report the number one product artifact,
* keep chat as a secondary layer for later phases,
* every report section must have evidence + confidence.

---

## 3.2 Aider

### Strengths

* the repo map is highly efficient,
* important symbols + signatures are enough for many first-level tasks,
* very good token efficiency.[R2]

### What to borrow

* use the repo map as the backbone of the system prompt,
* create symbol summaries before going into detailed snippets,
* make the map reusable across multiple sections.

### What is not enough for Repo CodeSpectra

* a repo map is not yet a “narrative explanation,”
* it does not answer hidden conventions, reading order, or blast radius.

### How Repo CodeSpectra should improve on it

* combine repo map + architecture card + important files radar + reading playlist.

---

## 3.3 Continue

### Strengths

* model roles are clearly separated,
* embeddings and reranking are first-class concepts,
* context providers are well structured,
* supports local workflows well.[R3][R4]

### What to borrow

* provider abstraction,
* capability matrix,
* embeddings should not be forced to use the same vendor as generation,
* retrieval pipelines can be modular.

### What is not enough for Repo CodeSpectra

* it is not centered on “understand this codebase as a persistent artifact.”

### How Repo CodeSpectra should improve on it

* index once, generate many artifacts:

  * identity card
  * conventions
  * feature map
  * important files
  * onboarding guide

---

## 3.4 Repomix

### Strengths

* packs the codebase into an AI-friendly format,
* respects `.gitignore`,
* counts tokens,
* compresses code using Tree-sitter,
* includes secret hygiene.[R5]

### What to borrow

* canonical packed summary,
* token budget visibility,
* secret-aware preflight checks,
* ability to choose output style.

### What is not enough for Repo CodeSpectra

* it does not have the intelligence layer needed to turn a packed repo into usable understanding.

### How Repo CodeSpectra should improve on it

* use the packed view as the foundational material,
* then run heuristics + retrieval + section-specific generation on top.

---

## 3.5 Tree-sitter and ast-grep

### Strengths

* multi-language AST parsing,
* incremental parsing,
* structural queries and rules,
* excellent for inspecting conventions and boundaries.[R6][R7]

### What to borrow

* symbol extraction,
* decorator / annotation discovery,
* service / controller / use case pattern recognition,
* anti-pattern detection through structural rules.

### Important caveat

* not every language is equally easy to parse deeply,
* v1 should choose a prioritized language group.

### Practical recommendation

Support these well first:

* TypeScript / JavaScript
* Python
* Java / Kotlin
* Go
* C#
  Other languages can fall back to manifest + lexical analysis.

---

## 4) Repo CodeSpectra product gap

### 4.1 The central artifact has not been done “painfully well”

Many current tools are strong at **interactive answers**.
Repo CodeSpectra should be strong at a **persistent understanding artifact**.

This artifact should read like a living onboarding document:

* clear,
* evidence-backed,
* confidence-labeled,
* ordered for reading,
* feature-mapped.

### 4.2 Hidden conventions are a gold mine

Many enterprise repositories have unwritten rules that onboarding docs never fully capture.
Repo CodeSpectra should focus on extracting:

* naming patterns,
* import boundaries,
* service orchestration shape,
* transaction handling style,
* error propagation style,
* logging style,
* testing style,
* “what not to do.”

### 4.3 Feature-to-file mapping is more valuable than vague summary

A summary like “this repository is a payment backend service” is worth only a few seconds.
But if the tool can explain:

* where payouts live,
* where the approval chain starts,
* which file is core,
* which tests cover it,
* which config controls it,
  then onboarding value increases dramatically.

---

## 5) Things that should be copied almost directly

1. **Hybrid retrieval** from Cody
2. **Repo map** from Aider
3. **Role separation** from Continue
4. **Local embeddings by default** from the Continue style
5. **Ignore + token + packing discipline** from Repomix
6. **AST-based structural mining** from Tree-sitter / ast-grep
7. **OpenAI-compatible local endpoint support** inspired by LM Studio / DeepSeek style

---

## 6) Things that should not be copied

1. Do not copy the model where chat is the product center
2. Do not copy the pattern of letting the model describe the repository without evidence
3. Do not copy UX that depends heavily on prompt engineering by the end user
4. Do not copy the habit of stuffing too much code into a single request
5. Do not copy edit/refactor product scope in the first phase

---

## 7) Differentiators that should be locked into positioning

### D1. Local-first repository understanding

Not cloud workspace first, but truly local-first.

### D2. Evidence-first explanations

Every conclusion is grounded in files / symbols / snippets.

### D3. Hidden convention mining

Extract unwritten rules, not just folder descriptions.

### D4. Feature-to-file map

Map functionality to actual modules and files.

### D5. Onboarding reading order

Turn the repository from a “jungle” into a “reading playlist.”

### D6. Blast radius awareness

Identify which files are important, risky, and should not be changed casually.

---

## 8) Final product thesis

If Repo CodeSpectra only does this:

* clone the repo,
* stuff the code into an LLM,
* ask for a summary,
  then it will disappear among many similar tools.

If it can do this:

* strong local indexing,
* structured reports,
* clear hidden rules,
* feature maps that are actually readable,
* a trustworthy important file radar,
* transparent confidence + evidence,
  then Repo CodeSpectra will have a very distinct product identity.

---

## References

* [R1] [https://sourcegraph.com/docs/cody/core-concepts/context](https://sourcegraph.com/docs/cody/core-concepts/context)
* [R2] [https://aider.chat/docs/repomap.html](https://aider.chat/docs/repomap.html)
* [R3] [https://docs.continue.dev/advanced/model-roles/embeddings](https://docs.continue.dev/advanced/model-roles/embeddings)
* [R4] [https://docs.continue.dev/guides/build-your-own-context-provider](https://docs.continue.dev/guides/build-your-own-context-provider)
* [R5] [https://repomix.com/](https://repomix.com/)
* [R6] [https://tree-sitter.github.io/tree-sitter/index.html](https://tree-sitter.github.io/tree-sitter/index.html)
* [R7] [https://ast-grep.github.io/](https://ast-grep.github.io/)
