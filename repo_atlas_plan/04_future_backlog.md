# Repo CodeSpectra

## Future Backlog After Phase 1 Is Sufficiently Strong

This document intentionally does not break the work down into detailed tickets, in order to avoid diluting phase 1.

---

## Phase 2 - Repo Q&A / Ask Mode

Goal:

* enable direct Q&A on the indexed repository,
* provide evidence-backed answers,
* reuse the same retrieval and report artifacts from phase 1.

Suggested scope:

* ask by file / feature / symbol / folder
* answer with file citations
* suggest the next files to open
* include answer confidence + unknowns

## Phase 3 - Task Planner / Review Helper

Goal:

* use existing repository context to help break down tasks, run impact analysis, and generate review checklists.

Suggested scope:

* “If I implement this task, which files will be affected?”
* “Create an implementation plan following repository conventions”
* “What are the risks of changing module X?”
* “Generate a review checklist for feature Y”
* “What test cases should be added?”

## Phase 4 - Repository Evolution Intelligence

* compare architecture drift over time
* git history hotspots / churn / ownership
* stale modules
* dead-code suspicion
* dependency creep

## Conditions for moving into phase 2 / 3

Expansion should only happen when phase 1 meets the minimum bar:

* report accuracy is sufficiently trustworthy,
* the evidence model is strong,
* the feature map is usable,
* the important file radar is not too noisy,
* local/cloud privacy modes do not create confusion.
