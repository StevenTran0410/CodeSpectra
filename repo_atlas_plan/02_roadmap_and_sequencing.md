# Repo CodeSpectra 

## Roadmap, Sequencing, and Dependency Map

Assumptions used for estimation:

* 1 strong full-stack developer for desktop + TypeScript
* 1 backend / AI pipeline developer
* 1 part-time QA / PM
* estimates are **relative**, not hard commitments

---

## 1) The correct build order

### The wrong order teams often fall into

1. build the report UI first
2. build chat first
3. build cloud provider support before the indexing backbone is solid
4. build wide Bitbucket / GitHub support before defining the output schema

### Recommended order

1. lock the product contract
2. lock the output schema
3. build the shell + storage + secrets
4. build the provider abstraction
5. build repo sync + indexing backbone
6. build the core report sections
7. build viewer/export
8. only then harden security and evaluation
9. future: chat / task planner

---

## 2) Milestones

## Milestone A - Product Foundation

Goal:

* lock the product boundary,
* lock the privacy modes,
* lock the data model and output schema.

Included tickets:

* RPA-001
* RPA-002

**Exit criteria**

* there is a product contract document strong enough that the team does not argue about scope every day,
* every report section has a clear schema,
* privacy wording has been finalized.

---

## Milestone B - Platform Shell

Goal:

* the app can launch,
* workspaces exist,
* settings exist,
* secret storage exists,
* basic job state management exists.

Included tickets:

* RPA-010
* RPA-011
* RPA-012

**Exit criteria**

* the desktop app runs reliably on at least 1 target OS,
* workspaces can be saved,
* secrets are not stored in plaintext,
* jobs support progress, cancel, and retry.

---

## Milestone C - Providers & Code Hosts

Goal:

* the user can connect both model providers and repository hosts.

Included tickets:

* RPA-020
* RPA-021
* RPA-022
* RPA-023
* RPA-024

**Exit criteria**

* local provider connection works,
* at least 1 cloud provider connection works,
* GitHub repository discovery works,
* Bitbucket repository discovery / clone works through the chosen flow.

---

## Milestone D - Repo Intelligence Backbone

Goal:

* the system can clone/sync,
* index,
* parse,
* build graphs,
* retrieve grounded context.

Included tickets:

* RPA-030
* RPA-031
* RPA-032
* RPA-033
* RPA-034

**Exit criteria**

* the full pipeline runs from repo -> index artifacts,
* incremental reruns do not rebuild blindly,
* repo map and search artifacts are usable.
* retrieval A/B experiment output exists (baseline hybrid vs vectorless navigation mode) with token + quality metrics.

---

## Milestone E - Analysis Experience V1

Goal:

* turn artifacts into a report that is genuinely useful.

Included tickets:

* RPA-040
* RPA-041
* RPA-042
* RPA-043

**Exit criteria**

* the user can read:

  * project identity,
  * architecture,
  * conventions/style,
  * functionality map,
  * important files,
  * reading order.
* Markdown/JSON export works successfully.

---

## Milestone F - Hardening & Release

Goal:

* increase reliability,
* reduce hallucination,
* package a usable internal build.

Included tickets:

* RPA-050
* RPA-051

**Exit criteria**

* golden repositories exist for testing,
* quality gates exist,
* there is an internal release build.

---

## 3) MVP cut vs Full v1

## MVP cut

* RPA-001
* RPA-002
* RPA-010
* RPA-011
* RPA-012
* RPA-020
* RPA-021
* RPA-023
* RPA-030
* RPA-031
* RPA-032
* RPA-040
* RPA-042
* RPA-043

### What the MVP will include

* desktop app
* workspace management
* local model support
* GitHub support
* repository cloning
* basic indexing
* project identity
* architecture skeleton
* important files
* onboarding reading order
* export

### What the MVP will still be weak at

* full cloud provider support
* Bitbucket polish
* deep convention mining
* deep feature mapping
* strong retrieval / embeddings

## Full v1

All tickets in this package.

---

## 4) Short dependency map

### RPA-001 -> foundation for every ticket

If privacy + data contract are not locked first, later work becomes vulnerable to cascading rewrites.

### RPA-002 depends on RPA-001

The output schema must follow the product contract.

### RPA-010 / 011 / 012 can run partially in parallel

But RPA-011 and RPA-012 need to align with the shell/platform choice from RPA-010.

### RPA-020 is the foundation for RPA-021 and RPA-022

Without building the abstraction first, provider adapters will become fragmented.

### RPA-023 and RPA-024 partially depend on RPA-011

Because they require secret handling and settings support.

### RPA-030 is the foundation for RPA-031 / 032 / 033 / 034

If clone/sync/index snapshots are not solid, everything after that becomes unstable.

### RPA-031 and RPA-032 can run in parallel

* one side handles file classification,
* the other handles symbol parsing.

### RPA-033 depends heavily on RPA-031 and RPA-032

Because graph construction requires both manifest data and symbols.

### RPA-034 depends on RPA-031 and ideally also RPA-032

Chunking/retrieval needs a clean manifest; symbols make retrieval better.
For vectorless retrieval experiments, RPA-033 graph artifacts are also required for fair comparison.

### RPA-040 / 041 / 042 depend on a nearly complete backbone

Report sections only become good once the artifacts are rich enough.

### RPA-043 depends on 040 / 041 / 042

Viewer/export needs real artifacts.

### RPA-050 and RPA-051 belong near the end, but should be prepared early

Do not wait until the end to think about evaluation and security.

---

## 5) Suggested sprint order

## Sprint 1

* RPA-001
* RPA-002
* RPA-010

## Sprint 2

* RPA-011
* RPA-012
* RPA-020

## Sprint 3

* RPA-021
* RPA-023
* RPA-030

## Sprint 4

* RPA-031
* RPA-032

## Sprint 5

* RPA-033
* RPA-034

## Sprint 6

* RPA-040
* RPA-042

## Sprint 7

* RPA-041
* RPA-043

## Sprint 8

* RPA-022
* RPA-024
* RPA-050
* RPA-051

Notes:

* If an early demo is the priority, RPA-043 can be pulled forward immediately after RPA-040/042 so the report viewer appears sooner.
* If marketability is the priority, RPA-022 (cloud providers) can be moved earlier.
* If correctness is the priority, keep the backbone first.

---

## 6) Scope-cutting decisions when capacity is low

If time is limited, cut in this order:

1. compare scan
2. deep Bitbucket polish
3. advanced convention mining
4. cloud provider breadth
5. complex embeddings
6. full multi-OS packaging

But these **should not be cut**:

* evidence model
* confidence
* repo map
* important files
* onboarding reading order
* structured output schema

---

## 7) Signs the product is drifting in the wrong direction

1. There are more UI tickets than indexing tickets
2. The team talks more about prompts than about data artifacts
3. Each provider has custom logic spread across many places
4. The report sounds good but does not point to actual files
5. The team wants chat before the viewer/report is usable
6. Adding a new code host is easier than improving report accuracy
   => this is a signal that the product is drifting off-center.

---

## 8) Short conclusion

The correct build order for Repo CodeSpectra is:
**contract -> schema -> shell -> providers/hosts -> ingestion -> intelligence -> report -> hardening**

If the team stays on this axis, phase 1 will have a strong backbone, and the later task planner / review phases will finally have solid ground to stand on.
