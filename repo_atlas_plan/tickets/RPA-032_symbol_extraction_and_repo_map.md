# RPA-032 - Symbol Extraction and Repo Map Generation Using Tree-sitter / Syntax-Aware Parsing

* **Epic**: Intelligence
* **Priority**: P0
* **Relative Estimate**: L
* **Milestone**: Milestone D
* **Dependencies**: RPA-031

## 1) Goal

Extract important symbols from the codebase and build a concise repo map that serves as the backbone for retrieval and report generation.

## 2) Problem this ticket solves

If the system only has a file manifest, it can see file names but not the actual shape of the code. That makes feature mapping and convention mining significantly weaker.

## 3) Detailed scope

* Integrate Tree-sitter or another suitable parser strategy for the prioritized language group.
* Extract symbol records such as class, function, method, interface/type, route, controller, service, use case, repository, job, and command when detectable.
* Store either the signature or a short representative snippet for each symbol.
* Generate a repo map artifact at the level of file -> key symbols.
* Generate parse coverage metrics: number of files parsed successfully / failed / fell back.

## 4) Implementation notes

* Aider’s repo map is a strong benchmark for the value of symbol-first context.
* Tree-sitter is strong at incremental parsing and multi-language support, making it a good foundation for a shared parser layer.
* Lexical fallback should be accepted when parsing fails, rather than treating parse failure as fatal.

## 5) Breakdown subtasks

* Select parser packages for the top 3 to 5 priority languages.
* Implement the symbol extraction pipeline.
* Implement the symbol persistence schema.
* Generate the repo map artifact.
* Expose parse coverage stats for debugging and evaluation.

## 6) Acceptance criteria

* A repo map can be generated successfully on the priority sample repositories.
* Symbol records contain enough information for later retrieval and evidence linking.
* Parse failures do not crash the entire pipeline.
* Coverage stats are visible or logged.

## 7) Out of scope

* Perfect AST support for every language.
* Absolute accuracy for deep call graph analysis.

## 8) Risks / watchpoints

* Parser ecosystems vary by language, so maintenance cost may increase quickly.
* If stored signatures are too long, the token budget will blow up.

## 9) Expected deliverables

* Symbol extraction pipeline
* Symbol storage schema
* Repo map artifact
* Parse coverage stats

## 10) Definition of done

* Related code/service/UI work has been merged and runs in the local environment.
* The ticket has at least minimal tests or an appropriate verification checklist.
* Logging and error states are not left blank or unhandled.
* Related docs/settings have been updated.
* No major ambiguity remains open without being clearly documented.

## 11) Suggested QA checklist

* Re-run this ticket against at least 1 public repository or 1 internal sample repository.
* Check empty states, error states, and cancel/retry states where applicable.
* Restart the app and return to the flow to verify that the data remains correct.
* Verify that logs do not expose secrets or tokens.
