# RPA-031 - File Manifest, Ignore Engine, Language Detection, and File Classification

* **Epic**: Ingestion
* **Priority**: P0
* **Relative Estimate**: M
* **Milestone**: Milestone D
* **Dependencies**: RPA-030

## 1) Goal

Turn the raw repository into a clean manifest: which files are worth indexing, which should be ignored, and which are source, test, config, generated, docs, or secret-risk files.

## 2) Problem this ticket solves

If the manifest is dirty, tokens will be wasted on useless files, parsers will spend time on irrelevant inputs, and the report will become noisy because of generated files, vendor code, and build artifacts.

## 3) Detailed scope

* Implement file walking over the repository snapshot.
* Respect `.gitignore` and system ignore rules.
* Exclude binary files, build outputs, cache folders, vendor/dependency folders according to policy.
* Implement language detection using extension + heuristic fallback.
* Implement file classification: source, test, config, migration, docs, infra, generated, assets, secret-risk.
* Create manifest records with checksum / mtime / size so incremental indexing can reuse them.

## 4) Implementation notes

* Users should have a place to add custom ignore patterns per workspace/repository.
* Repomix is a strong benchmark for ignore handling, token awareness, and secret hygiene.
* Secret-risk classification is not for reading secrets, but for avoiding accidental sending or deep indexing of risky files.

## 5) Breakdown subtasks

* Implement the ignore rule pipeline.
* Implement the file classifier.
* Implement the language detector.
* Implement manifest persistence and delta detection.
* Add UI/settings for custom ignore patterns.

## 6) Acceptance criteria

* The manifest filters out most common junk files.
* Custom ignore patterns take effect correctly.
* Manifest delta detection supports incremental re-scan.
* File classification is usable by downstream tickets.

## 7) Out of scope

* Deep semantic domain-level classification.
* Enterprise-grade DLP/secret scanning.

## 8) Risks / watchpoints

* If ignore rules are too aggressive, important files may be skipped.
* If ignore rules are too weak, reports will become noisy and scans will slow down.

## 9) Expected deliverables

* Manifest engine
* Ignore/classification rules
* Language detection service
* Custom ignore settings

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
