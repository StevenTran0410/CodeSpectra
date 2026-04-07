# CodeSpectra Analysis Agents

This document describes the current analysis pipeline used in CodeSpectra.

## Overview

CodeSpectra uses one dedicated LLM agent per report section (`A` to `K`) and executes them through a dependency-aware pipeline.

- Sections `A-J` are content agents.
- Section `K` is an auditor agent that evaluates `A-J`.
- Each section has a fixed output schema for consistency across runs.

## Orchestration Model

Pipeline runtime uses Haystack `AsyncPipeline` with async components.

Current dependency graph:

- Parallel base set: `A, B, C, D, F, G, I, J`
- Dependent stage: `E` depends on `D`, `H` depends on `G`
- Final stage: `K` depends on all `A-J`

Section completion emits per-section events for incremental UI updates.

## Agent Responsibilities

### `A` Project Identity Agent

- Infers repo identity, purpose, runtime type, and tech stack.
- Uses retrieval plus direct manifest/doc context.

### `B` Architecture Overview Agent

- Summarizes layers, frameworks, entrypoints, services, integrations.

### `C` Repository Structure Agent

- Maps major folders to roles and explains repo structure.

### `D` Coding Conventions Agent

- Extracts naming, async, DI, test, and error-handling conventions.
- Uses static convention signals as grounding input.

### `E` Forbidden Patterns Agent

- Finds anti-patterns and violated conventions.
- Uses `D` output as negative-space context.

### `F` Feature Map Agent

- Builds feature-level mapping with key files and data flow hints.

### `G` Important Files Radar Agent

- Picks high-value files: entrypoint, backbone, risky files, read-first.

### `H` Onboarding Reading Order Agent

- Produces practical reading path for a new engineer.
- Depends on `G`.

### `I` Glossary Agent

- Extracts domain terms grounded in evidence files.

### `J` Risk / Complexity Agent

- Produces risks and hotspots, combined with static risk context.

### `K` Evidence Auditor Agent

- Reviews `A-J` outputs and scores section-level confidence.
- Reports weakest sections and coverage quality summary.
- No repository retrieval; this is meta-analysis of section outputs.

## Context and Retrieval Model

The pipeline does not use a centralized retrieval broker anymore.

- Each section agent owns its retrieval strategy.
- Agents call the shared retrieval service directly.
- Static analysis outputs are injected as precomputed context for relevant sections.

## Output Contract

Analysis report payload is versioned:

```json
{
  "version": 2,
  "sections": {
    "A": {},
    "B": {},
    "...": {},
    "K": {}
  }
}
```

Older saved reports that still use `sections_v2` are supported by compatibility logic in readers.

