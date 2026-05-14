# Lab Notebook

Persistent scientific reasoning log that survives context compaction and session boundaries. This is the most important file for cross-session continuity. Read this at the start of every session. Update it whenever hypotheses, observations, or dead ends change.

**Last Updated:** *(update on every edit)*

---

## Active Hypotheses

Track all current hypotheses. Number them sequentially (never reuse numbers). Update status as evidence accumulates.

| # | Hypothesis | Status | Evidence Summary | Confidence | Date Added |
|---|-----------|--------|-----------------|------------|------------|
| H1 | *(state hypothesis clearly and specifically)* | active | *(brief evidence for/against)* | low/medium/high | YYYY-MM-DD |

**Status values:** `active` (under investigation), `supported` (evidence consistent), `weakened` (some counter-evidence), `refuted` (definitively disproven), `modified` (evolved into new form - note which), `parked` (not currently investigating).

---

## Key Observations

Dated observations from analysis runs. Include specific numbers, not vague summaries.

### YYYY-MM-DD
- **Source:** `results/<run_directory>/`
- **Observation:** *(what was observed, with specific values)*
- **Relevance:** *(which hypothesis does this relate to, or is it unexplained?)*

---

## Dead Ends

Approaches that were tried and definitively failed. Recording these prevents re-attempting the same ideas across sessions.

| # | Approach | Why It Failed | Constraint Revealed | Date | Source Run |
|---|---------|--------------|--------------------| -----|------------|
| D1 | *(what was tried)* | *(specific reason for failure)* | *(what this teaches us about the problem)* | YYYY-MM-DD | *(run dir)* |

---

## Open Questions

Questions raised by observations that no current hypothesis addresses. Ranked by scientific importance.

1. **Q1:** *(question)* - *(why it matters)* - Priority: high/medium/low
2. **Q2:** *(question)* - *(why it matters)* - Priority: high/medium/low

---

## Methodological Decisions

Record rationale for key methodological choices so they are not revisited without reason.

| Decision | Rationale | Alternatives Considered | Date |
|----------|-----------|------------------------|------|
| *(what was decided)* | *(why)* | *(what else was considered and why it was rejected)* | YYYY-MM-DD |

---

## Experiment Log Summary

Pointers to detailed experiment logs in run directories. This is the high-level view.

| Run | Date | Goal | Key Result | Outcome |
|-----|------|------|------------|---------|
| *(run dir)* | YYYY-MM-DD | *(what was tested)* | *(primary metric/finding)* | success/partial/failure |

---

## Session State (auto-saved before compaction)

This section is automatically updated by `/compact-smart` before context compaction. It records the working state so the next session (or post-compaction context) can resume seamlessly.

- **Date:** *(auto-filled)*
- **Working on:** *(current task description)*
- **Modified files:** *(list of files changed this session)*
- **Current run:** *(active run directory if any)*
- **Chain in progress:** *(chain name and current position, or "none")*
- **Next step:** *(what to do immediately after resuming)*
- **Focus hint:** *(topic for compaction focus)*

---

## Cross-Session Notes

Free-form notes for communicating with your future self or other agents across session boundaries. Anything that does not fit the structured sections above.

*(Add notes here)*
