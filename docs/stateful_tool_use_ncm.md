# Extending Next-Concept Modeling (NCM) to Stateful Tool Use and MCP

This module (`evaluation/stateful_tool_use.py`) provides an offline reference implementation and benchmark harness extending Next-Concept Modeling (`arXiv:2610.14162`) from single-turn text generation to stateful, multi-turn Model Context Protocol (MCP) tool invocations.

## Motivation

All four benchmarks in the base repository (`OpenWebText`, `WMT14`, `XSum`, and `Sudoku`) evaluate single-turn generation. When agents interact with transactional databases over MCP across multiple turns, two additional bottlenecks arise:

1. **Outbound JSON token syntax drift**: Standard autoregressive decoding predicts tool calls character-by-character (`{`, `"`, `:`, `,`), requiring dozens of sequential forward passes per tool invocation and allowing malformed key names (`"stae"` instead of `"state"`).
2. **Inbound multi-turn context window bloat**: Relational tables frequently carry 80 to 150 columns per row. During a multi-turn investigation (summary query, anomaly filter, record drilldown, log inspection, and field update), stateless MCP servers re-transmit identical record payloads into the context window on every turn.

## Three NCM Extensions for Tool Use

### 1. Coarse-to-Fine Concept & Slot Gate (`ConceptInvariantGate`)

Maps tool invocations onto NCM's hierarchy:
- **Level 0 (Coarse Intent & Role Invariant)**: Selects the discrete capability concept (`C_GLOBAL_RESOLVE_INTENT`, `C_REVIEWER_AGGREGATE_HEALTH`, `C_CONTRIBUTOR_UPDATE_RECORD`, `C_REVIEWER_APPROVE_REQUEST`) and verifies role permissions before slot realization.
- **Level 1 (Schema Lock)**: Locks the required and optional typed parameter slots for the chosen concept and rejects unknown or missing keys prior to execution.
- **Level 2 (Parallel Slot Realization)**: Populates the locked parameter slots in parallel.

### 2. Temporal Residual Context Deltas (`TemporalResidualWorkingSet`)

In NCM's multiscale residual VQ-VAE, hierarchy level $k$ encodes only the spatial residual relative to level $k-1$:

$$r_k = r_{k-1} - \hat{z}_{k-1}$$

`TemporalResidualWorkingSet` applies the same residual principle across conversation turns ($t-1 \to t$) using each record's monotonic modification counter (`mod_version`):
- **Turn 1**: Projects wide database rows into compact concept snapshots (`95%+` smaller than raw 100-column rows).
- **Turn $t$ (unchanged `mod_version`)**: Collapses previously delivered records into compact `@modN` residual stubs:
  ```json
  { "id": "REC-001042", "title": "Workflow Automation Deployment 1", "state": "Active", "_context_delta": "unchanged@mod11" }
  ```
- **Turn $t$ (incremented `mod_version`)**: Emits the updated concept snapshot together with `_context_delta: "updated@mod12"` and `_changed_fields: ["milestone_2_date"]`.

### 3. Level-1 Speculative Policy Verification (`SpeculativeDryRunVerifier`)

When an agent calls a mutating tool (`update_record` or `approve_request`), the runtime evaluates a non-mutating `DRY_RUN_PREVIEW` receipt at Level 1. The check computes before-and-after field diffs (`changes: [{ field, before, after }]`), verifies domain ordering rules (such as enforcing monotonic milestone dates `milestone_1_date <= milestone_2_date <= milestone_3_date <= milestone_4_date` and blocking completion when `milestone_3_date` is unscheduled), and calculates downstream state-machine side effects like target completion dates.

## Running the Benchmark and Offline Tests

```bash
# Run offline unit tests (no GPU required)
python3 -m unittest discover -s tests -v

# Run the 20-record multi-turn compaction benchmark
python3 -m evaluation.stateful_tool_use
```
