"""Stateful multi-turn tool use, temporal residual context deltas (@modN),
and Level-1 speculative dry-run policy verification for Next-Concept Modeling (NCM).

Extends NCM's multiscale residual paradigm (r_k = r_{k-1} - z_hat_{k-1}) from
spatial token hierarchies to temporal multi-turn agent state and structured
Model Context Protocol (MCP) tool invocations.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


MILESTONE_FIELDS: Tuple[str, ...] = (
    "milestone_1_date",
    "milestone_2_date",
    "milestone_3_date",
    "milestone_4_date",
)

TERMINAL_STATES: Set[str] = {"Complete", "Cancelled", "Declined"}


@dataclass(frozen=True)
class ConceptSpec:
    """Level-0 / Level-1 discrete concept contract for structured tool calls."""

    concept_id: str
    tool_name: str
    required_slots: Tuple[str, ...] = ()
    optional_slots: Tuple[str, ...] = ()
    allowed_roles: Tuple[str, ...] = (
        "role_admin",
        "role_reviewer",
        "role_contributor",
        "role_viewer",
    )
    mutates_state: bool = False


DEFAULT_CONCEPT_REGISTRY: Dict[str, ConceptSpec] = {
    "resolve_intent": ConceptSpec(
        concept_id="C_GLOBAL_RESOLVE_INTENT",
        tool_name="resolve_intent",
        required_slots=("query",),
        optional_slots=("scope", "limit"),
    ),
    "get_aggregate_health": ConceptSpec(
        concept_id="C_REVIEWER_AGGREGATE_HEALTH",
        tool_name="get_aggregate_health",
        required_slots=(),
        optional_slots=("segment", "include_unchanged"),
        allowed_roles=("role_admin", "role_reviewer"),
    ),
    "list_records": ConceptSpec(
        concept_id="C_CONTRIBUTOR_LIST_RECORDS",
        tool_name="list_records",
        required_slots=(),
        optional_slots=("segment", "state", "limit", "include_unchanged"),
    ),
    "update_record": ConceptSpec(
        concept_id="C_CONTRIBUTOR_UPDATE_RECORD",
        tool_name="update_record",
        required_slots=("record_id",),
        optional_slots=(
            "state",
            "owner",
            "summary",
            "milestone_1_date",
            "milestone_2_date",
            "milestone_3_date",
            "milestone_4_date",
            "dry_run",
        ),
        allowed_roles=("role_admin", "role_contributor"),
        mutates_state=True,
    ),
    "approve_request": ConceptSpec(
        concept_id="C_REVIEWER_APPROVE_REQUEST",
        tool_name="approve_request",
        required_slots=("request_id",),
        optional_slots=("comment", "dry_run"),
        allowed_roles=("role_admin", "role_reviewer"),
        mutates_state=True,
    ),
}


class ConceptInvariantGate:
    """Validates Level-0 concept permissions and Level-1 parameter slot schemas."""

    def __init__(self, registry: Optional[Dict[str, ConceptSpec]] = None) -> None:
        self.registry = dict(registry or DEFAULT_CONCEPT_REGISTRY)

    def validate(
        self,
        tool_name: str,
        caller_role: str,
        slots: Dict[str, Any],
    ) -> Tuple[bool, Optional[str], Optional[ConceptSpec]]:
        spec = self.registry.get(tool_name)
        if spec is None:
            return False, f"Unknown tool concept: {tool_name}", None

        if caller_role not in spec.allowed_roles:
            return (
                False,
                f"Role '{caller_role}' is not permitted to invoke {spec.concept_id} ({tool_name})",
                spec,
            )

        missing = [
            req
            for req in spec.required_slots
            if req not in slots or slots[req] is None or str(slots[req]).strip() == ""
        ]
        if missing:
            return (
                False,
                f"Missing required Level-1 slots for {spec.concept_id}: {', '.join(missing)}",
                spec,
            )

        permitted = set(spec.required_slots) | set(spec.optional_slots)
        unknown = [k for k in slots.keys() if k not in permitted]
        if unknown:
            return (
                False,
                f"Unknown Level-1 slots for {spec.concept_id}: {', '.join(sorted(unknown))}",
                spec,
            )

        return True, None, spec


def _parse_iso_date(value: Any) -> Optional[date]:
    if not value or not isinstance(value, str):
        return None
    raw = value.strip()[:10]
    try:
        return datetime.strptime(raw, "%Y-%m-%d").date()
    except ValueError:
        return None


def compute_target_completion_date(record: Dict[str, Any], sprint_weeks: int = 3) -> str:
    """Computes a target completion date from explicit end dates or anchor milestones."""
    for explicit_key in ("milestone_4_date", "milestone_3_date", "target_completion_date"):
        val = str(record.get(explicit_key) or "").strip()
        if _parse_iso_date(val) is not None:
            return val[:10]

    for anchor_key in ("milestone_1_date", "milestone_2_date", "created_date"):
        anchor = _parse_iso_date(record.get(anchor_key))
        if anchor is not None:
            estimated = anchor + timedelta(days=sprint_weeks * 7)
            return f"{estimated.isoformat()} (estimated)"

    return "TBD"


def compute_stage_velocity(
    record: Dict[str, Any],
    reference_date: Optional[date] = None,
) -> Dict[str, Any]:
    """Computes stage dwell time, milestone coverage, and velocity anomaly flags."""
    today = reference_date or date.today()
    state = str(record.get("state") or "Active")
    is_terminal = state in TERMINAL_STATES

    scheduled_count = sum(
        1 for m in MILESTONE_FIELDS if _parse_iso_date(record.get(m)) is not None
    )

    updated_dt = _parse_iso_date(record.get("updated_date"))
    created_dt = _parse_iso_date(record.get("created_date"))

    days_since_update = (today - updated_dt).days if updated_dt is not None else 0
    days_in_flight = (today - created_dt).days if created_dt is not None else 0

    target_str = compute_target_completion_date(record)
    target_dt = _parse_iso_date(target_str)

    flags: List[str] = []
    if not is_terminal:
        if days_since_update >= 14:
            flags.append("STALLED_UPDATES_14D")
        if scheduled_count == 0 and state not in {"New", "Draft"}:
            flags.append("UNSCHEDULED_MILESTONES")
        if target_dt is not None and target_dt < today:
            flags.append("TARGET_DATE_OVERDUE")

    return {
        "scheduled_milestone_count": scheduled_count,
        "days_since_last_update": max(0, days_since_update),
        "days_in_flight": max(0, days_in_flight),
        "target_completion_date": target_str,
        "velocity_flags": flags,
    }


def project_record_concept(
    raw_row: Dict[str, Any],
    reference_date: Optional[date] = None,
) -> Dict[str, Any]:
    """Projects a wide database row (80-150 columns) into a compact NCM concept."""
    velocity = compute_stage_velocity(raw_row, reference_date=reference_date)
    projected: Dict[str, Any] = {
        "id": str(raw_row.get("id") or ""),
        "title": str(raw_row.get("title") or ""),
        "summary": str(raw_row.get("summary") or ""),
        "segment": str(raw_row.get("segment") or ""),
        "state": str(raw_row.get("state") or "Active"),
        "owner": str(raw_row.get("owner") or "Unassigned"),
        "secondary_owner": str(raw_row.get("secondary_owner") or ""),
        "linked_request_id": str(raw_row.get("linked_request_id") or ""),
        "architecture_pattern": str(raw_row.get("architecture_pattern") or ""),
        "needs_attention": bool(raw_row.get("needs_attention", False)),
        "milestone_1_date": str(raw_row.get("milestone_1_date") or ""),
        "milestone_2_date": str(raw_row.get("milestone_2_date") or ""),
        "milestone_3_date": str(raw_row.get("milestone_3_date") or ""),
        "milestone_4_date": str(raw_row.get("milestone_4_date") or ""),
        "target_completion_date": velocity["target_completion_date"],
        "scheduled_milestone_count": velocity["scheduled_milestone_count"],
        "days_since_last_update": velocity["days_since_last_update"],
        "days_in_flight": velocity["days_in_flight"],
        "velocity_flags": velocity["velocity_flags"],
        "_mod_version": int(raw_row.get("mod_version", 0)),
        "_updated_date": str(raw_row.get("updated_date") or ""),
    }
    return projected


@dataclass
class _SnapshotEntry:
    mod_version: int
    updated_date: str
    snapshot: Dict[str, Any]


@dataclass
class _CachedQuery:
    expires_at: float
    rows: List[Dict[str, Any]]


class TemporalResidualWorkingSet:
    """Session working set implementing temporal residual context deltas (@modN).

    Analogous to NCM spatial residual quantization (r_k = r_{k-1} - z_hat_{k-1}),
    tracks delivered record versions across turns (t-1 -> t) so unchanged records
    collapse to lightweight `@modN` stubs and mutated records highlight only the
    changed fields.
    """

    def __init__(self, ttl_seconds: float = 45.0, clock=time.monotonic) -> None:
        self.ttl_seconds = float(ttl_seconds)
        self._clock = clock
        self._records: Dict[str, _SnapshotEntry] = {}
        self._query_cache: Dict[str, _CachedQuery] = {}

    def get_cached_query(self, query_key: str) -> Optional[List[Dict[str, Any]]]:
        entry = self._query_cache.get(query_key)
        if entry is None:
            return None
        if self._clock() > entry.expires_at:
            self._query_cache.pop(query_key, None)
            return None
        return copy.deepcopy(entry.rows)

    def set_cached_query(self, query_key: str, rows: List[Dict[str, Any]]) -> None:
        self._query_cache[query_key] = _CachedQuery(
            expires_at=self._clock() + self.ttl_seconds,
            rows=copy.deepcopy(rows),
        )

    def invalidate_on_mutation(self, record_id: Optional[str] = None) -> None:
        """Clears cached queries on any write so subsequent reads never serve stale state."""
        self._query_cache.clear()
        if record_id is not None:
            self._records.pop(record_id, None)

    def compact_list(
        self,
        concept_rows: Sequence[Dict[str, Any]],
        include_unchanged: bool = False,
    ) -> Dict[str, Any]:
        """Compacts a list of projected concept rows against the session working set."""
        full_bytes = len(json.dumps(list(concept_rows), separators=(",", ":")).encode("utf-8"))
        output_rows: List[Dict[str, Any]] = []
        unchanged_count = 0
        updated_count = 0
        new_count = 0

        for row in concept_rows:
            rec_id = str(row.get("id") or "")
            mod_ver = int(row.get("_mod_version", 0))
            upd_date = str(row.get("_updated_date") or "")
            clean_row = {
                k: v for k, v in row.items() if k not in {"_mod_version", "_updated_date"}
            }

            prev = self._records.get(rec_id)
            if include_unchanged or prev is None:
                self._records[rec_id] = _SnapshotEntry(
                    mod_version=mod_ver,
                    updated_date=upd_date,
                    snapshot=copy.deepcopy(clean_row),
                )
                new_count += 1
                output_rows.append(clean_row)
                continue

            if prev.mod_version == mod_ver and prev.updated_date == upd_date:
                unchanged_count += 1
                stub: Dict[str, Any] = {
                    "id": rec_id,
                    "state": clean_row.get("state", ""),
                    "_context_delta": f"unchanged@mod{mod_ver}",
                }
                if clean_row.get("needs_attention"):
                    stub["needs_attention"] = True
                output_rows.append(stub)
            else:
                updated_count += 1
                changed_fields = [
                    k
                    for k, v in clean_row.items()
                    if prev.snapshot.get(k) != v
                ]
                self._records[rec_id] = _SnapshotEntry(
                    mod_version=mod_ver,
                    updated_date=upd_date,
                    snapshot=copy.deepcopy(clean_row),
                )
                delta_row = dict(clean_row)
                delta_row["_context_delta"] = f"updated@mod{mod_ver}"
                delta_row["_changed_fields"] = changed_fields
                output_rows.append(delta_row)

        compacted_bytes = len(json.dumps(output_rows, separators=(",", ":")).encode("utf-8"))
        saved_bytes = max(0, full_bytes - compacted_bytes)
        pct = round((saved_bytes / full_bytes) * 100.0, 1) if full_bytes > 0 else 0.0

        return {
            "records": output_rows,
            "_context_compaction": {
                "total_records": len(concept_rows),
                "new_records": new_count,
                "updated_records": updated_count,
                "unchanged_compact_stubs": unchanged_count,
                "full_concept_bytes": full_bytes,
                "compacted_bytes": compacted_bytes,
                "saved_tokens_est": max(0, saved_bytes // 4),
                "additional_reduction_pct": pct,
            },
        }


class SpeculativeDryRunVerifier:
    """Level-1 speculative policy verifier for mutating tool calls (`dry_run=True`)."""

    @staticmethod
    def verify_record_update(
        current_record: Dict[str, Any],
        proposed_updates: Dict[str, Any],
    ) -> Dict[str, Any]:
        changes: List[Dict[str, Any]] = []
        merged = copy.deepcopy(current_record)

        for field_name, new_val in proposed_updates.items():
            if field_name in {"record_id", "dry_run"} or new_val is None:
                continue
            old_val = current_record.get(field_name, "")
            if str(old_val) != str(new_val):
                changes.append(
                    {
                        "field": field_name,
                        "before": old_val,
                        "after": new_val,
                    }
                )
                merged[field_name] = new_val

        policy_warnings: List[str] = []
        downstream_impacts: List[str] = []

        # Enforce monotonic ordering across milestone_1_date <= ... <= milestone_4_date
        parsed_milestones: List[Tuple[str, date]] = []
        for m_field in MILESTONE_FIELDS:
            dt = _parse_iso_date(merged.get(m_field))
            if dt is not None:
                parsed_milestones.append((m_field, dt))

        for idx in range(len(parsed_milestones) - 1):
            earlier_name, earlier_dt = parsed_milestones[idx]
            later_name, later_dt = parsed_milestones[idx + 1]
            if earlier_dt > later_dt:
                policy_warnings.append(
                    f"CHRONOLOGY_VIOLATION: {earlier_name} ({earlier_dt.isoformat()}) "
                    f"occurs after {later_name} ({later_dt.isoformat()})"
                )

        # Require milestone_3_date before transitioning state to Complete
        proposed_state = str(merged.get("state") or "")
        if proposed_state == "Complete" and _parse_iso_date(merged.get("milestone_3_date")) is None:
            policy_warnings.append(
                "MISSING_COMPLETION_PREREQUISITE: Cannot transition state to 'Complete' "
                "when milestone_3_date is unscheduled"
            )

        old_target = compute_target_completion_date(current_record)
        new_target = compute_target_completion_date(merged)
        if old_target != new_target:
            downstream_impacts.append(
                f"target_completion_date recalculates from '{old_target}' to '{new_target}'"
            )

        return {
            "mode": "DRY_RUN_PREVIEW",
            "committed": False,
            "record_id": str(current_record.get("id") or ""),
            "mod_version_current": int(current_record.get("mod_version", 0)),
            "changes": changes,
            "downstream_impacts": downstream_impacts,
            "policy_warnings": policy_warnings,
            "policy_passed": len(policy_warnings) == 0,
        }


def generate_synthetic_wide_table(num_records: int = 20) -> List[Dict[str, Any]]:
    """Generates deterministic 100-column wide database records for offline evaluation."""
    rows: List[Dict[str, Any]] = []
    for i in range(1, num_records + 1):
        row: Dict[str, Any] = {
            "id": f"REC-{1000 + i:06d}",
            "title": f"Workflow Automation Deployment {i}",
            "summary": (
                f"Multi-stage workflow deployment {i} integrating automated triage, "
                "document extraction, and policy verification across regional operations."
            ),
            "segment": "Segment_Alpha" if i % 2 == 0 else "Segment_Beta",
            "state": "Active" if i % 5 != 0 else "On Hold",
            "owner": f"Contributor_{(i % 4) + 1}",
            "secondary_owner": f"Contributor_{((i + 1) % 4) + 1}",
            "linked_request_id": f"REQ-{2000 + i:06d}",
            "architecture_pattern": "Multi-Agent Coordinator with Policy Verifier",
            "needs_attention": i % 7 == 0,
            "milestone_1_date": "2026-10-01" if i % 3 != 0 else "",
            "milestone_2_date": "2026-10-08" if i % 3 != 0 else "",
            "milestone_3_date": "2026-10-18" if i % 4 == 0 else "",
            "milestone_4_date": "",
            "created_date": "2026-09-25",
            "updated_date": "2026-10-05" if i % 6 == 0 else "2026-10-24",
            "mod_version": 10 + (i % 5),
            "activity_journal_log": (
                "2026-10-20T14:22:00Z [user_1]: Completed initial schema validation and "
                "verified configuration endpoints across staging nodes.\n" * 6
            ),
        }
        # Populate 85 additional administrative/metadata columns typical of relational schemas
        for col_idx in range(1, 86):
            row[f"column_attr_{col_idx:03d}"] = {
                "value": f"attr_val_{i}_{col_idx}_9f8e7d6c5b4a3210",
                "display_value": f"Attribute Label {col_idx} for Record {i}",
            }
        rows.append(row)
    return rows


def run_multi_turn_benchmark(num_records: int = 20) -> Dict[str, Any]:
    """Benchmarks raw table payloads vs. Turn-1 NCM projection vs. Turn-2 @modN deltas."""
    ref_date = date(2026, 10, 28)
    wide_rows = generate_synthetic_wide_table(num_records=num_records)
    raw_bytes = len(json.dumps(wide_rows, separators=(",", ":")).encode("utf-8"))

    # Uncompacted table-first MCP (strips metadata dicts but keeps verbose journal logs)
    table_first_rows = [
        {k: v for k, v in r.items() if not k.startswith("column_attr_")}
        for r in wide_rows
    ]
    table_first_bytes = len(json.dumps(table_first_rows, separators=(",", ":")).encode("utf-8"))

    working_set = TemporalResidualWorkingSet(ttl_seconds=45.0)
    projected_rows = [project_record_concept(r, reference_date=ref_date) for r in wide_rows]

    turn1 = working_set.compact_list(projected_rows)
    turn1_bytes = turn1["_context_compaction"]["compacted_bytes"]

    # Turn 2: 1 record updated, remaining N-1 records unchanged
    wide_rows[0]["mod_version"] += 1
    wide_rows[0]["updated_date"] = "2026-10-28"
    wide_rows[0]["milestone_3_date"] = "2026-10-29"
    projected_turn2 = [project_record_concept(r, reference_date=ref_date) for r in wide_rows]
    turn2 = working_set.compact_list(projected_turn2)
    turn2_bytes = turn2["_context_compaction"]["compacted_bytes"]

    return {
        "num_records": num_records,
        "raw_table_tokens_est": raw_bytes // 4,
        "table_first_mcp_tokens_est": table_first_bytes // 4,
        "ncm_turn1_tokens_est": turn1_bytes // 4,
        "ncm_turn2_delta_tokens_est": turn2_bytes // 4,
        "turn1_reduction_vs_raw_pct": round((1.0 - (turn1_bytes / raw_bytes)) * 100.0, 1),
        "turn2_additional_reduction_pct": turn2["_context_compaction"]["additional_reduction_pct"],
    }


if __name__ == "__main__":
    summary = run_multi_turn_benchmark(num_records=20)
    print(json.dumps(summary, indent=2))
