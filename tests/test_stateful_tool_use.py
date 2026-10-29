import unittest
from datetime import date

from evaluation.stateful_tool_use import (
    ConceptInvariantGate,
    SpeculativeDryRunVerifier,
    TemporalResidualWorkingSet,
    compute_stage_velocity,
    generate_synthetic_wide_table,
    project_record_concept,
    run_multi_turn_benchmark,
)


class StatefulToolUseNCMTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ref_date = date(2026, 10, 28)
        self.gate = ConceptInvariantGate()

    def test_level0_role_and_level1_slot_invariants(self) -> None:
        ok, err, spec = self.gate.validate(
            "update_record",
            caller_role="role_contributor",
            slots={"record_id": "REC-001001", "milestone_2_date": "2026-10-15"},
        )
        self.assertTrue(ok)
        self.assertIsNone(err)
        self.assertIsNotNone(spec)

        # Viewer role cannot invoke state-mutating concepts
        ok_viewer, err_viewer, _ = self.gate.validate(
            "update_record",
            caller_role="role_viewer",
            slots={"record_id": "REC-001001"},
        )
        self.assertFalse(ok_viewer)
        self.assertIn("not permitted", err_viewer or "")

        # Missing required slot rejected at Level 1
        ok_missing, err_missing, _ = self.gate.validate(
            "update_record",
            caller_role="role_contributor",
            slots={"state": "Active"},
        )
        self.assertFalse(ok_missing)
        self.assertIn("record_id", err_missing or "")

        # Hallucinated slot name rejected at Level 1
        ok_unknown, err_unknown, _ = self.gate.validate(
            "update_record",
            caller_role="role_contributor",
            slots={"record_id": "REC-001001", "stae": "Active"},
        )
        self.assertFalse(ok_unknown)
        self.assertIn("stae", err_unknown or "")

    def test_temporal_residual_context_deltas_across_turns(self) -> None:
        rows = generate_synthetic_wide_table(num_records=10)
        projected = [project_record_concept(r, reference_date=self.ref_date) for r in rows]
        working_set = TemporalResidualWorkingSet(ttl_seconds=45.0)

        turn1 = working_set.compact_list(projected)
        self.assertEqual(turn1["_context_compaction"]["new_records"], 10)
        self.assertEqual(turn1["_context_compaction"]["unchanged_compact_stubs"], 0)

        # Turn 2 with no changes: all 10 records collapse to @modN stubs
        turn2 = working_set.compact_list(projected)
        self.assertEqual(turn2["_context_compaction"]["unchanged_compact_stubs"], 10)
        self.assertGreater(
            turn2["_context_compaction"]["additional_reduction_pct"],
            70.0,
        )
        self.assertEqual(turn2["records"][0]["_context_delta"], "unchanged@mod11")

        # Turn 3 with 1 mutated record: emits updated@modN + _changed_fields for record 0
        rows[0]["mod_version"] = 12
        rows[0]["updated_date"] = "2026-10-28"
        rows[0]["milestone_2_date"] = "2026-10-12"
        projected_t3 = [project_record_concept(r, reference_date=self.ref_date) for r in rows]
        turn3 = working_set.compact_list(projected_t3)
        self.assertEqual(turn3["_context_compaction"]["updated_records"], 1)
        self.assertEqual(turn3["_context_compaction"]["unchanged_compact_stubs"], 9)
        self.assertEqual(turn3["records"][0]["_context_delta"], "updated@mod12")
        self.assertIn("milestone_2_date", turn3["records"][0]["_changed_fields"])

        # Escape hatch returns full concepts
        turn4_full = working_set.compact_list(projected_t3, include_unchanged=True)
        self.assertEqual(turn4_full["_context_compaction"]["unchanged_compact_stubs"], 0)

    def test_query_cache_hit_and_mutation_invalidation(self) -> None:
        now = [100.0]
        working_set = TemporalResidualWorkingSet(ttl_seconds=30.0, clock=lambda: now[0])
        sample = [{"id": "REC-001001", "state": "Active"}]

        self.assertIsNone(working_set.get_cached_query("segment=Alpha"))
        working_set.set_cached_query("segment=Alpha", sample)
        self.assertEqual(len(working_set.get_cached_query("segment=Alpha") or []), 1)

        # Write mutation invalidates query cache immediately
        working_set.invalidate_on_mutation("REC-001001")
        self.assertIsNone(working_set.get_cached_query("segment=Alpha"))

        # TTL expiration
        working_set.set_cached_query("segment=Alpha", sample)
        now[0] = 135.0
        self.assertIsNone(working_set.get_cached_query("segment=Alpha"))

    def test_speculative_dry_run_verifier_and_monotonic_milestones(self) -> None:
        record = {
            "id": "REC-001042",
            "state": "Active",
            "milestone_1_date": "2026-10-10",
            "milestone_2_date": "2026-10-15",
            "milestone_3_date": "",
            "milestone_4_date": "",
            "mod_version": 4,
        }

        # Valid update preview
        ok_preview = SpeculativeDryRunVerifier.verify_record_update(
            record,
            {"milestone_3_date": "2026-10-22", "dry_run": True},
        )
        self.assertEqual(ok_preview["mode"], "DRY_RUN_PREVIEW")
        self.assertFalse(ok_preview["committed"])
        self.assertTrue(ok_preview["policy_passed"])
        self.assertEqual(len(ok_preview["changes"]), 1)
        self.assertEqual(record["milestone_3_date"], "")  # original record untouched

        # Out-of-order milestone date caught at Level 1
        bad_order = SpeculativeDryRunVerifier.verify_record_update(
            record,
            {"milestone_2_date": "2026-10-01"},
        )
        self.assertFalse(bad_order["policy_passed"])
        self.assertTrue(
            any("CHRONOLOGY_VIOLATION" in w for w in bad_order["policy_warnings"])
        )

        # Completing without milestone_3_date caught at Level 1
        bad_complete = SpeculativeDryRunVerifier.verify_record_update(
            record,
            {"state": "Complete"},
        )
        self.assertFalse(bad_complete["policy_passed"])
        self.assertTrue(
            any("MISSING_COMPLETION_PREREQUISITE" in w for w in bad_complete["policy_warnings"])
        )

    def test_stage_velocity_and_benchmark_reductions(self) -> None:
        stalled = {
            "id": "REC-001099",
            "state": "Active",
            "milestone_1_date": "",
            "milestone_2_date": "",
            "milestone_3_date": "",
            "milestone_4_date": "",
            "created_date": "2026-09-15",
            "updated_date": "2026-10-01",
        }
        vel = compute_stage_velocity(stalled, reference_date=self.ref_date)
        self.assertIn("STALLED_UPDATES_14D", vel["velocity_flags"])
        self.assertIn("UNSCHEDULED_MILESTONES", vel["velocity_flags"])
        self.assertIn("TARGET_DATE_OVERDUE", vel["velocity_flags"])

        bench = run_multi_turn_benchmark(num_records=20)
        self.assertGreater(bench["turn1_reduction_vs_raw_pct"], 90.0)
        self.assertGreater(bench["turn2_additional_reduction_pct"], 75.0)


if __name__ == "__main__":
    unittest.main()
