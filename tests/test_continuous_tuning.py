"""Tuning selection, immutable parameter identities and reference separation."""
import copy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("continuous_tuning", ROOT / "tools/tune_continuous_solvers.py")
tuning = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tuning)


def inputs():
    return [dict(instance_id=name, n_variables=200 if index == 0 else 500,
                 npz=f"/objectives/{name}.npz", npz_sha256=str(index),
                 reference_objective=-100, reference_status="published_best_known")
            for index, name in enumerate(("a", "b"))]


def row(instance="a", energy=-100, **overrides):
    return dict(dict(instance=instance, energies=[energy] * 9, status="complete",
                     phase="screen", solver="lib_random_search", candidate="profile"), **overrides)


def test_all_five_protected_exact_and_large_size_profiles():
    settings = tuning.make_settings()
    baseline = tuning.read_json(ROOT / "configs/benchmark_solvers.json")["solvers"]
    plan = tuning.make_plan()
    for name in tuning.PROTECTED:
        assert settings["solvers"][name] == baseline[name]
        assert tuning.resolve_parameters(name, settings["solvers"][name], plan["candidates"][1],
                                         1000, plan) == baseline[name]
        large = tuning.resolve_parameters(name, settings["solvers"][name], plan["candidates"][1],
                                           10000, plan)
        assert {k: v for k, v in large.items() if k != "memory_limit_bytes"} == {
            k: v for k, v in baseline[name].items() if k != "memory_limit_bytes"}
        if "memory_limit_bytes" in baseline[name]:
            assert large["memory_limit_bytes"] == 8 * 1024**3
        else:
            assert "memory_limit_bytes" not in large
    mutable = set(settings["solvers"]) - set(tuning.PROTECTED)
    assert len(mutable) == 18
    for name in mutable:
        result = tuning.resolve_parameters(name, settings["solvers"][name], plan["candidates"][0],
                                           10000, plan)
        assert result["run_batch_size"] == result["runs"]
        if result.get("graph_block"):
            assert result["memory_limit_bytes"] is None
        elif name in tuning.NATIVE:
            assert result["memory_limit_bytes"] == 8 * 1024**3
        if "max_dense_variables" in result:
            assert result["max_dense_variables"] == 16384


def test_shipped_plan_and_configuration_hashes_match():
    plan = tuning.read_json(ROOT / "configs/continuous_tuning_plan.json")
    settings = tuning.read_json(ROOT / "configs/benchmark_solvers_20s.json")
    tuning.validate_plan(plan, settings)
    assert settings == tuning.make_settings()
    assert plan == tuning.make_plan()


def test_protected_change_is_rejected():
    settings = tuning.make_settings()["solvers"]
    settings["lib_tap_annealing"]["max_steps"] += 1
    with pytest.raises(ValueError, match="Protected parameters"):
        tuning.assert_protected(settings)


@pytest.mark.parametrize("size", [100, 200, 500, 1000, 5000, 10000])
def test_protected_memory_override_requires_explicit_large_profile(size):
    settings, plan = tuning.make_settings(), tuning.make_plan()
    resolved = {name: tuning.resolve_parameters(name, settings["solvers"][name],
                plan["candidates"][0], size, plan) for name in tuning.PROTECTED}
    tuning.assert_protected(resolved, n_variables=size, protected_memory_guard_override=True)
    if size >= 5000:
        with pytest.raises(ValueError, match="Protected parameters"):
            tuning.assert_protected(resolved)
        resolved["lib_tap_annealing"]["max_steps"] += 1
        with pytest.raises(ValueError, match="Protected parameters"):
            tuning.assert_protected(resolved, n_variables=size, protected_memory_guard_override=True)
    else:
        assert resolved["lib_tap_annealing"]["memory_limit_bytes"] == 536870912


def test_continuous_worker_api_import_and_checkpoint_contract():
    import inspect
    from qubo_benchmark.runtime.continuous_worker import CHECKPOINTS, PROTECTED, run_continuous_trial
    assert CHECKPOINTS == tuning.CHECKPOINTS
    assert set(PROTECTED) == set(tuning.PROTECTED)
    assert list(inspect.signature(run_continuous_trial).parameters) == ["worker", "job", "stop"]


def test_plan_rejects_changed_seed_and_protected_control_identity():
    plan, settings = tuning.make_plan(), tuning.make_settings()
    tuning.validate_plan(plan, settings)
    changed = copy.deepcopy(plan)
    changed["holdout_seeds"] = list(changed["training_seeds"])
    with pytest.raises(ValueError, match="fresh holdout"):
        tuning.validate_plan(changed, settings)
    changed = copy.deepcopy(plan)
    changed["protected_configuration_sha256"] = "bad-control"
    with pytest.raises(ValueError, match="Protected configuration hash"):
        tuning.validate_plan(changed, settings)


def test_paired_screen_is_108_and_holdout_at_most_72():
    plan, settings = tuning.make_plan(), tuning.make_settings()
    screen = tuning.tasks_for(plan, settings, inputs())
    assert len(screen) == 108
    assert set(t["seed"] for t in screen) == {101, 102}
    assert all(t["solver"] not in tuning.PROTECTED for t in screen)
    assert all("reference_objective" not in t for t in screen)
    selected = {s: "population64" for s in settings["solvers"] if s not in tuning.PROTECTED}
    holdout = tuning.tasks_for(plan, settings, inputs(), "holdout", selected)
    assert len(holdout) == 72
    assert set(t["seed"] for t in holdout) == {201, 202}
    assert set(t["id"] for t in holdout).isdisjoint(t["id"] for t in screen)
    assert all(t["parameters"]["run_batch_size"] == t["parameters"]["runs"] for t in screen)


def test_holdout_identical_baseline_is_not_rerun():
    plan, settings = tuning.make_plan(), tuning.make_settings()
    selected = {s: "profile" for s in settings["solvers"] if s not in tuning.PROTECTED}
    assert len(tuning.tasks_for(plan, settings, inputs(), "holdout", selected)) == 36


def test_exact_hits_signed_raw_gaps_tts99_and_failure_denominator():
    refs = {e["instance_id"]: e for e in inputs()}
    metrics = tuning.checkpoint_metrics([row(energy=-99), row(instance="b", energy=-100)], refs)
    assert metrics[-1]["mean_gap_raw"] == .5
    assert metrics[-1]["exact_bks_hit_probability"] == .5
    assert metrics[-1]["tts99_s"] == pytest.approx(132.8771237954945)
    failed = tuning.checkpoint_metrics([row(), row(instance="b", energy=None, status="failed")], refs)
    assert failed[-1]["mean_gap_raw"] is None
    assert failed[-1]["exact_bks_hit_probability"] == .5
    assert "ttt" not in str(metrics).lower()


def test_tts99_zero_hits_unavailable_one_hit_probability_uses_literal_limit():
    refs = {"a": inputs()[0]}
    assert tuning.checkpoint_metrics([row(energy=-99)], refs)[-1]["tts99_s"] is None
    assert tuning.checkpoint_metrics([row(energy=-101)], refs)[-1]["tts99_s"] == 0


def test_proven_optimum_improvement_is_validation_alarm():
    refs = {"a": dict(inputs()[0], reference_status="published_proven_optimum")}
    metrics = tuning.checkpoint_metrics([row(energy=-101)], refs)[-1]
    assert metrics["validation_alarms"] == 1
    assert metrics["mean_gap_raw"] is None
    assert metrics["exact_bks_hit_probability"] == 0


def test_selection_uses_only_declared_metrics_not_wall_or_near_target():
    plan = tuning.make_plan()
    refs = {e["instance_id"]: e for e in inputs()}
    rows = []
    for candidate, energy in (("profile", -99), ("population64", -100), ("late_freeze", -98)):
        rows.extend(row(instance=e["instance_id"], candidate=candidate, energy=energy,
                        actual_solve_wall_s=1 if candidate == "profile" else 100,
                        success_1pct=candidate == "profile") for e in inputs())
    selected, summary = tuning.select_winners(rows, refs, plan)
    assert selected == {"lib_random_search": "population64"}
    assert len(summary) == 3
    with pytest.raises(ValueError, match="Incomplete"):
        tuning.select_winners(rows[:-1], refs, plan)


def test_negative_bks_improvement_ranks_ahead_of_equal_reference():
    plan = tuning.make_plan()
    refs = {e["instance_id"]: e for e in inputs()}
    rows = []
    for candidate, energy in (("profile", -100), ("population64", -101), ("late_freeze", -99)):
        rows.extend(row(instance=e["instance_id"], candidate=candidate, energy=energy) for e in inputs())
    selected, summary = tuning.select_winners(rows, refs, plan)
    assert selected == {"lib_random_search": "population64"}
    improved = next(s for s in summary if s["candidate"] == "population64")
    assert improved["metrics"][-1]["mean_gap_raw"] == -1
    assert improved["metrics"][-1]["exact_bks_hit_probability"] == 1


def test_dense_padding_limit_changes_only_for_mutable_large_profiles():
    settings, plan = tuning.make_settings(), tuning.make_plan()
    for solver, original in settings["solvers"].items():
        if "max_dense_variables" not in original:
            continue
        assert solver not in tuning.PROTECTED
        small = tuning.resolve_parameters(solver, original, plan["candidates"][0], 1000, plan)
        large = tuning.resolve_parameters(solver, original, plan["candidates"][0], 10000, plan)
        assert small["max_dense_variables"] == original["max_dense_variables"]
        assert large["max_dense_variables"] == 16384
        assert {k: v for k, v in large.items() if k != "max_dense_variables"} == {
            k: v for k, v in small.items() if k != "max_dense_variables"}


def test_result_payload_is_compact_and_monotonic():
    result = tuning.normalize_result(dict(energies=[None, None, -1, -1, -2, -2, -2, -2, -2],
        status="complete", bitstring="not-retained", iterations=[1, 2], telemetry={},
        actual_solve_wall_s=20.1, stop_reason="deadline"), tuning.CHECKPOINTS)
    assert "bitstring" not in result and "iterations" not in result and "telemetry" not in result
    assert "actual_solve_wall_s" not in result and "stop_reason" not in result
    with pytest.raises(ValueError, match="nonincreasing"):
        tuning.normalize_result(dict(energies=[-1, 0] + [-1] * 7), tuning.CHECKPOINTS)
    with pytest.raises(ValueError, match="disappear"):
        tuning.normalize_result(dict(energies=[-1, None] + [-1] * 7), tuning.CHECKPOINTS)
    missing = tuning.normalize_result(dict(energies=[float("nan")] * 9), tuning.CHECKPOINTS)
    assert missing["energies"] == [None] * 9
    with pytest.raises(ValueError, match="finite"):
        tuning.normalize_result(dict(energies=[float("inf")] * 9), tuning.CHECKPOINTS)


def test_native_memory_estimate_includes_quadratic_storage():
    settings, plan = tuning.make_settings(), tuning.make_plan()
    params = settings["solvers"]["lib_random_search"]
    small = tuning.memory_estimate("lib_random_search", params, 1000, plan)
    large = tuning.memory_estimate("lib_random_search", params, 10000, plan)
    assert large > 12 * 10000**2 * 4
    assert large - 1024**3 > 50 * (small - 1024**3)


def test_config_resolution_does_not_mutate_base_or_use_reference():
    settings, plan = tuning.make_settings(), tuning.make_plan()
    params = settings["solvers"]["lib_exchange_cascade"]
    original = copy.deepcopy(params)
    first = tuning.resolve_parameters("lib_exchange_cascade", params, plan["candidates"][2], 10000, plan)
    assert first["freeze_fraction"] == .95
    assert params == original
    assert first["time_step"] == original["time_step"]
    assert tuning.digest(first) == tuning.digest(tuning.resolve_parameters(
        "lib_exchange_cascade", params, plan["candidates"][2], 10000, plan))


def test_frozen_resume_artifact_rejects_changes(tmp_path):
    path = tmp_path / "selected.json"
    tuning.freeze_json(path, {"lib_random_search": "profile"})
    tuning.freeze_json(path, {"lib_random_search": "profile"})
    with pytest.raises(ValueError, match="Frozen resume artifact"):
        tuning.freeze_json(path, {"lib_random_search": "population64"})
