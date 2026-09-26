import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "presslight_author_runner", Path(__file__).parents[1] / "scripts/run_presslight_author.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_phase_hold_includes_final_streak_and_keeps_intersections_separate():
    result = runner.phase_summary([[0, 1], [1, 1], [1, 2], [1, 3]])
    assert result[0]["longest_same_action_s"] == 90
    assert result[1]["longest_same_action_s"] == 60
    assert result[0]["action_counts"] == [1, 3, 0, 0]
    with pytest.raises(ValueError):
        runner.phase_summary([[0, 1], [1]])


def test_author_prefixed_data_path_resolves_to_pinned_input(tmp_path):
    job = tmp_path / "jobs/Hangzhou1/seed_0"
    relative = runner.author_data_path(tmp_path, "Hangzhou1", job)
    assert (job / ("./" + relative) / "roadnet_4_4.json").resolve() == (
        tmp_path / "inputs/Hangzhou1/roadnet_4_4.json")


def test_aggregate_uses_last_ten_per_seed_and_rejects_incomplete_group(tmp_path):
    tasks = []
    for seed in range(5):
        job = "jobs/Hangzhou1/seed_%d" % seed
        tasks.append({"flow": "Hangzhou1", "seed": seed, "job": job, "state": "completed"})
        rounds = [{"evaluation": {"test_avg_travel_time_over": 999 if r < 90 else 300 + seed},
                   "lifecycle": {"completion_rate": 0.8}} for r in range(100)]
        runner.write_json(tmp_path / job / "summary.json", {"seed": seed, "rounds": rounds})
    runner.aggregate(tmp_path, tasks)
    result = json.loads((tmp_path / "comparison.json").read_text())
    assert result["complete_groups"] == 1
    assert result["groups"][0]["att_mean_s"] == 302
    assert result["groups"][0]["att_sample_sd_s"] == pytest.approx(2.5**0.5)
    tasks[-1]["state"] = "running"
    runner.aggregate(tmp_path, tasks)
    assert json.loads((tmp_path / "comparison.json").read_text())["complete_groups"] == 0
