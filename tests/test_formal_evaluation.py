import numpy as np

from cityflow_tsc.benchmark_formal_effects import ObjectiveRecorder
from cityflow_tsc.evaluate_formal_effects import aggregate, compose, errors, joint_metrics
from cityflow_tsc.run_formal_effects import command_environment


def test_sparse_composition_preserves_integer_physical_sums():
    factors = np.zeros((3, 272, 48), dtype=np.float32)
    factors[0, 0, 0] = 7
    factors[1, 0, 0] = -8
    incidence = np.array([[0, 0, 0], [1, 1, 0]], dtype=np.float32)
    prediction = compose(incidence, factors)
    assert prediction.dtype == np.float64
    assert prediction[0].sum() == 0 and prediction[1].sum() == -1
    stats = errors(np.array([0, 0]), np.array([2, -2]))
    assert stats["absolute_error_sum"] == 4 and stats["effect_l1_relative"] == 1
    assert errors(np.array([1]), np.array([0]))["effect_l1_relative"] is None


def test_decisions_include_reference_and_keep_negative_capture():
    truth = np.zeros((3, 272, 48), dtype=np.float32)
    truth[1, 0, 0], truth[2, 0, 0] = -10, 20
    prediction = np.zeros_like(truth); prediction[2, 0, 0] = -30
    root = {"joint": truth, "joint_sizes": np.array([0, 3, 4]),
            "joint_groups": np.array(["reference", "connected", "uniform"])}
    result = joint_metrics(prediction, root)
    assert result["regret"] == 30 and result["benefit_capture"] == -2
    assert result["worse_than_reference"]
    assert result["slices"]["size_4"]["decision"]["benefit_capture"] is None
    assert set(result["decisions"]) == {"30", "60", "90", "120", "180", "240"}


def test_group_averages_not_per_branch_weighted():
    rows = [{"cohort_id": c, "flow_id": f, "metrics": {"regret": r}}
            for c, f, r in [("a", "a1", 0), ("a", "a1", 2), ("a", "a2", 3), ("b", "b1", 10)]]
    result = aggregate(rows)["regret"]
    assert result["mean"] == 6 and result["root_count"] == 4
    assert errors(np.zeros((2,)), np.zeros((2,)))["nonzero_mae"] is None


def test_objective_recorder_counts_interior_and_pending_once():
    class Engine:
        def get_vehicle_speed(self):
            return {"road": 0., "interior": .05, "moving": 2.}
        def get_vehicles(self, include_waiting):
            assert include_waiting
            return ["road", "interior", "moving", "pending"]
    assert ObjectiveRecorder().observe(Engine()) == {"waiting": 3}


def test_launch_env_keeps_all_generated_caches_on_run(tmp_path):
    env = command_environment(tmp_path, 2)
    assert env["TMPDIR"] == str(tmp_path / "tmp")
    assert env["XDG_CACHE_HOME"] == str(tmp_path / "cache")
    assert env["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
