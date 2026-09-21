import json
from types import SimpleNamespace

import numpy as np
import pytest

from cityflow_tsc.counterfactual.plan import make_branches, continuation
from cityflow_tsc.counterfactual.recorder import PhysicalRecorder, SignalRunner, stack_rows, window_view, counts16


def test_branch_counts_and_references():
    adjacency = [set() for _ in range(16)]
    for i in range(16):
        for j in range(16):
            if abs(i // 4 - j // 4) + abs(i % 4 - j % 4) == 1:
                adjacency[i].add(j)
    a0 = [i % 4 for i in range(16)]
    full, pairs = make_branches("root", a0, True, adjacency, [0, 1, 4, 5])
    assert len(pairs) == 120
    assert 1193 <= len(full) <= 1382
    assert sum("single" in r["roles"] for r in full) == 48
    assert sum("pair" in r["roles"] for r in full) == 1080
    assert sum("panel_exact" in r["roles"] for r in full) == 256
    assert sum(any(s.startswith("joint_") for s in r["roles"]) for r in full) == 64
    ids = {r["branch_id"] for r in full}
    assert len(ids) == len(full)
    for r in full:
        assert r["baseline_id"] in ids
        assert set(r["single_ids"] + r["pair_ids"]) <= ids
        assert r["compositional_holdout"] == (len(r["changed_intersections"]) >= 3)
        assert r["requests"][1:] == continuation(a0, a0).tolist()[1:]
    partial, pairs = make_branches("partial", a0, False, adjacency, [0, 1, 4, 5])
    assert len(pairs) == 32 and len(partial) == 337
    assert not any(r["compositional_holdout"] for r in partial)


def test_signal_runner_switch_and_hold():
    class Backend:
        def __init__(self):
            self.t = 0
            self.events = []
            self.engine = self

        def set_phase(self, iid, p):
            self.events.append((self.t, iid, p))

        def next_step(self):
            self.t += 1

    backend = Backend()
    recorder = SimpleNamespace(observe=lambda e: {"time_s": np.asarray(e.t)})
    runner = SignalRunner(backend, SimpleNamespace(intersections=["A", "B"]), recorder)
    runner.initialize()
    rows = stack_rows(runner.step([1, 0]))
    assert rows["phase_used"][:5].tolist() == [[0, 1]] * 5
    assert rows["phase_used"][5:].tolist() == [[2, 1]] * 25
    assert backend.events == [(0, "A", 1), (0, "B", 1), (0, "A", 0), (5, "A", 2)]
    assert runner.elapsed.tolist() == [25, 30]
    held = stack_rows(runner.step([1, 0]))
    assert held["phase_elapsed_start_s"][0].tolist() == [25, 30]
    assert runner.elapsed.tolist() == [55, 60]


def recorder_fixture():
    geometry = SimpleNamespace(
        lanes=["in_0", "out_0"], lane_index={"in_0": 0, "out_0": 1},
        intersections=["A"], movements=["m"], boundaries=["in"], boundary_index={"in": 0},
        lengths=np.array([300., 300.]), ends=np.array([100., 100.]), speeds=np.array([10., 10.]),
        links={"link": (0, 0, 0, 1)}, lane_roads=["in", "out"], exit_roads={"out"},
        road_pair_movement={("in", "out"): 0})
    flows = [{"route": ["in", "out"], "vehicle": {"maxSpeed": 10., "length": 5.}}]

    class Engine:
        def __init__(self, locations, speeds, distances, pending=()):
            self.locations, self.speeds, self.distances, self.pending = locations, speeds, distances, pending

        def get_lane_vehicles(self):
            return {lid: [vid for vid, loc in self.locations.items() if loc == lid] for lid in geometry.lanes}

        def get_vehicle_speed(self):
            return self.speeds

        def get_vehicle_distance(self):
            return self.distances

        def get_vehicles(self, include_waiting=False):
            return list(self.locations) + (list(self.pending) if include_waiting else [])

        def get_vehicle_info(self, vid):
            return {"drivable": self.locations[vid]}

        def get_current_time(self):
            return 1

    return PhysicalRecorder(geometry, flows), Engine


def test_physical_thresholds_segments_and_pending():
    recorder, Engine = recorder_fixture()
    ids = [f"flow_0_{i}" for i in range(4)]
    engine = Engine(dict.fromkeys(ids, "in_0"), dict(zip(ids, [0., .1, 2., 10.])),
                    dict(zip(ids, [0., 100., 200., 300.])), pending=["flow_0_4"])
    row = recorder.observe(engine)
    assert row["lane_n"].tolist() == [[1, 1, 2], [0, 0, 0]]
    assert row["lane_q"].tolist() == [[1, 0, 0], [0, 0, 0]]
    assert row["boundary_pending"].tolist() == [1]
    assert row["receiving_unavailable"].tolist() == [1, 0]
    assert row["lane_tail"][1].tolist() == [0., 0.]
    engine.distances[ids[0]] = 10.
    assert recorder.observe(engine)["receiving_unavailable"][0] == 1
    engine.speeds[ids[0]] = 2.
    assert recorder.observe(engine)["receiving_unavailable"][0] == 0
    engine.speeds[ids[0]] = 0.
    engine.distances[ids[0]] = 10.001
    assert recorder.observe(engine)["receiving_unavailable"][0] == 0


def test_crossings_and_window_sums():
    recorder, Engine = recorder_fixture()
    vid = "flow_0_0"
    recorder.observe(Engine({vid: "in_0"}, {vid: 0.}, {vid: 299.}))
    release = recorder.observe(Engine({vid: "link"}, {vid: 0.}, {vid: 1.}))
    assert release["movement_events"].tolist() == [[1, 0]]
    receive = recorder.observe(Engine({vid: "out_0"}, {vid: 0.}, {vid: 1.}))
    assert receive["movement_events"].tolist() == [[0, 1]]
    exit_row = recorder.observe(Engine({}, {}, {}))
    assert exit_row["exit_count"] == 1
    recorder.restore_context({"previous_locations": {vid: "in_0"}, "previous_all": [vid]})
    direct = recorder.observe(Engine({vid: "out_0"}, {vid: 0.}, {vid: 1.}))
    assert direct["movement_events"].tolist() == [[1, 1]]
    windows = window_view(stack_rows([direct] * 5))
    assert windows["lane_wait_vehicle_s"].tolist() == [[0, 5]]
    assert windows["movement_events"].tolist() == [[[5, 5]]]
    assert windows["lane_wait_vehicle_s"].dtype == np.int64


def test_count_overflow_is_not_silently_wrapped():
    with pytest.raises(OverflowError):
        counts16([65536])


def test_new_task_completion_registers_once(tmp_path, monkeypatch):
    from cityflow_tsc import collect_counterfactual as collector

    # Exercise task finalization without starting CityFlow or collecting data.
    flow = tmp_path / "flow.json"
    roadnet = tmp_path / "roadnet.json"
    flow.write_text("[]")
    roadnet.write_text("{}")
    np.savez(tmp_path / "trajectory.npz", actions=np.zeros((0, 16), dtype=np.uint8))
    panel_ids = ["intersection_1_1", "intersection_1_2", "intersection_2_1", "intersection_2_2"]
    monkeypatch.setattr(collector, "Geometry", lambda _: SimpleNamespace(node_index=dict(zip(panel_ids, range(4)))))
    closed = []
    monkeypatch.setattr(collector, "CityFlowBackend", lambda _: SimpleNamespace(
        reset=lambda _: None, close=lambda: closed.append(True)))
    monkeypatch.setattr(collector, "PhysicalRecorder", lambda *args: object())
    monkeypatch.setattr(collector, "SignalRunner", lambda *args: SimpleNamespace(initialize=lambda: None))
    monkeypatch.setattr(collector, "create_roots", lambda *args: ([], {}))
    task = {"task_id": "new_task", "ordinal": 0, "manifest_path": str(tmp_path / "manifest.json"),
            "manifest": {"scenario": {"flow_path": str(flow), "roadnet_path": str(roadnet), "seed": 0}}}

    result = collector.collect_task(str(tmp_path), task, 32)

    assert result["task_id"] == "new_task"
    assert json.loads((tmp_path / "indexes/new_task.complete.json").read_text()) == result
    assert json.loads((tmp_path / "logs/new_task.progress.json").read_text())["stage"] == "complete"
    assert not (tmp_path / "logs/new_task.error.json").exists()
    assert closed == [True]


def test_completed_task_resume_repairs_stale_progress_without_recollecting(tmp_path, monkeypatch):
    from cityflow_tsc import collect_counterfactual as collector
    from cityflow_tsc.counterfactual.writer import atomic_json

    result = {"task_id": "saved_task", "root_count": 5, "branch_count": 2730,
              "unresolved_events": 4, "wall_time_s": 12., "finished_at": "earlier"}
    completed = tmp_path / "indexes/saved_task.complete.json"
    atomic_json(completed, result)
    original_bytes = completed.read_bytes()
    original_mtime = completed.stat().st_mtime_ns
    atomic_json(tmp_path / "logs/saved_task.progress.json", {"stage": "failed", "error": "old registration error"})

    def forbidden_backend(*args):
        raise AssertionError("Completed tasks must not restart CityFlow")

    monkeypatch.setattr(collector, "CityFlowBackend", forbidden_backend)
    # Payload validation is exercised by the concrete fixtures in bounded tests.
    validated = []
    monkeypatch.setattr(collector, "verify_completed_task", lambda *args: validated.append(args))
    assert collector.collect_task(str(tmp_path), {"task_id": "saved_task"}, 32) == result
    assert len(validated) == 1
    progress = json.loads((tmp_path / "logs/saved_task.progress.json").read_text())
    assert progress["stage"] == "complete"
    assert progress["branch_count"] == 2730
    assert "error" not in progress
    assert completed.read_bytes() == original_bytes
    assert completed.stat().st_mtime_ns == original_mtime
