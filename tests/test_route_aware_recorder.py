import numpy as np

from cityflow_tsc.counterfactual.route_aware_recorder import RouteAwareRecorder
from .test_counterfactual import recorder_fixture


def run_disappearance(last_road, distance, exists_in_buffer=False):
    legacy, Engine = recorder_fixture()
    legacy.flows[0]["route"][-1] = last_road
    new = RouteAwareRecorder(legacy.g, legacy.flows)
    vid = "flow_0_0"
    start = Engine({vid: "in_0"}, {vid: 10.}, {vid: distance})
    new.observe(start)
    saved = new.context()
    new.restore_context(saved)
    end = Engine({}, {}, {}, pending=[vid] if exists_in_buffer else [])
    end.get_current_time = lambda: 2
    return new.observe(end)


def test_internal_destination_is_completion_not_boundary_exit():
    row = run_disappearance("in", 295.)
    assert row["unresolved_events"] == 0
    assert row["internal_trip_completion_count"] == 1
    assert row["trip_completion_count"] == 1
    assert row["exit_count"] == 0
    assert row["internal_trip_completion_by_lane"].tolist() == [1, 0]
    assert row["lane_events"].tolist() == [[0, 1], [0, 0]]
    assert np.all(row["movement_events"] == 0)


def test_wrong_destination_remains_unresolved():
    row = run_disappearance("out", 295.)
    assert row["unresolved_events"] == 1
    assert row["trip_completion_count"] == 0


def test_implausibly_early_deletion_remains_unresolved():
    row = run_disappearance("in", 100.)
    assert row["unresolved_events"] == 1
    assert row["trip_completion_count"] == 0


def test_vehicle_still_in_buffer_is_not_completed():
    row = run_disappearance("in", 295., exists_in_buffer=True)
    assert row["unresolved_events"] == 1
    assert row["trip_completion_count"] == 0


def test_boundary_exit_retains_original_meaning():
    legacy, Engine = recorder_fixture()
    new = RouteAwareRecorder(legacy.g, legacy.flows)
    first = Engine({"flow_0_0": "out_0"}, {"flow_0_0": 10.}, {"flow_0_0": 295.})
    new.observe(first)
    end = Engine({}, {}, {})
    end.get_current_time = lambda: 2
    row = new.observe(end)
    assert row["exit_count"] == row["trip_completion_count"] == 1
    assert row["internal_trip_completion_count"] == row["unresolved_events"] == 0


def test_window_keeps_all_route_completion_fields():
    from cityflow_tsc.counterfactual.recorder import stack_rows, window_view

    row = run_disappearance("in", 295.)
    arrays = stack_rows([row] * 10)
    windows = window_view(arrays)
    for key in ("internal_trip_completion_by_lane", "internal_trip_completion_count",
                "trip_completion_count"):
        np.testing.assert_array_equal(windows[key], np.stack([row[key] * 5] * 2))
        assert windows[key].dtype == np.int64
