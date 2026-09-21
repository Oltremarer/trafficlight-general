from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cityflow_tsc.baselines.codecs import LaneCodec, SOURCE_PHASES
from cityflow_tsc.baselines.observations import BaselineObservationBuilder
from cityflow_tsc.baselines.profiles import get_profile
from cityflow_tsc.baselines.rewards import BaselineReward, RewardAccumulator
from cityflow_tsc.config import ControlConfig
from cityflow_tsc.topology import load_network_spec
from cityflow_tsc.types import IntersectionSpec, MovementSpec, NetworkSnapshot, NetworkSpec


FIXTURES = Path(__file__).parent / "fixtures"
CANONICAL = FIXTURES / "roadnet_baseline_four_arms.json"


def canonical_network(reordered=False):
    phases = (3, 1, 4, 2) if reordered else (1, 2, 3, 4)
    return load_network_spec(CANONICAL, ControlConfig(green_phase_ids=phases))


def snapshot(network, time_s=0, factor=1):
    incoming = tuple(f"in_{side}_{idx}" for side in "WENS" for idx in (2, 0, 1))
    counts = {lane: factor * (j + 1) * 10 for j, lane in enumerate(incoming)}
    waiting = {lane: factor * (j + 1) for j, lane in enumerate(incoming)}
    for side, base in zip("WENS", (1, 4, 7, 10)):
        for idx in range(3):
            counts[f"out_{side}_{idx}"] = (base + idx) * 10
            waiting[f"out_{side}_{idx}"] = base + idx
    return NetworkSnapshot(time_s, counts, waiting, {},
                           {lane: () for inter in network.intersections for lane in inter.incoming_lanes}, {})


def observe(builder, snap, action=0):
    n = builder.network.num_intersections
    return builder.build(snap, np.full(n, action, dtype=np.int64), np.zeros(n, dtype=np.int8), np.zeros(n)).baseline_view


def test_canonical_lane_and_phase_order_is_derived_from_links():
    network = canonical_network(reordered=True)
    builder = BaselineObservationBuilder(network, "colight", CANONICAL)
    expected_lanes = tuple(f"in_{side}_{idx}" for side in "WENS" for idx in (2, 0, 1))
    assert builder.codec.lane_ids == (expected_lanes,)
    np.testing.assert_array_equal(builder.codec.source_action_to_local[0], [1, 3, 0, 2, -1, -1, -1, -1])
    view = observe(builder, snapshot(network), action=0)
    np.testing.assert_array_equal(view.source_action_to_local, builder.codec.source_action_to_local)
    np.testing.assert_array_equal(view.phase_encoding[0], SOURCE_PHASES[2])
    np.testing.assert_array_equal(view.lane_features[0, :, 0], np.arange(1, 13) * 10)
    # Local action 0 is WL/EL, despite engine phase 3 and source action 2.
    np.testing.assert_array_equal(np.flatnonzero(view.phase_lane_mask[0, 0]), [0, 2, 3, 5, 8, 11])
    np.testing.assert_array_equal(view.features[0, :8], SOURCE_PHASES[2])
    assert view.features.shape == (1, 20)
    assert view.node_ids == network.intersection_ids


def test_general_and_efficient_pressure_use_whole_destination_road():
    network = canonical_network()
    snap = snapshot(network)
    # Canonical incoming lanes' destination sides, determined by roadLinks.
    targets = ("N", "E", "S", "S", "W", "N", "E", "S", "W", "W", "N", "E")
    road_sums = {"W": 6, "E": 15, "N": 24, "S": 33}
    incoming = np.arange(1, 13)
    total = np.asarray([road_sums[target] for target in targets])
    general = observe(BaselineObservationBuilder(network, "presslight", CANONICAL), snap)
    efficient = observe(BaselineObservationBuilder(network, "e-presslight", CANONICAL), snap)
    vehicles = observe(BaselineObservationBuilder(network, "mplight", CANONICAL), snap)
    np.testing.assert_allclose(general.lane_features[0, :, 0], incoming - total)
    np.testing.assert_allclose(efficient.lane_features[0, :, 0], incoming - total / 3)
    np.testing.assert_allclose(vehicles.lane_features[0, :, 0], 10 * (incoming - total))
    # WL only lane-links to one output lane, but pressure subtracts all 3 output lanes.
    assert general.lane_features[0, 0, 0] == 1 - (7 + 8 + 9)


def test_advanced_running_window_boundary_and_feature_blocks():
    network = canonical_network()
    snap = snapshot(network)
    lane_vehicles = dict(snap.lane_vehicles)
    lane_vehicles["in_W_2"] = ("at_edge", "inside", "outside", "waiting")
    snap = replace(snap, lane_vehicles=lane_vehicles,
                   vehicle_distances={"at_edge": 333, "inside": 400, "outside": 332.9, "waiting": 499},
                   vehicle_speeds={"at_edge": 0.11, "inside": 5, "outside": 5, "waiting": 0.1})
    builder = BaselineObservationBuilder(network, "a-colight", CANONICAL)
    view = observe(builder, snap)
    assert view.lane_features.shape == (1, 12, 2)
    assert view.lane_features[0, 0, 1] == 2
    assert view.lane_features[..., 1].sum() == 2
    np.testing.assert_array_equal(view.features[:, 8:20], view.lane_features[..., 0])
    np.testing.assert_array_equal(view.features[:, 20:32], view.lane_features[..., 1])
    with pytest.raises(ValueError, match="position/speed"):
        observe(builder, replace(snap, vehicle_distances={}))
    with pytest.raises(ValueError, match="per-lane vehicle IDs"):
        observe(builder, replace(snap, lane_vehicles={}))


def test_old_movement_observation_is_preserved_and_views_do_not_alias():
    network = canonical_network()
    builder = BaselineObservationBuilder(network, "colight", CANONICAL)
    args = (snapshot(network), np.array([0]), np.array([0]), np.array([0.0]))
    observation = builder.build(*args)
    assert builder.schema_id == builder.base.schema_id
    assert observation.baseline_view.schema_id == builder.view_schema_id
    assert observation.features.shape == (1, 12, 6)
    assert observation.baseline_view.features.shape == (1, 20)
    observation.baseline_view.phase_lane_mask[:] = False
    assert builder.build(*args).baseline_view.phase_lane_mask.any()


def test_configure_loads_static_codec_once_and_unsupported_layout_rejected(tmp_path):
    network = canonical_network()
    builder = BaselineObservationBuilder(network, "colight")
    with pytest.raises(ValueError, match="configure"):
        observe(builder, snapshot(network))
    scenario = SimpleNamespace(roadnet_path=CANONICAL)
    builder.configure(scenario)
    codec = builder.codec
    builder.configure(scenario)
    builder.reset()
    assert builder.codec is codec
    two = load_network_spec(FIXTURES / "roadnet_two_intersections.json", ControlConfig(green_phase_ids=(1, 2)))
    with pytest.raises(ValueError, match="four incoming"):
        BaselineObservationBuilder(two, "colight", FIXTURES / "roadnet_two_intersections.json")
    raw = json.loads(CANONICAL.read_text())
    raw["intersections"][0]["roadLinks"][0]["laneLinks"][0]["startLaneIndex"] = 0
    path = tmp_path / "shared_lane.json"
    path.write_text(json.dumps(raw))
    invalid = load_network_spec(path, ControlConfig())
    with pytest.raises(ValueError, match="dedicated"):
        BaselineObservationBuilder(invalid, "colight", path)


def test_generic_supports_existing_two_intersection_fixture():
    network = load_network_spec(FIXTURES / "roadnet_two_intersections.json", ControlConfig(green_phase_ids=(1, 2)))
    builder = BaselineObservationBuilder(network, "ippo")
    counts = {lane: 4 for inter in network.intersections for lane in inter.incoming_lanes}
    queues = {lane: 2 for lane in counts}
    view = observe(builder, NetworkSnapshot(0, counts, queues, {}))
    assert view.features.shape == (2, 6)  # two action bits, two counts, two queues
    np.testing.assert_array_equal(view.lane_features[..., 0], 4)
    np.testing.assert_array_equal(view.lane_features[..., 1], 2)
    assert view.phase_lane_mask.shape == (2, 2, 2)
    assert view.lane_mask.all()


def directed_network():
    specs = []
    for i, (start_road, end_road, point) in enumerate([
        ("boundary_0", "road_0_1", (0, 0)),
        ("road_0_1", "boundary_1", (100, 0)),
        ("boundary_2", "boundary_3", (1, 0)),
    ]):
        movement = MovementSpec(0, "go_straight", start_road, end_road, (start_road + "_0",), (end_road + "_0",))
        specs.append(IntersectionSpec(i, f"i{i}", point, movement.incoming_lanes, movement.outgoing_lanes,
                                      (movement,), (1,), np.ones((1, 1), dtype=np.bool_)))
    return NetworkSpec(tuple(specs), np.full((3, 4), -1), np.zeros((3, 4), dtype=np.bool_), 1, 1)


def test_geometric_knn_and_incoming_road_graph_have_distinct_semantics():
    network = directed_network()
    codec = LaneCodec.build(network, "generic")
    knn, kmask = codec.neighbors(network, "knn", top_k=2)
    road, rmask = codec.neighbors(network, "road")
    np.testing.assert_array_equal(knn[0], [0, 2])
    assert kmask.all()
    assert road[0, rmask[0]].tolist() == [0]
    assert road[1, rmask[1]].tolist() == [1, 0]  # directed source 0 -> destination 1
    assert road[2, rmask[2]].tolist() == [2]
    assert np.all(road[~rmask] == -1)


def test_reward_formulas_stay_local_even_when_learner_requests_global_reward():
    network = load_network_spec(FIXTURES / "roadnet_two_intersections.json", ControlConfig(green_phase_ids=(1, 2)))
    snap = NetworkSnapshot(0, {}, {"road_w_a_0": 9, "road_b_a_0": 3,
                                  "road_a_b_0": 4, "road_a_w_0": 2,
                                  "road_e_b_0": 2, "road_b_e_0": 1}, {})
    np.testing.assert_allclose(BaselineReward(network, "presslight").compute(snap), [-1.5, -0.5])
    np.testing.assert_allclose(BaselineReward(network, "colight").compute(snap), [-3, -1.5])
    np.testing.assert_allclose(BaselineReward(network, "idqn").compute(snap), [-6, -3])
    # The PPO learner, not the environment, owns global reward aggregation.
    np.testing.assert_allclose(BaselineReward(network, "ippo").compute(snap), [-12, -6])


@pytest.mark.parametrize("window,expected", [("last", -7), ("pre_mean", -2.5),
                                               ("post_mean", -6), ("sum", -10), ("integral", -24)])
def test_reward_windows_include_every_tick_and_respect_elapsed_time(window, expected):
    network = directed_network()
    profile = replace(get_profile("idqn"), reward_kind="queue", reward_factor=-1)
    reward = BaselineReward(network, profile)
    def at(t, count):
        return NetworkSnapshot(t, {}, {inter.incoming_lanes[0]: count for inter in network.intersections}, {})
    start, middle, end = at(0, 1), at(1, 3), at(4, 7)
    accumulator = RewardAccumulator(reward, window)
    accumulator.begin()
    accumulator.observe_tick(start, middle, 1)
    accumulator.observe_tick(middle, end, 3)
    np.testing.assert_allclose(accumulator.finish(end), expected)
    np.testing.assert_allclose(accumulator.last_components["incoming_queue"], -expected)
    accumulator.begin()
    assert accumulator.last_components == {}
    np.testing.assert_allclose(accumulator.finish(start), -1)
    np.testing.assert_allclose(accumulator.last_components["incoming_queue"], 1)
    accumulator.reset()
    assert accumulator.last_components == {}
    with pytest.raises(ValueError, match="positive"):
        accumulator.observe_tick(start, middle, 0)


def test_reward_components_use_same_window_before_factor_and_absolute_reduction():
    network = canonical_network()
    profile = replace(get_profile("presslight"), reward_factor=-0.25)
    reward = BaselineReward(network, profile)
    inter = network.intersections[0]
    def at(t, incoming, outgoing):
        queues = {lane: incoming for lane in inter.incoming_lanes}
        queues.update({lane: outgoing for lane in inter.outgoing_lanes})
        return NetworkSnapshot(t, {}, queues, {})
    start, middle, end = at(0, 1, 2), at(1, 3, 1), at(4, 4, 5)
    raw = reward.components(start)
    for name, expected in {"incoming_queue": 12, "outgoing_queue": 24,
                           "absolute_queue_pressure": 12, "queue_mean_road": 1}.items():
        np.testing.assert_allclose(raw[name], [expected])
    np.testing.assert_allclose(reward.compute(start), [-3])
    accumulator = RewardAccumulator(reward, "pre_mean")
    accumulator.begin()
    accumulator.observe_tick(start, middle, 1)
    accumulator.observe_tick(middle, end, 3)
    # Weight the start/middle values 1:3, including abs pressure BEFORE averaging.
    # Incoming=30, outgoing=15, abs pressure=21 (not abs(30-15)), road mean=2.5.
    final_reward = accumulator.finish(end)
    for name, expected in {"incoming_queue": 30, "outgoing_queue": 15,
                           "absolute_queue_pressure": 21, "queue_mean_road": 2.5}.items():
        np.testing.assert_allclose(accumulator.last_components[name], [expected])
    np.testing.assert_allclose(final_reward, [-5.25])
    np.testing.assert_allclose(final_reward, accumulator.last_components["absolute_queue_pressure"] * -0.25)


def test_libsignal_queue_mean_weights_roads_equally_with_unequal_lane_counts():
    first = MovementSpec(0, "go_straight", "one_lane_road", "exit_road",
                         ("one_lane_road_0",), ("exit_road_0",))
    second = MovementSpec(1, "go_straight", "three_lane_road", "exit_road",
                          ("three_lane_road_0", "three_lane_road_1", "three_lane_road_2"),
                          ("exit_road_0",))
    inter = IntersectionSpec(0, "i0", (0, 0), first.incoming_lanes + second.incoming_lanes,
                             ("exit_road_0",), (first, second), (1,),
                             np.ones((1, 2), dtype=np.bool_))
    network = NetworkSpec((inter,), np.array([[0]]), np.array([[True]]), 2, 1)
    snap = NetworkSnapshot(0, {}, {"one_lane_road_0": 9, "three_lane_road_0": 0,
                                   "three_lane_road_1": 3, "three_lane_road_2": 6}, {})
    # -(9 + mean(0,3,6))/2 = -6; flat lane mean would incorrectly give -4.5.
    np.testing.assert_allclose(BaselineReward(network, "idqn").compute(snap), [-6])
    np.testing.assert_allclose(BaselineReward(network, "libsignal-mplight").compute(snap), [-6])
    np.testing.assert_allclose(BaselineReward(network, "ippo").compute(snap), [-18])


@pytest.mark.parametrize("name", ["frap", "libsignal-mplight"])
def test_libsignal_pairs_exclude_right_turns_and_preserve_shared_demand_relations(tmp_path, name):
    raw = json.loads(CANONICAL.read_text())
    # Add a single-approach WL+WT phase, overlapping with two of the original four.
    raw["intersections"][0]["trafficLight"]["lightphases"].append(
        {"time": 30, "availableRoadLinks": [0, 1, 2, 5, 8, 11]})
    path = tmp_path / "five_phases.json"
    path.write_text(json.dumps(raw))
    network = load_network_spec(path, ControlConfig(green_phase_ids=(1, 2, 3, 4, 5)))
    builder = BaselineObservationBuilder(network, name, path)
    view = observe(builder, snapshot(network))
    pairs = view.phase_lane_mask[0]
    lanes = builder.codec.lane_ids[0]
    actual = [{lanes[i] for i in np.flatnonzero(row)} for row in pairs]
    assert actual == [{"in_W_0", "in_E_0"}, {"in_N_0", "in_S_0"},
                      {"in_W_2", "in_E_2"}, {"in_N_2", "in_S_2"}, {"in_W_2", "in_W_0"}]
    expected = np.zeros((5, 5), dtype=bool)
    expected[[0, 2], 4] = True
    expected[4, [0, 2]] = True
    np.testing.assert_array_equal((pairs[:, None] & pairs[None, :]).sum(-1) == 1, expected)
    # The environment's physical phase still includes its permissive right turns.
    assert network.intersections[0].phase_movement_mask[0].sum() == 6


@pytest.mark.parametrize("kind", ["one_demand", "shared_turn"])
def test_libsignal_pairs_reject_unsupported_demand_semantics(tmp_path, kind):
    raw = json.loads(CANONICAL.read_text())
    inter = raw["intersections"][0]
    if kind == "one_demand":
        inter["trafficLight"]["lightphases"][1]["availableRoadLinks"].remove(4)
    else:
        inter["roadLinks"][0]["laneLinks"][0]["startLaneIndex"] = 0
    path = tmp_path / "unsupported.json"
    path.write_text(json.dumps(raw))
    network = load_network_spec(path, ControlConfig())
    with pytest.raises(ValueError, match="phase_pairs require"):
        BaselineObservationBuilder(network, "frap", path)
