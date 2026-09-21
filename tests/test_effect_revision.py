import numpy as np
import torch

from cityflow_tsc.counterfactual.timed_plan import formal_t1_plan, reference_requests, timing_plan
from cityflow_tsc.collect_timed_effects import select_roots
from cityflow_tsc.effect_model.revision import RevisedEffectModel, checkpoint_key, revision_loss, sequence_bank
from cityflow_tsc.effect_model.model import EffectModel, effect_loss
from cityflow_tsc.counterfactual.recorder import SignalRunner
from cityflow_tsc.counterfactual.timing_analysis import panel_decision, waiting_windows_240


def grid():
    return [[j for j in range(16) if abs(i // 4 - j // 4) + abs(i % 4 - j % 4) == 1] for i in range(16)]


def static():
    return {"adjacency": np.zeros((8, 272, 272), dtype=np.float32),
            "relative": np.ones((16, 272, 11), dtype=np.float32),
            "permission": np.ones((16, 5, 272), dtype=np.float32)}


def test_formal_full_sequence_counts_and_current_phase_hold():
    base = np.arange(16) % 4
    plan = formal_t1_plan("formal", base, grid())
    rows = plan["branches"]
    assert len(rows) == 2049 and len(plan["candidate_ids"]) == 65
    assert sum(len(r["changed_intersections"]) == 1 for r in rows) == 64
    assert sum(len(r["changed_intersections"]) == 2 for r in rows) == 1920
    assert len({str(r["requests"]) for r in rows}) == 2049
    lookup = {r["branch_id"]: r for r in rows}
    assert plan["candidate_ids"][0] == rows[0]["branch_id"]
    ref = reference_requests(base)
    held = next(r for r in rows if r["changed_intersections"] == [0] and r["plan_ids"][0] == 1)
    assert held["requests"][0][0] == base[0]
    assert held["requests"][1][0] == base[0] != ref[1, 0]
    np.testing.assert_array_equal(np.array(held["requests"])[3:], ref[3:])
    for row in rows:
        changed = np.flatnonzero(np.any(np.array(row["requests"]) != ref, axis=0)).tolist()
        assert row["changed_intersections"] == changed
        for sid in row["single_ids"]:
            one = lookup[sid]; node = one["changed_intersections"][0]
            np.testing.assert_array_equal(np.array(one["requests"])[:, node], np.array(row["requests"])[:, node])


def test_timing_offsets_pair_strata_and_reference_correctness():
    base = np.arange(16) % 4
    plan = timing_plan("timing", base, grid())
    assert len(plan["panels"]) == 48 and plan["branch_count"] <= 912
    assert [sum(p["stratum"] == s for p in plan["selected_pairs"]) for s in ("adjacent", "two_hop", "far")] == [8, 4, 4]
    lookup = {r["branch_id"]: r for r in plan["branches"]}
    for panel in plan["panels"]:
        assert len(panel["candidate_ids"]) == (25 if panel["timing"] == "T1" else 16)
        if panel["timing"] == "T2":
            i = panel["B"]
            assert (base[i] + 2) % 4 not in map(int, panel["single_B"])
            for phase, sid in panel["single_B"].items():
                requests = np.array(lookup[sid]["requests"])
                assert requests[2, i] == int(phase)
                np.testing.assert_array_equal(requests[:2], reference_requests(base)[:2])
        for combo in panel["combinations"]:
            pair = np.array(lookup[combo["branch_id"]]["requests"])
            for sid in combo["single_ids"]:
                one = lookup[sid]; node = one["changed_intersections"][0]
                np.testing.assert_array_equal(np.array(one["requests"])[:, node], pair[:, node])


def test_natural_loss_and_gate_loss():
    pred = torch.zeros((1, 272, 36), requires_grad=True)
    target = torch.zeros_like(pred); target[0, 0, 0] = 2
    scale, total = torch.ones((272, 1)), torch.tensor(2.)
    loss = revision_loss(pred, target, scale, total, "A1")
    assert torch.allclose(loss, torch.tensor(2 / (272 * 36) + .05))
    logits = torch.zeros_like(pred, requires_grad=True)
    gated = revision_loss(pred, target, scale, total, "A2", logits)
    assert torch.allclose(gated, loss + .01 * torch.tensor(2.).log())
    gated.backward()
    assert torch.isfinite(pred.grad).all() and torch.isfinite(logits.grad).all()
    assert revision_loss(pred, target, scale, total, "A0") == effect_loss(pred, target, scale, total)


def test_a0_original_initialization_and_a2_zero_product():
    torch.set_num_threads(2)
    torch.manual_seed(42); old = EffectModel(static())
    torch.manual_seed(42); new = RevisedEffectModel(static(), "A0")
    assert all(torch.equal(value, new.state_dict()[key]) for key, value in old.state_dict().items())
    model = RevisedEffectModel(static(), "A2").eval()
    inputs = torch.zeros(1, 30, 272, 20)
    encoded = model.encode(inputs, torch.zeros(1, 16, 4, 218))
    product, logits = model.single(encoded, torch.tensor([0]), torch.tensor([0]), torch.tensor([1]),
                                   torch.zeros(1, 16, dtype=torch.long), return_gate=True)
    assert not product.any() and torch.all(logits.sigmoid() == .5)


def test_new_model_uses_complete_request_equality_not_first_phase():
    torch.set_num_threads(2)
    bank, requests = sequence_bank(np.zeros(16, dtype=np.int64), np.ones((16, 5, 12), dtype=np.float32))
    assert bank.shape == (16, 5, 288)
    assert requests[0, 1, 0] == requests[0, 0, 0]
    assert not np.array_equal(requests[0, 1], requests[0, 0])
    model = RevisedEffectModel(static(), "A1", windows=48, full_sequence=True).eval()
    encoded = model.encode(torch.zeros(1, 30, 272, 20), torch.tensor(bank[None]))
    base = torch.zeros(1, 16, dtype=torch.long)
    hold = model.single(encoded, torch.tensor([0]), torch.tensor([0]), torch.tensor([1]), base)
    ref = model.single(encoded, torch.tensor([0]), torch.tensor([0]), torch.tensor([0]), base)
    assert hold.shape == (1, 272, 48) and hold.any() and not ref.any()
    forward = model.pair(encoded, torch.tensor([0]), torch.tensor([[0, 15]]), torch.tensor([[1, 2]]), base)
    reverse = model.pair(encoded, torch.tensor([0]), torch.tensor([[15, 0]]), torch.tensor([[2, 1]]), base)
    assert torch.equal(forward, reverse) and forward.shape == (1, 272, 48)


def test_checkpoint_is_regret_then_mae_then_earlier_epoch():
    assert checkpoint_key(1, 100, 100) < checkpoint_key(2, 1, 5)
    assert checkpoint_key(1, 2, 100) < checkpoint_key(1, 3, 5)
    assert checkpoint_key(1, 2, 5) < checkpoint_key(1, 2, 10)


def test_root_selection_unique_and_does_not_invent_missing_receiving_coverage():
    roots = [{"root_id": str(i), "backlog": i, "unavailable_lanes": 1 if i == 17 else 0} for i in range(18)]
    chosen, coverage = select_roots(roots)
    assert coverage == {"receiving_unavailable": 1, "lower_queue": 2, "higher_queue": 2}
    assert len({r["root"]["root_id"] for r in chosen}) == 5


def test_return_to_reference_does_not_reset_actual_phase_or_elapsed():
    from types import SimpleNamespace
    changes = []
    backend = SimpleNamespace(engine=object(), set_phase=lambda node, phase: changes.append(phase), next_step=lambda: None)
    recorder = SimpleNamespace(observe=lambda engine: {"value": np.asarray(0)})
    runner = SignalRunner(backend, SimpleNamespace(intersections=["A"]), recorder)
    runner.initialize()
    first = runner.step([3])
    runner.step([3]); runner.step([3])
    returned = runner.step([3])  # At 90 s the common reference also requests phase index 3.
    assert [int(r["phase_used"][0]) for r in first[:5]] == [0] * 5
    assert int(first[5]["phase_used"][0]) == 4
    assert returned[0]["phase_elapsed_start_s"][0] == 85
    assert changes == [1, 0, 4]


def test_h240_labels_and_within_pair_decision_error():
    raw = {"lane_q": np.ones((1, 240, 240, 3), dtype=np.uint16),
           "intersection_nq": np.ones((1, 240, 16, 2), dtype=np.uint16),
           "boundary_pending": np.ones((1, 240, 16), dtype=np.uint16)}
    y = waiting_windows_240(raw)
    assert y.shape == (1, 272, 48) and y.dtype == np.int64
    assert y[0, 0].sum() == 720
    values = {k: np.array([[v]], dtype=np.int64) for k, v in zip(("base", "a", "b", "ab"), (100, 90, 90, 200))}
    plan = {"branches": [{"branch_id": "base", "changed_intersections": []},
                         {"branch_id": "a", "changed_intersections": [0]},
                         {"branch_id": "b", "changed_intersections": [1]},
                         {"branch_id": "ab", "changed_intersections": [0, 1], "single_ids": ["a", "b"]}]}
    report = panel_decision(values, plan, {"baseline_id": "base", "candidate_ids": ["base", "a", "b", "ab"]})
    assert report["oracle_single"]["regret_vehicle_s"] == 110
    assert report["oracle_pair"]["regret_vehicle_s"] == 0
