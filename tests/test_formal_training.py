import itertools

import numpy as np
import torch

from cityflow_tsc.effect_model.formal_model import (
    PAIRS, Ranker, create_model, decision_metrics, group_mean, multiscale_loss,
    predict_root, ranking_features, tie_argmin,
)
from cityflow_tsc.effect_model.revision import sequence_bank


def static():
    geometry = []
    for i, j in PAIRS:
        dx, dy = abs(i % 4 - j % 4), abs(i // 4 - j // 4)
        geometry.append((dx, dy, np.hypot(dx, dy), (dx + dy) / 8, float(dx + dy == 1)))
    return {"adjacency": np.zeros((8, 272, 272), np.float32),
            "relative": np.zeros((16, 272, 11), np.float32),
            "permission": np.ones((16, 5, 272), np.float32),
            "rank_geometry": np.asarray(geometry, np.float32)}


def scales():
    return {"s5": torch.ones(272, 1), "sp": torch.ones(272, 4), "sj": torch.ones(4)}


def test_multiscale_exact_natural_means_and_signed_cumulative():
    prediction = torch.zeros(1, 272, 48, requires_grad=True)
    target = torch.zeros_like(prediction)
    target[0, 0, 0] = 2
    logits = torch.zeros_like(prediction, requires_grad=True)
    l5, lp, lj = 2 / (272 * 48), 1.5 / 272, 1.5
    gate = .01 * np.log(2)
    assert torch.isclose(multiscale_loss(prediction, target, scales(), "A_ref", logits),
                         torch.tensor(l5 + .1 * lj + gate, dtype=torch.float32))
    loss = multiscale_loss(prediction, target, scales(), "A_MS", logits)
    assert torch.isclose(loss, torch.tensor(.1 * l5 + lp + lj + gate, dtype=torch.float32))
    assert torch.isclose(multiscale_loss(prediction, target, scales(), "B1"), loss - gate)
    loss.backward()
    assert torch.isfinite(prediction.grad).all() and torch.isfinite(logits.grad).all()
    target[0, 1, 0] = -2  # Network cancellation must not cancel location supervision.
    b = multiscale_loss(prediction, target, scales(), "B1")
    assert torch.isclose(b, torch.tensor(.1 * 2 * l5 + 2 * lp, dtype=torch.float32))


def test_loss_linear_de_normalization_and_horizon_boundaries():
    s = scales()
    s["s5"] *= 3
    prediction = torch.ones(1, 272, 48)
    target = torch.ones_like(prediction) * 3
    assert multiscale_loss(prediction, target, s, "B1") == 0
    # Effects cancel in the same location before the first cumulative horizon.
    target.zero_()
    prediction.zero_()
    target[0, 0, 0], target[0, 0, 1] = 2, -2
    assert torch.isclose(multiscale_loss(prediction, target, scales(), "B1"),
                         torch.tensor(.1 * 4 / (272 * 48)))


def test_pair_stage_freezes_gate_dropout_and_preserves_single_parameters():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = create_model(static())
    before = {k: v.clone() for k, v in model.state_dict().items() if not k.startswith("pair_head.")}
    model.start_pair_stage()
    model.train(True)
    assert model.pair_head.training and not model.single_gate.training
    assert not any(m.training for m in model.single_gate.modules())
    assert all(p.requires_grad == name.startswith("pair_head.") for name, p in model.named_parameters())
    assert not model.pair_head[-1].weight.any() and not model.pair_head[-1].bias.any()
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    model.eval()
    assert not model.pair_head.training


def test_a_versions_identical_seed_state_and_request_semantics():
    torch.manual_seed(42)
    first = create_model(static())
    torch.manual_seed(42)
    second = create_model(static())
    assert all(torch.equal(value, second.state_dict()[key]) for key, value in first.state_dict().items())
    bank, _ = sequence_bank(np.zeros(16, dtype=np.int64), np.ones((16, 5, 12), np.float32))
    first.eval()
    with torch.no_grad():
        first.single_head[-1].bias.fill_(2)
        encoded = first.encode(torch.zeros(1, 30, 272, 20), torch.tensor(bank[None]))
        root = torch.tensor([0])
        base = torch.zeros(1, 16, dtype=torch.long)
        held = first.single(encoded, root, root, torch.tensor([1]), base)
        reference = first.single(encoded, root, root, root, base)
    assert torch.all(held == 1) and not reference.any()


def test_ranker_parameter_count_symmetric_and_nonnegative():
    model = create_model(static())
    encoded = {"objects": torch.randn(2, 272, 64), "global": torch.randn(2, 64)}
    forward = ranking_features(model, encoded)
    reverse = ranking_features(model, encoded, PAIRS[:, ::-1].copy())
    assert forward.shape == (2, 120, 197) and torch.equal(forward, reverse)
    ranker = Ranker()
    assert sum(p.numel() for p in ranker.parameters()) == 12737
    assert torch.all(ranker(forward) > 0)


def test_fp64_tie_reference_first_and_group_equal_not_query_equal():
    assert tie_argmin([0, -5e-7, 1]) == 0
    assert tie_argmin(torch.tensor([0, -2e-6, 1], dtype=torch.float64)) == 1
    metrics = decision_metrics([0, -5e-7, 1], [0, -2, 1])
    assert metrics["selected"] == 0 and metrics["regret"] == 2
    assert not metrics["worse_than_reference"]
    records = [{"cohort_id": "a", "flow_id": "a1", "value": 0},
               {"cohort_id": "a", "flow_id": "a1", "value": 2},
               {"cohort_id": "a", "flow_id": "a2", "value": 5},
               {"cohort_id": "b", "flow_id": "b1", "value": 9}]
    assert group_mean(records, "value") == 6  # ((1+5)/2 + 9)/2


def test_sparse_inference_only_calls_selected_pairs_and_no_truth_input():
    class CountingModel:
        def __init__(self):
            self.rank_geometry = torch.tensor(static()["rank_geometry"])
            self.calls = []

        def eval(self):
            return self

        def encode(self, history, bank):
            return {"objects": torch.zeros(1, 272, 64), "global": torch.zeros(1, 64)}

        def single(self, encoded, roots, nodes, actions, base):
            return torch.ones(len(nodes), 272, 48)

        def pair(self, encoded, roots, nodes, actions, base):
            self.calls.append(nodes.cpu().numpy())
            return torch.ones(len(nodes), 272, 48) * 2

    pairs = np.repeat(PAIRS, 16, axis=0)
    root = {"normalized_history": np.zeros((30, 272, 20), np.float32),
            "action_bank": np.zeros((16, 5, 288), np.float32), "base_phase": np.zeros(16, np.int64),
            "single_nodes": np.repeat(np.arange(16), 4)[:, None],
            "single_actions": np.tile(np.arange(1, 5), 16)[:, None],
            "s_incidence": np.zeros((65, 64), np.float32),
            "p_incidence": np.zeros((65, 1920), np.float32),
            "pair_nodes": pairs, "pair_actions": np.tile(list(itertools.product(range(1, 5), repeat=2)), (120, 1)),
            "pair_unique_nodes": PAIRS, "pair_query_pair_index": np.repeat(np.arange(120), 16)}
    root["s_incidence"][1, 0] = 1
    root["p_incidence"][1, 0] = 1
    stats = {kind + "_" + key: value.numpy() for kind in ("single", "pair") for key, value in scales().items()}
    ranker = Ranker()
    for parameter in ranker.parameters():
        parameter.data.zero_()  # All tied: first eight lexicographic pairs.
    model = CountingModel()
    result = predict_root(model, root, stats, torch.device("cpu"), "Top-8", ranker, return_fields=False)
    assert sum(len(c) for c in model.calls) == 128 and max(map(len, model.calls)) <= 128
    np.testing.assert_array_equal(result["selected_pairs"], PAIRS[:8])
    assert result["joint"] is None and result["selected"] == 0
    assert result["scores"].dtype == torch.float64
    assert result["scores"][1] == 3 * 272 * 48
    assert not result["selected_field"].any()
    model.calls.clear()
    result = predict_root(model, root, stats, torch.device("cpu"), "Adj-24")
    assert sum(len(c) for c in model.calls) == 384 and len(result["selected_pairs"]) == 24
    torch.testing.assert_close(result["scores"], result["joint"].sum((1, 2)))
