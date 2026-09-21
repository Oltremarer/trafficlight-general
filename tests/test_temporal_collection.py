from collections import Counter
import json

from cityflow_tsc.collect_temporal_effects import expand_tasks, emit_readiness, link_directory, prepare_arm


def parent_tasks():
    tasks = []
    for split, count in (("train", 220), ("validation", 26), ("test", 12)):
        for i in range(count):
            tasks.append({"split": split, "task_id": f"{split}_{i}", "root_id": f"old_{split}_{i}",
                          "time_s": 600 if split != "test" else 1800, "policy": "FixedTime",
                          "manifest": {"flow_sha256": f"flow{i}", "roadnet_sha256": "road",
                                       "scenario": {"seed": 1}}})
    return {"tasks": tasks}


def test_expansion_preserves_parent_and_prioritizes_validation():
    parent = parent_tasks()
    before = json.dumps(parent)
    tasks = expand_tasks(parent)
    assert json.dumps(parent) == before
    assert Counter(t["split"] for t in tasks) == {"train": 660, "validation": 78, "test": 12}
    assert [t["root_id"] for t in tasks[:258]] == [t["root_id"] for t in parent["tasks"]]
    added = tasks[258:]
    assert len(added) * 129 == 63468
    assert all(t["split"] == "validation" for t in added[:52])
    assert all(t["time_s"] in (1800, 2700) for t in added)
    assert len({t["task_id"] for t in tasks}) == 750
    assert len({t["root_id"] for t in tasks}) == 750


def test_readiness_requires_all_validation_without_test(tmp_path):
    tasks = expand_tasks(parent_tasks())
    results = {t["root_id"]: {} for t in tasks if t["temporal_reused"] and t["split"] != "test"}
    emit_readiness(tmp_path, tasks, results)
    assert not (tmp_path / "early_ready.json").exists()
    results.update({t["root_id"]: {} for t in tasks if t["split"] == "validation"})
    emit_readiness(tmp_path, tasks, results)
    assert json.loads((tmp_path / "early_ready.json").read_text())["root_count"] == 298
    assert not (tmp_path / "multi_ready.json").exists()
    results.update({t["root_id"]: {} for t in tasks if t["split"] == "train"})
    emit_readiness(tmp_path, tasks, results)
    assert json.loads((tmp_path / "multi_ready.json").read_text())["root_count"] == 738


def test_raw_link_can_target_future_shared_root(tmp_path):
    source, target = tmp_path / "shared/roots/new", tmp_path / "arm/roots/new"
    link_directory(source, target)
    link_directory(source, target)
    assert target.is_symlink()
    assert target.resolve() == source
    assert not source.exists()


def test_arms_have_separate_training_populations_and_no_inherited_statistics(tmp_path):
    tasks = expand_tasks(parent_tasks())
    (tmp_path / "selection.json").write_text(json.dumps({"tasks": tasks}))
    (tmp_path / "protocol.json").write_text("{}")
    (tmp_path / "static").mkdir()
    (tmp_path / "static/geometry.json").write_text("{}")
    for arm, counts in (("early", {"train": 220, "validation": 78, "test": 12}),
                        ("multi", {"train": 660, "validation": 78, "test": 12})):
        run = prepare_arm(tmp_path, arm)
        selection = json.loads((run / "selection.json").read_text())
        assert selection["temporal_arm"] == arm
        assert selection["split_roots"] == counts
        assert not (run / "derived").exists()
        assert not (run / "checkpoints_locked.json").exists()
