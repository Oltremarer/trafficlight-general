import pytest

from cityflow_tsc.collect_single_revision import demand_root_time
from cityflow_tsc.effect_model.formal_data import _selected_tasks, SINGLE_REVISION_SCHEMA


def test_root_time_depends_only_on_departures():
    assert demand_root_time([{"startTime": 1246}, {"startTime": 1250}]) == (1860, 1246)
    assert demand_root_time([{"startTime": 0}]) == (600, 0)
    with pytest.raises(ValueError, match="horizon"):
        demand_root_time([{"startTime": 3000}])


def test_declared_variable_splits_preserve_group_boundary():
    tasks = [{"root_id": f"{split}-{i}", "source_id": "source", "cohort_id": split, "split": split}
             for split, count in {"train": 4, "validation": 2, "test": 2}.items() for i in range(count)]
    selection = {"schema": SINGLE_REVISION_SCHEMA, "tasks": tasks,
                 "split_roots": {"train": 4, "validation": 2, "test": 2}}
    assert len(_selected_tasks(selection, ("train", "validation"))) == 6
    selection["split_roots"]["train"] = 3
    with pytest.raises(ValueError, match="exact declared"):
        _selected_tasks(selection, ("train",))
    selection["split_roots"]["train"] = 4
    tasks[-1]["cohort_id"] = "train"
    with pytest.raises(ValueError, match="cohorts"):
        _selected_tasks(selection, ("train",))


def test_old_schema_does_not_accept_variable_counts():
    with pytest.raises(ValueError, match="36/12/12"):
        _selected_tasks({"schema": "old", "tasks": []}, ("train",))
