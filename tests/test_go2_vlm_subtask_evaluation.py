from scripts.evaluate_go2_vlm_subtasks import (
    parse_generated_subtask,
    summarize_records,
)


def test_parse_generated_subtask() -> None:
    assert (
        parse_generated_subtask("<|nav|><|subtask|>nav_turn<|end_subtask|>")
        == "nav_turn"
    )
    assert parse_generated_subtask("<|nav|>") is None


def test_summary_reports_micro_macro_and_action_subset() -> None:
    records = [
        {
            "target_route": "nav",
            "target_subtask": "nav_turn",
            "predicted_subtask": "nav_turn",
            "route_correct": True,
            "subtask_correct": True,
            "joint_correct": True,
        },
        {
            "target_route": "nav",
            "target_subtask": "nav_turn",
            "predicted_subtask": "nav_straight",
            "route_correct": True,
            "subtask_correct": False,
            "joint_correct": False,
        },
        {
            "target_route": "done",
            "target_subtask": "task_completed",
            "predicted_subtask": "task_completed",
            "route_correct": True,
            "subtask_correct": True,
            "joint_correct": True,
        },
    ]
    summary = summarize_records(records)
    assert summary["route_accuracy"] == 1.0
    assert summary["subtask_exact_accuracy"] == 2 / 3
    assert summary["macro_subtask_exact_accuracy"] == 0.75
    assert summary["action_route_samples"] == 2
    assert summary["action_subtask_exact_accuracy"] == 0.5
