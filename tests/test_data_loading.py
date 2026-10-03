import json

import pytest
from datasets import Dataset

from exotic_trainer.data_loading import (
    _group_aware_split,
    _normalize_dataset,
    load_training_data,
)


def test_group_split_keeps_trajectory_turns_together() -> None:
    raw = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "task a"},
                    {"role": "assistant", "content": "step one"},
                    {"role": "user", "content": "continue"},
                    {"role": "assistant", "content": "step two"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "task b"},
                    {"role": "assistant", "content": "done"},
                ]
            },
        ]
    )
    normalized = _normalize_dataset(raw)

    training, validation = _group_aware_split(normalized, eval_ratio=0.5, seed=42)

    assert set(training["group_id"]).isdisjoint(set(validation["group_id"]))
    assert len(training) + len(validation) == 3


def test_external_validation_uses_distinct_instructions(tmp_path) -> None:
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    train_path.write_text(
        "\n".join(
            json.dumps({"instruction": f"train {index}", "output": f"answer {index}"})
            for index in range(5)
        ),
        encoding="utf-8",
    )
    validation_path.write_text(
        "\n".join(
            json.dumps({"instruction": f"novel {index}", "output": f"result {index}"})
            for index in range(3)
        ),
        encoding="utf-8",
    )

    training, validation = load_training_data(
        [str(train_path)],
        eval_ratio=0.2,
        seed=42,
        max_source_rows=100,
        max_training_samples=100,
        validation_sources=[str(validation_path)],
        max_validation_samples=20,
    )

    assert len(training) == 5
    assert len(validation) == 3
    assert validation[0]["prompt"][0]["content"].startswith("novel")


def test_external_validation_blocks_duplicate_content(tmp_path) -> None:
    path = tmp_path / "same.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"instruction": f"task {index}", "output": f"answer {index}"})
            for index in range(3)
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="identical source records"):
        load_training_data(
            [str(path)],
            eval_ratio=0.2,
            seed=42,
            max_source_rows=100,
            max_training_samples=100,
            validation_sources=[str(path)],
            max_validation_samples=20,
        )


def test_tool_menu_conditioning_applies_once_to_training_and_validation(tmp_path) -> None:
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    train_path.write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": "Policy\nList of tools: [already-conditioned]",
                    },
                    {"role": "user", "content": "Read the file"},
                    {"role": "assistant", "content": "Done"},
                ]
            }
        ),
        encoding="utf-8",
    )
    validation_path.write_text(
        json.dumps({"instruction": "Explain reading", "output": "Reading observes data."}),
        encoding="utf-8",
    )

    training, validation = load_training_data(
        [str(train_path)],
        eval_ratio=0.2,
        seed=42,
        max_source_rows=100,
        max_training_samples=100,
        allowed_tools=["read"],
        validation_sources=[str(validation_path)],
        max_validation_samples=20,
        tool_menu_conditioning=True,
    )

    training_system = training[0]["prompt"][0]["content"]
    validation_system = validation[0]["prompt"][0]["content"]
    assert training_system.count("List of tools: [") == 1
    assert validation_system.count("List of tools: [") == 1
    assert '"name": "read"' in validation_system
