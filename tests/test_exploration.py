from exotic_trainer.exploration import generate_trial_recipes
from exotic_trainer.schema import ExplorationConfig, TrainingRecipe


def test_random_exploration_is_reproducible_and_unique() -> None:
    base = TrainingRecipe(model="model", datasets=["dataset"], workbook_sheet_id="sheet-1")
    config = ExplorationConfig(
        trial_count=4,
        seed=7,
        adapter_methods=["lora", "rslora"],
        lora_ranks=[4, 8],
        target_profiles=["attention", "auto"],
        sequence_lengths=[512, 1024],
    )

    first = generate_trial_recipes(base, config)
    second = generate_trial_recipes(base, config)

    assert [item.exploration_signature for item in first] == [
        item.exploration_signature for item in second
    ]
    assert len({item.exploration_signature for item in first}) == 4
    assert all(item.workbook_sheet_id == "sheet-1" for item in first)
    assert {item.use_rslora for item in first}.issubset({True, False})
