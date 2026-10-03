from exotic_trainer.model_probe import choose_lora_targets, inspect_model

MODEL_PATH = "/mnt/git0/models/microLLM/LFM2.5-1.2B-Instruct"


def test_inspects_lfm_without_loading_weights() -> None:
    probe = inspect_model(MODEL_PATH)
    assert probe.architecture == "Lfm2ForCausalLM"
    assert 1_000_000_000 < probe.parameter_count < 1_500_000_000
    assert probe.dtype == "bf16"
    assert probe.layer_count == 16


def test_auto_targets_do_not_select_convolutional_out_proj() -> None:
    probe = inspect_model(MODEL_PATH)
    targets = choose_lora_targets(probe, "auto")
    assert targets
    assert all(".conv." not in name for name in targets)
    assert any(name.endswith("q_proj") for name in targets)
    assert any(name.endswith("w1") for name in targets)

