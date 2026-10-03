"""A real one-step CPU LoRA run on a tiny random model and synthetic data."""

import json
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from exotic_trainer.schema import TrainingRecipe
from exotic_trainer.trainer import run_training


def test_one_cpu_step_saves_adapter(tmp_path, monkeypatch):
    root = tmp_path
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("EXOTIC_TRAINER_ROOT", str(root))
    monkeypatch.setenv("EXOTIC_TRAINER_REGISTRY", str(root / "data/registry.sqlite3"))
    monkeypatch.setenv("EXOTIC_TRAINER_RUNS_DIR", str(root / "runs"))
    torch.manual_seed(42)
    model_dir = root / "tiny-random-model"
    vocab = {
        word: i
        for i, word in enumerate(
            [
                "[UNK]",
                "[PAD]",
                "[EOS]",
                "user",
                "assistant",
                ":",
                "Reply",
                "ready",
                "for",
                "case",
                ".",
            ]
            + [str(i) for i in range(120)]
        )
    }
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]"
    )
    tokenizer.chat_template = "{% for message in messages %}{{ message['role'] + ': ' + message['content'] + ' ' + eos_token + ' ' }}{% endfor %}"
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=len(vocab),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=256,
            pad_token_id=1,
            eos_token_id=2,
        )
    )
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    data = root / "synthetic.jsonl"
    data.write_text(
        "".join(
            json.dumps({"instruction": f"Reply ready for case {i}.", "output": "ready"}) + "\n"
            for i in range(120)
        )
    )
    recipe = TrainingRecipe(
        model=str(model_dir),
        datasets=[str(data)],
        name="cpu-publication-smoke",
        device="cpu",
        dtype="fp32",
        sequence_length=128,
        micro_batch_size=1,
        gradient_accumulation=1,
        max_steps=1,
        budget_mode="steps",
        time_limit_minutes=5,
        reserve_minutes=1,
        agent_eval_samples=0,
        max_source_rows=120,
        max_training_samples=120,
        max_validation_samples=8,
        lora_rank=2,
        lora_alpha=4,
        target_profile="attention",
        gradient_checkpointing=False,
    )
    result = run_training(recipe)
    assert result["status"] == "complete", result["status"]
    assert result["global_step"] == 1, result["global_step"]
    assert result["fixed_step_target_reached"] is True
    files = list(Path(result["output_dir"]).rglob("adapter_model.safetensors"))
    assert files, "adapter was not saved"
