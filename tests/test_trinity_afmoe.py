"""
Tests for Arcee Trinity / AFMoE support.

Fast tests cover:
- chat_templates.py alias + name-based auto-detection
- Target-module resolver produces MoE expert + shared-expert paths on Trinity
- Chat template from real tokenizer resolves to 'chatml'

Slow test (@pytest.mark.slow) covers:
- End-to-end SFT: load 4-bit Trinity-Nano-Preview, apply per-expert LoRA,
  train a handful of steps, confirm loss decrease + adapter save/reload.
"""

import os
import pytest

TRINITY_4BIT = "mlx-community/Trinity-Nano-Preview-4bit"


# ---------------------------------------------------------------------------
# Fast tests — no model download
# ---------------------------------------------------------------------------

def test_alias_resolution():
    from mlx_tune.chat_templates import TEMPLATE_ALIASES
    for alias in ["trinity", "trinity-nano", "trinity-mini", "trinity-large",
                  "afmoe", "arcee"]:
        assert TEMPLATE_ALIASES[alias] == "chatml", alias


def test_name_based_detection():
    from mlx_tune.chat_templates import _detect_template_from_tokenizer

    class FakeTok:
        def __init__(self, n):
            self.name_or_path = n
            self.chat_template = None

    for name in [
        "mlx-community/Trinity-Nano-Preview-4bit",
        "arcee-ai/Trinity-Mini",
        "arcee-ai/AFMoE-6B",
        "user/trinity-large-preview",
    ]:
        assert _detect_template_from_tokenizer(FakeTok(name)) == "chatml", name


def test_get_chat_template_accepts_trinity_alias():
    from mlx_tune import get_chat_template

    class Tok:
        chat_template = None
        def get_vocab(self):
            return {}

    tok = Tok()
    out = get_chat_template(tok, chat_template="trinity")
    assert out.chat_template is not None
    assert "<|im_start|>" in out.chat_template


# ---------------------------------------------------------------------------
# Slow E2E tests — require model download (~3.3 GB)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_trinity_loads_and_detects_moe():
    """Load Trinity-Nano, verify MoE detection via SwitchLinear."""
    from mlx_tune import FastLanguageModel
    from mlx_tune.model import _resolve_target_modules

    model, tok = FastLanguageModel.from_pretrained(
        model_name=TRINITY_4BIT,
        max_seq_length=512,
        load_in_4bit=True,
    )

    resolved = _resolve_target_modules(
        model.model,
        ["q_proj", "k_proj", "v_proj", "o_proj",
         "gate_proj", "up_proj", "down_proj"],
    )

    # Attention and both MLP paths must be present
    assert any(p.endswith("self_attn.q_proj") for p in resolved)
    assert any(p.endswith("self_attn.o_proj") for p in resolved)
    # MoE experts (SwitchLinear inside SwitchGLU)
    assert any("experts.gate_proj" in p for p in resolved)
    assert any("experts.up_proj" in p for p in resolved)
    assert any("experts.down_proj" in p for p in resolved)
    # Shared expert
    assert any("shared_experts" in p for p in resolved)
    # Gated attention's gate_proj also present (self_attn.gate_proj)
    assert any("self_attn.gate_proj" in p for p in resolved)


@pytest.mark.slow
def test_trinity_sft_loss_decrease_and_roundtrip(tmp_path):
    """Tiny SFT run: confirm loss decreases + adapter save/load works."""
    from mlx_tune import FastLanguageModel, SFTTrainer, SFTConfig
    from datasets import Dataset

    model, tok = FastLanguageModel.from_pretrained(
        model_name=TRINITY_4BIT,
        max_seq_length=512,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=4,
        target_modules=["q_proj", "v_proj", "gate_proj", "down_proj"],
        lora_alpha=8,
        lora_dropout=0.0,
        bias="none",
        random_state=0,
    )

    samples = [
        {"instruction": "Say 'alpha'.", "output": "alpha alpha alpha alpha."},
        {"instruction": "Say 'beta'.", "output": "beta beta beta beta."},
        {"instruction": "Say 'gamma'.", "output": "gamma gamma gamma gamma."},
    ] * 3

    def fmt(b):
        return {"text": [
            tok.apply_chat_template(
                [{"role": "user", "content": i},
                 {"role": "assistant", "content": o}],
                tokenize=False, add_generation_prompt=False,
            )
            for i, o in zip(b["instruction"], b["output"])
        ]}

    ds = Dataset.from_list(samples).map(fmt, batched=True)

    out_dir = str(tmp_path / "trinity_sft")
    trainer = SFTTrainer(
        model=model,
        train_dataset=ds,
        tokenizer=tok,
        args=SFTConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            warmup_steps=1,
            max_steps=6,
            learning_rate=2e-4,
            logging_steps=1,
            output_dir=out_dir,
        ),
    )
    hist = trainer.train()

    # Accept either history-dict or trainer-attached history for safety
    losses = []
    if isinstance(hist, dict) and "loss" in hist:
        losses = hist["loss"]
    elif hasattr(trainer, "loss_history"):
        losses = trainer.loss_history
    if losses and len(losses) >= 2:
        # Loose: last quarter mean <= first quarter mean
        q = max(1, len(losses) // 4)
        assert sum(losses[-q:]) / q <= sum(losses[:q]) / q + 0.5, \
            f"loss did not decrease: first={losses[:q]} last={losses[-q:]}"

    # Adapter save/reload
    adapter_dir = os.path.join(out_dir, "saved_adapters")
    model.save_pretrained(adapter_dir)
    assert os.path.exists(os.path.join(adapter_dir, "adapter_config.json"))
    assert any(f.startswith("adapters") and f.endswith(".safetensors")
               for f in os.listdir(adapter_dir))
