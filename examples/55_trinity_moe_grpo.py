"""
Example 55: Arcee Trinity-Nano (AFMoE) — GRPO Reasoning on GSM8K

Teach Trinity-Nano to produce <reasoning>...</reasoning><answer>...</answer>
reasoning traces via Group Relative Policy Optimization (GRPO).

GRPO works by generating N completions per prompt, scoring each with a reward
function, normalizing by the group mean/std (→ advantage), and applying a
policy-gradient update. This only produces signal when the N completions have
*different* rewards. If every completion gets reward 0 (e.g. because the base
model never emits the required format), advantage collapses to 0 and the
optimizer no-ops.

That is exactly what happens if you point GRPO at the Trinity-Nano-Preview
checkpoint cold — the preview isn't tuned on this output format. Real-world
GRPO recipes (DeepSeek-R1, Tulu 3, etc.) always run an SFT warmup on
chain-of-thought demonstrations FIRST, then GRPO.

This example does both phases:

    Phase 1 (SFT warmup)  → teach the <reasoning>/<answer> format from
                            GSM8K's own rationales (the "#### N" marker is
                            dropped and the preceding text becomes the
                            reasoning; the number becomes the answer).

    Phase 2 (GRPO)        → sharpen correctness with a *softened* reward
                            function that gives partial credit for just
                            surfacing the right number, so reward variance
                            appears even when the XML format isn't yet
                            perfect.

-----------------------------------------------------------------------
PRODUCTION RECIPE (scale the knobs below when you move past this demo)
-----------------------------------------------------------------------
These demo values are tiny so the example runs in ~10 minutes on a Mac.
For a real reasoning fine-tune, use roughly:

    Phase 1 (SFT-on-CoT warmup)
        dataset:             full GSM8K train (7.5k) + optionally MATH
                             + a few k of open-math-instruct style CoT
        max_steps:           300-800  (1-2 epochs on ~8k rows)
        per_device_bsz:      1 (MLX) with grad_accum=8-16
        learning_rate:       2e-4
        r / lora_alpha:      16 / 32
        expected outcome:    model reliably emits <reasoning>/<answer>
                             on unseen prompts. Sanity-check this BEFORE
                             you start Phase 2 — GRPO amplifies whatever
                             format the model already produces.

    Phase 2 (GRPO)
        dataset:             ≥2k fresh prompts (don't reuse warmup rows)
        num_generations:     8-16  (bigger group = more reliable variance)
        temperature:         0.7-1.0
        max_completion_length: 256-512
        learning_rate:       1e-6 to 5e-6   (GRPO is more sensitive than SFT)
        beta (KL):           0.02-0.08
        max_steps:           300-1000
        logging_steps:       10
        reward function:     start with this example's layered reward;
                             tighten tiers only once ≥60% of completions
                             already produce correct XML.

Hardware note: Trinity-Nano is 1B active params but the sampling phase
of GRPO dominates wall-clock. A 32-64 GB Mac can comfortably run the
production recipe overnight. For bigger models (Trinity-Mini 26B MoE),
drop num_generations to 4 and max_completion_length to 256.

Dataset: openai/gsm8k (main, train split).
Model:   mlx-community/Trinity-Nano-Preview-4bit (~3.3 GB).
"""

import re
from mlx_tune import (
    FastLanguageModel,
    SFTTrainer, SFTConfig,
    GRPOTrainer, GRPOConfig,
)
from datasets import load_dataset, Dataset


MODEL_ID = "mlx-community/Trinity-Nano-Preview-4bit"
DATASET_ID = "openai/gsm8k"

WARMUP_ROWS = 40       # Phase-1 SFT demonstrations
GRPO_ROWS = 40         # Phase-2 GRPO prompts

SYSTEM_PROMPT = """You are a careful reasoner. Solve the problem step by step.
Respond EXACTLY in this format and nothing else:
<reasoning>
...your step-by-step work...
</reasoning>
<answer>
...final number only...
</answer>"""


# --------------------------------------------------------------------------
# GSM8K parsing helpers
# --------------------------------------------------------------------------

_GT_RE = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")

def _split_gsm8k_answer(full_answer: str):
    """GSM8K stores '<rationale>\n#### <N>'. Return (rationale, numeric_gt)."""
    m = _GT_RE.search(full_answer)
    if not m:
        return full_answer.strip(), full_answer.strip().splitlines()[-1]
    gt = m.group(1).strip()
    rationale = full_answer[: m.start()].rstrip(" \n")
    # drop GSM8K's inline arithmetic annotations like <<48+24=72>>
    rationale = re.sub(r"<<.*?>>", "", rationale).strip()
    return rationale, gt


def _build_cot_chat(question: str, rationale: str, gt: str, tokenizer):
    """Format a GSM8K row as a full <reasoning>/<answer> SFT demonstration."""
    target = (
        f"<reasoning>\n{rationale}\n</reasoning>\n"
        f"<answer>\n{gt}\n</answer>"
    )
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
        {"role": "assistant", "content": target},
    ]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=False
    )


# --------------------------------------------------------------------------
# SOFTENED reward function (critical for base/preview models)
#
# Why softened? A reward that only fires for strict XML + correct answer
# produces zero variance when the model can't yet format its output — every
# generation gets reward 0, so GRPO's advantage collapses and no update
# happens. Partial credit creates a gradient even when the model is still
# bad at both format and arithmetic.
# --------------------------------------------------------------------------

def _extract_answer(response: str) -> str | None:
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", response, re.DOTALL)
    return m.group(1).strip() if m else None


def _numbers_match(extracted: str, gt: str) -> bool:
    nums_e = re.findall(r"-?\d+\.?\d*", extracted)
    nums_g = re.findall(r"-?\d+\.?\d*", gt)
    if not (nums_e and nums_g):
        return False
    try:
        return float(nums_e[-1]) == float(nums_g[-1])
    except ValueError:
        return False


def soft_reasoning_reward(response: str, ground_truth: str) -> float:
    """
    Layered reward that produces variance even on a model that's bad at the
    required XML format.

        1.00  strict XML <answer>...</answer> contains the correct number
        0.70  <answer>...</answer> present AND the correct number appears
              anywhere in the response (format partially correct)
        0.50  correct number appears anywhere in the response (no XML)
        0.25  response at least emits ONE of <reasoning>/<answer> tags
        0.00  nothing
    """
    answer_body = _extract_answer(response)
    if answer_body is not None and _numbers_match(answer_body, ground_truth):
        return 1.0

    gt_match_anywhere = _numbers_match(response, ground_truth)

    has_ans_tag = "<answer>" in response and "</answer>" in response
    has_rea_tag = "<reasoning>" in response and "</reasoning>" in response

    if has_ans_tag and gt_match_anywhere:
        return 0.7
    if gt_match_anywhere:
        return 0.5
    if has_ans_tag or has_rea_tag:
        return 0.25
    return 0.0


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Trinity-Nano GRPO — AFMoE reasoning on GSM8K (SFT warmup + GRPO)")
    print("=" * 70)

    # 1. Load model + LoRA
    model, tokenizer = FastLanguageModel.from_pretrained(
        MODEL_ID,
        max_seq_length=1024,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
    )

    # 2. Load GSM8K once, build both splits
    print(f"\n[2] Loading {DATASET_ID} (main/train)...")
    raw = load_dataset(DATASET_ID, "main",
                       split=f"train[:{WARMUP_ROWS + GRPO_ROWS}]")
    warmup_rows = raw.select(range(WARMUP_ROWS))
    grpo_rows = raw.select(range(WARMUP_ROWS, WARMUP_ROWS + GRPO_ROWS))
    print(f"  Phase-1 SFT rows: {len(warmup_rows)}")
    print(f"  Phase-2 GRPO rows: {len(grpo_rows)}")

    # ------------------------------------------------------------------
    # PHASE 1 — SFT warmup on formatted chain-of-thought
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Phase 1 / 2 — SFT warmup: teach the <reasoning>/<answer> format")
    print("=" * 70)

    warmup_texts = []
    for row in warmup_rows:
        rationale, gt = _split_gsm8k_answer(row["answer"])
        warmup_texts.append(
            _build_cot_chat(row["question"], rationale, gt, tokenizer)
        )
    warmup_ds = Dataset.from_dict({"text": warmup_texts})

    sft_trainer = SFTTrainer(
        model=model,
        train_dataset=warmup_ds,
        tokenizer=tokenizer,
        args=SFTConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            warmup_steps=2,
            max_steps=20,
            learning_rate=2e-4,
            logging_steps=2,
            output_dir="outputs_trinity_grpo/warmup",
        ),
    )
    sft_trainer.train()

    # ------------------------------------------------------------------
    # PHASE 2 — GRPO with softened reward
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Phase 2 / 2 — GRPO with softened reward")
    print("=" * 70)

    reasoning_data = []
    for row in grpo_rows:
        _, gt = _split_gsm8k_answer(row["answer"])
        chat_text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row["question"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        reasoning_data.append({"prompt": chat_text, "answer": gt})

    grpo_config = GRPOConfig(
        loss_type="grpo",
        beta=0.04,
        num_generations=4,          # bigger group → more chance of variance
        temperature=1.0,
        max_completion_length=192,
        learning_rate=5e-6,
        max_steps=8,
        logging_steps=1,
        output_dir="./outputs_trinity_grpo",
    )

    grpo_trainer = GRPOTrainer(
        model=model,
        train_dataset=reasoning_data,
        tokenizer=tokenizer,
        reward_fn=soft_reasoning_reward,
        args=grpo_config,
    )
    print("\nReward: layered — 1.0 strict, 0.7 loose-format+correct,"
          " 0.5 correct-only, 0.25 any-tag, 0.0 otherwise")
    result = grpo_trainer.train()
    print(f"\nGRPO result: {result.get('status')}")

    # ------------------------------------------------------------------
    # 3. Save adapter + reload + merge (full lifecycle)
    # ------------------------------------------------------------------
    print("\n[3] Saving adapter + reload + merged-model export...")
    adapter_dir = "outputs_trinity_grpo/saved_adapters"
    model.save_pretrained(adapter_dir)

    fresh, _ = FastLanguageModel.from_pretrained(
        MODEL_ID, max_seq_length=1024, load_in_4bit=True,
    )
    fresh.load_adapter(adapter_dir)
    print(f"✓ Reloaded adapter from {adapter_dir} onto a fresh base model")

    merged_dir = "outputs_trinity_grpo/merged_model"
    try:
        model.save_pretrained_merged(merged_dir, tokenizer=tokenizer)
        print(f"✓ Merged model saved to {merged_dir}")
    except Exception as e:
        print(f"(merged-model export skipped: {type(e).__name__}: {e})")

    print("\n" + "=" * 70)
    print("Trinity-Nano GRPO complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
