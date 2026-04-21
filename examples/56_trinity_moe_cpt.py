"""
Example 56: Arcee Trinity-Nano (AFMoE) — Continual Pretraining on WikiText

Continual pretraining adapts a model to a new domain or knowledge base by
training on raw text (no chat template, loss on all tokens, with embedding +
lm_head LoRA so vocabulary distributions can shift).

Dataset: Salesforce/wikitext (wikitext-2-raw-v1) — the canonical small CPT
corpus, ~2M tokens of cleaned Wikipedia text. Swap to a domain corpus
(`allenai/scientific_papers`, `bigbio/pubmed_qa`, a legal-text dump, or
your own raw `.txt` dump loaded via `datasets.load_dataset("text", ...)`)
to adapt to that domain instead.

Notes:
  - CPTTrainer auto-adds `embed_tokens` and `lm_head` to the LoRA target
    list, and applies a lower LR to the embedding parameters to avoid
    catastrophic forgetting of the general-purpose representations.
  - On 4-bit Trinity, `embed_tokens` and `lm_head` are quantized; the
    v0.4.25 `_is_quantized` fix means CPT correctly skips their direct
    gradient updates (including the LoRA-wrapped `lm_head`). Training
    still proceeds through the LoRA-wrapped attention + expert paths,
    which is the standard quantized-CPT recipe.
  - Per-expert LoRA still applies to all 128 experts via LoRASwitchLinear.

Model: mlx-community/Trinity-Nano-Preview-4bit (~3.3 GB)
"""

from mlx_tune import FastLanguageModel, CPTTrainer, CPTConfig
from datasets import load_dataset


MODEL_ID = "mlx-community/Trinity-Nano-Preview-4bit"
DATASET_ID = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
NUM_DOCS = 120          # raw paragraph-ish chunks to train on (demo slice)
MIN_DOC_LEN = 200       # chars; skip headings / empties


def main():
    print("=" * 70)
    print("Trinity-Nano CPT — AFMoE domain adaptation on WikiText")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    print("\n[1] Loading Trinity-Nano-Preview (4-bit)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=2048,
        load_in_4bit=True,
    )

    # ------------------------------------------------------------------
    # 2. LoRA (CPTTrainer extends with embed_tokens + lm_head)
    # ------------------------------------------------------------------
    print("\n[2] Applying LoRA (MoE + CPT will extend with embeddings)...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
    )

    # ------------------------------------------------------------------
    # 3. Real HF raw-text corpus
    # ------------------------------------------------------------------
    print(f"\n[3] Loading {DATASET_ID}/{DATASET_CONFIG}...")
    raw = load_dataset(DATASET_ID, DATASET_CONFIG, split="train")
    # Filter headings / empty lines and take a demo slice
    docs = [
        {"text": t}
        for t in raw["text"]
        if t.strip() and len(t.strip()) >= MIN_DOC_LEN
    ][:NUM_DOCS]
    from datasets import Dataset
    dataset = Dataset.from_list(docs)
    print(f"Prepared {len(dataset)} raw-text docs (>= {MIN_DOC_LEN} chars each)")

    # ------------------------------------------------------------------
    # 4. CPT config
    # ------------------------------------------------------------------
    print("\n[4] Configuring CPT...")
    config = CPTConfig(
        learning_rate=5e-5,
        embedding_learning_rate=5e-6,   # 10x smaller — protects embeddings
        include_embeddings=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=3,
        max_steps=20,
        logging_steps=1,
        output_dir="outputs_trinity_cpt",
        max_seq_length=2048,
        lr_scheduler_type="cosine",
    )

    # ------------------------------------------------------------------
    # 5. Train
    # ------------------------------------------------------------------
    print("\n[5] Running continual pretraining...")
    trainer = CPTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=config,
    )
    trainer.train()

    # ------------------------------------------------------------------
    # 6. Generation sanity-check
    # ------------------------------------------------------------------
    print("\n[6] Generation sample...")
    FastLanguageModel.for_inference(model)
    from mlx_lm import generate

    prompt = "Valkyria Chronicles III is a"
    response = generate(
        model.model, tokenizer, prompt=prompt,
        max_tokens=80, verbose=False,
    )
    print(f"\nPrompt: {prompt}")
    print(f"Continuation: {response}")

    # ------------------------------------------------------------------
    # 7. Save adapter + reload roundtrip
    # ------------------------------------------------------------------
    print("\n[7] Saving adapter + reload roundtrip...")
    adapter_dir = "outputs_trinity_cpt/saved_adapters"
    model.save_pretrained(adapter_dir)

    fresh, _ = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID, max_seq_length=2048, load_in_4bit=True,
    )
    fresh.load_adapter(adapter_dir)
    print(f"✓ Reloaded adapter from {adapter_dir} onto a fresh base model")

    print("\n" + "=" * 70)
    print("Trinity-Nano CPT complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
