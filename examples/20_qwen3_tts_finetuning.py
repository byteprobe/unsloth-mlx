"""
Example 20: Qwen3-TTS Fine-Tuning with MLX-Tune

Fine-tune Alibaba's Qwen3-TTS model on Apple Silicon using LoRA.

Qwen3-TTS is a multilingual TTS model (ZH, EN, JA, KO, +more) that converts
text to speech by:
1. Embedding text through a separate text projection path (2048 -> 1024 dim)
2. Predicting discrete audio codes (code_0) via a 28-layer talker transformer
3. Generating remaining 15 codebooks via a 5-layer code predictor
4. Decoding all 16 codebooks to a 24kHz waveform via a built-in speech tokenizer

Key differences from other TTS models:
- Dual embedding path: text and codec tokens use separate embeddings, added together
- 16-codebook Split RVQ speech tokenizer at 12.5Hz (built-in, not external codec)
- Talker predicts code_0 only; code predictor fills code_1-15 during inference
- Forward pass takes inputs_embeds (pre-computed), not raw input_ids

The API is identical -- same FastTTSModel, TTSSFTTrainer, TTSDataCollator.

Requirements:
    uv pip install 'mlx-tune[audio]'
    # Also needs: datasets, soundfile

Usage:
    python examples/20_qwen3_tts_finetuning.py
"""

from mlx_tune import FastTTSModel, TTSSFTTrainer, TTSSFTConfig, TTSDataCollator


def main():
    # =========================================================================
    # 1. Load Model + Built-in Speech Tokenizer
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 1: Loading Qwen3-TTS Model")
    print("=" * 70)

    model, tokenizer = FastTTSModel.from_pretrained(
        # Qwen3-TTS 1.7B VoiceDesign (bf16) from mlx-community
        model_name="mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
        max_seq_length=2048,
        # Speech tokenizer (codec) is built-in — no external codec needed
    )

    # =========================================================================
    # 2. Add LoRA Adapters
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 2: Adding LoRA Adapters")
    print("=" * 70)

    model = FastTTSModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        # Target all attention + MLP layers in the 28-layer talker
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    # =========================================================================
    # 3. Load & Prepare Dataset
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 3: Loading Dataset")
    print("=" * 70)

    from datasets import load_dataset, Audio

    # Example: Elise voice dataset
    # Each sample has 'audio' (waveform) and 'text' (transcript)
    dataset = load_dataset("MrDragonFox/Elise", split="train[:10]")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=24000))

    print(f"Dataset: {len(dataset)} samples")
    print(f"Columns: {dataset.column_names}")
    print(f"Sample text: {dataset[0]['text'][:80]}...")

    # =========================================================================
    # 4. Create Data Collator
    # =========================================================================
    collator = TTSDataCollator(
        model=model,
        tokenizer=tokenizer,
        text_column="text",
        audio_column="audio",
    )

    # =========================================================================
    # 5. Train
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 4: Fine-Tuning")
    print("=" * 70)

    # Convert to list for trainer
    train_data = [dataset[i] for i in range(len(dataset))]

    trainer = TTSSFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=collator,
        train_dataset=train_data,
        args=TTSSFTConfig(
            output_dir="./qwen3_tts_output",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            max_steps=60,
            warmup_steps=5,
            logging_steps=1,
            weight_decay=0.01,
            train_on_completions=True,
        ),
    )

    result = trainer.train()
    print(f"\nFinal loss: {result.metrics['train_loss']:.4f}")

    # =========================================================================
    # 6. Save Adapters
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 5: Saving Adapters")
    print("=" * 70)

    model.save_pretrained("./qwen3_tts_output/final_adapter")
    print("Done! Adapters saved to ./qwen3_tts_output/final_adapter")


if __name__ == "__main__":
    main()
