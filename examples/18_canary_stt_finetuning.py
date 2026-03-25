"""
Example 18: NVIDIA Canary STT Fine-Tuning with MLX-Tune

Fine-tune NVIDIA's Canary model for multilingual speech-to-text on Apple Silicon.

Canary is an encoder-decoder model:
1. Audio -> mel spectrogram -> FastConformer encoder -> audio features
2. Audio features -> Transformer decoder (with cross-attention) -> transcript

Supports 25+ languages and speech translation.

Requirements:
    uv pip install 'mlx-tune[audio]'
    brew install ffmpeg

Usage:
    python examples/18_canary_stt_finetuning.py
"""

from mlx_tune import FastSTTModel, STTSFTTrainer, STTSFTConfig, STTDataCollator


def main():
    # =========================================================================
    # 1. Load Canary Model
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 1: Loading NVIDIA Canary Model")
    print("=" * 70)

    model, processor = FastSTTModel.from_pretrained(
        # Canary 1B v2 — NVIDIA's multilingual ASR with translation
        model_name="eelcor/canary-1b-v2-mlx",
        max_seq_length=448,
    )

    # =========================================================================
    # 2. Add LoRA Adapters
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 2: Adding LoRA Adapters")
    print("=" * 70)

    model = FastSTTModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        # Canary uses q/k/v/out_proj in both encoder and decoder
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        finetune_encoder=True,   # LoRA on FastConformer encoder
        finetune_decoder=True,   # LoRA on Transformer decoder
    )

    # =========================================================================
    # 3. Prepare Synthetic Dataset
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 3: Preparing Dataset")
    print("=" * 70)

    import numpy as np
    np.random.seed(42)

    dataset = []
    texts = [
        "The Canary model supports multilingual transcription.",
        "NVIDIA developed this model for accurate speech recognition.",
        "It uses a FastConformer encoder architecture.",
        "Translation from one language to another is also supported.",
        "Fine-tuning with LoRA adapters is memory efficient.",
    ]
    for text in texts:
        audio = np.random.randn(32000).astype(np.float32) * 0.1
        dataset.append({
            "audio": {"array": audio, "sampling_rate": 16000},
            "text": text,
        })

    print(f"  Dataset: {len(dataset)} samples")

    # =========================================================================
    # 4. Create Data Collator & Config
    # =========================================================================
    data_collator = STTDataCollator(
        model=model,
        processor=processor,
        language="en",
        task="transcribe",
    )

    config = STTSFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        warmup_steps=2,
        max_steps=10,
        learning_rate=1e-4,
        logging_steps=1,
        output_dir="./canary_outputs",
    )

    # =========================================================================
    # 5. Train
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 5: Training")
    print("=" * 70)

    trainer = STTSFTTrainer(
        model=model,
        processor=processor,
        data_collator=data_collator,
        train_dataset=dataset,
        args=config,
    )

    stats = trainer.train()
    print(f"\nFinal loss: {stats.metrics['train_loss']:.4f}")

    # =========================================================================
    # 6. Save Adapters
    # =========================================================================
    model.save_pretrained("./canary_outputs/adapters")
    print("\nDone! Adapters saved to ./canary_outputs/adapters")


if __name__ == "__main__":
    main()
