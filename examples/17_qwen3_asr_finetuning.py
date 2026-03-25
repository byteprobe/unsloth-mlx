"""
Example 17: Qwen3-ASR Fine-Tuning with MLX-Tune

Fine-tune Alibaba's Qwen3-ASR model for speech-to-text on Apple Silicon.

Qwen3-ASR is an "audio-LLM" architecture:
1. Audio -> mel spectrogram -> audio encoder -> features
2. Features are injected into the Qwen3 LLM token sequence
3. Qwen3 LLM generates transcription autoregressively

LoRA targets the Qwen3 decoder (q/k/v/o_proj) and optionally the audio encoder.

Requirements:
    uv pip install 'mlx-tune[audio]'
    brew install ffmpeg

Usage:
    python examples/17_qwen3_asr_finetuning.py
"""

from mlx_tune import FastSTTModel, STTSFTTrainer, STTSFTConfig, STTDataCollator


def main():
    # =========================================================================
    # 1. Load Qwen3-ASR Model
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 1: Loading Qwen3-ASR Model")
    print("=" * 70)

    model, processor = FastSTTModel.from_pretrained(
        # Qwen3-ASR 1.7B 8-bit — multilingual ASR with 30+ languages
        model_name="mlx-community/Qwen3-ASR-1.7B-8bit",
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
        # Qwen3-ASR decoder targets (same as standard Qwen3)
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        finetune_encoder=True,   # LoRA on audio encoder
        finetune_decoder=True,   # LoRA on Qwen3 text decoder
    )

    # =========================================================================
    # 3. Prepare Synthetic Dataset (replace with your real data)
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 3: Preparing Dataset")
    print("=" * 70)

    import numpy as np
    np.random.seed(42)

    # Synthetic audio samples (replace with real audio from datasets)
    dataset = []
    texts = [
        "Hello, how are you today?",
        "The weather is nice outside.",
        "Machine learning on Apple Silicon is fast.",
        "Qwen3-ASR supports over thirty languages.",
        "Fine-tuning speech models with LoRA is efficient.",
    ]
    for text in texts:
        # 2 seconds of synthetic audio at 16kHz
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
        output_dir="./qwen3_asr_outputs",
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
    model.save_pretrained("./qwen3_asr_outputs/adapters")
    print("\nDone! Adapters saved to ./qwen3_asr_outputs/adapters")


if __name__ == "__main__":
    main()
