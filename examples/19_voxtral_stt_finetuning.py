"""
Example 19: Voxtral STT Fine-Tuning with MLX-Tune

Fine-tune Mistral's Voxtral model for speech-to-text on Apple Silicon.

Voxtral is an "audio-LLM" architecture (similar to Qwen3-ASR):
1. Audio -> mel spectrogram -> audio encoder -> features
2. Features merged into Llama LM token sequence via projector
3. Llama LM generates transcription autoregressively

LoRA targets the Llama decoder and optionally the audio encoder.

Requirements:
    uv pip install 'mlx-tune[audio]'
    brew install ffmpeg

Usage:
    python examples/19_voxtral_stt_finetuning.py
"""

from mlx_tune import FastSTTModel, STTSFTTrainer, STTSFTConfig, STTDataCollator


def main():
    # =========================================================================
    # 1. Load Voxtral Model
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 1: Loading Voxtral Model")
    print("=" * 70)

    model, processor = FastSTTModel.from_pretrained(
        # Voxtral Mini 3B — Mistral's speech model with Llama decoder
        model_name="mlx-community/Voxtral-Mini-3B-2507-bf16",
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
        # Llama-style targets for decoder
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        finetune_encoder=True,   # LoRA on audio encoder
        finetune_decoder=True,   # LoRA on Llama decoder
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
        "Voxtral is Mistral's speech recognition model.",
        "It combines a Conformer audio encoder with a Llama decoder.",
        "The model supports multiple languages for transcription.",
        "Fine-tuning with LoRA is efficient on Apple Silicon.",
        "MLX-Tune makes speech model training accessible on Mac.",
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
        output_dir="./voxtral_outputs",
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
    model.save_pretrained("./voxtral_outputs/adapters")
    print("\nDone! Adapters saved to ./voxtral_outputs/adapters")


if __name__ == "__main__":
    main()
