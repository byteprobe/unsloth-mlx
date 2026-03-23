"""
Example 13: Whisper STT Fine-Tuning with MLX-Tune

Fine-tune OpenAI's Whisper model for speech-to-text on Apple Silicon.

Whisper is an encoder-decoder model:
1. Audio -> mel spectrogram -> encoder -> audio features
2. Audio features -> decoder (with teacher forcing) -> transcript

LoRA can target the encoder, decoder, or both.

Requirements:
    uv pip install 'mlx-tune[audio]'
    # Also needs: datasets

Usage:
    python examples/13_whisper_stt_finetuning.py
"""

from mlx_tune import FastSTTModel, STTSFTTrainer, STTSFTConfig, STTDataCollator


def main():
    # =========================================================================
    # 1. Load Whisper Model
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 1: Loading Whisper Model")
    print("=" * 70)

    model, processor = FastSTTModel.from_pretrained(
        # Use a small Whisper model for demo (use whisper-large-v3-turbo for quality)
        model_name="mlx-community/whisper-tiny-asr-fp16",
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
        # Target attention projections in both encoder and decoder
        target_modules=["query", "key", "value", "out"],
        finetune_encoder=True,   # LoRA on encoder attention
        finetune_decoder=True,   # LoRA on decoder attention + cross-attention
    )

    # =========================================================================
    # 3. Load & Prepare Dataset
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 3: Loading Dataset")
    print("=" * 70)

    from datasets import load_dataset, Audio

    # Example: Common Voice dataset (English)
    # Each sample has 'audio' (waveform) and 'sentence' (transcript)
    dataset = load_dataset(
        "mozilla-foundation/common_voice_16_0",
        "en",
        split="train[:10]",
        trust_remote_code=True,
    )
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    print(f"Dataset: {len(dataset)} samples")
    print(f"Columns: {dataset.column_names}")
    print(f"Sample text: {dataset[0].get('sentence', 'N/A')[:80]}...")

    # =========================================================================
    # 4. Create Data Collator
    # =========================================================================
    collator = STTDataCollator(
        model=model,
        processor=processor,
        language="en",
        task="transcribe",
        audio_column="audio",
        text_column="sentence",  # Common Voice uses 'sentence'
    )

    # =========================================================================
    # 5. Train
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 4: Fine-Tuning")
    print("=" * 70)

    # Convert to list for trainer
    train_data = [dataset[i] for i in range(len(dataset))]

    trainer = STTSFTTrainer(
        model=model,
        processor=processor,
        data_collator=collator,
        train_dataset=train_data,
        args=STTSFTConfig(
            output_dir="./stt_output",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=1e-5,  # Lower LR for Whisper
            max_steps=100,
            warmup_steps=5,
            logging_steps=1,
            weight_decay=0.01,
            language="en",
            sample_rate=16000,
        ),
    )

    result = trainer.train()
    print(f"\nFinal loss: {result.metrics['train_loss']:.4f}")

    # =========================================================================
    # 6. Transcribe Audio
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 5: Transcribing Audio")
    print("=" * 70)

    FastSTTModel.for_inference(model)

    # Transcribe a sample from the dataset
    sample_audio = dataset[0]["audio"]["array"]
    text = model.transcribe(sample_audio, language="en")
    print(f"Transcription: {text}")
    print(f"Reference:     {dataset[0].get('sentence', 'N/A')}")

    # =========================================================================
    # 7. Save Adapters
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 6: Saving Adapters")
    print("=" * 70)

    model.save_pretrained("./stt_output/final_adapter")
    print("Done! Adapters saved to ./stt_output/final_adapter")


if __name__ == "__main__":
    main()
