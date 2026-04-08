"""
Example 47: Gemma 4 Audio ASR Fine-Tuning with mlx-tune

Fine-tune Google's Gemma 4 E4B model for speech-to-text / ASR tasks.
Gemma 4 E2B/E4B have a built-in 12-layer Conformer audio tower that
processes 16kHz audio into text — no separate STT model needed.

Supported audio models:
  - gemma-4-E2B-it  (~1GB 4-bit)  — edge/mobile
  - gemma-4-E4B-it  (~2GB 4-bit)  — edge/mobile

NOTE: The 26B and 31B variants do NOT have audio towers.

This example generates synthetic .wav files for a self-contained demo.
Replace with real ASR data (Common Voice, LibriSpeech, etc.) for
production use.

Usage:
    python examples/47_gemma4_audio_asr_finetuning.py
"""

import os
import tempfile
import numpy as np

from mlx_tune import FastVisionModel, UnslothVisionDataCollator, VLMSFTTrainer
from mlx_tune.vlm import VLMSFTConfig


def generate_synthetic_audio(output_dir: str, num_samples: int = 10):
    """Generate synthetic .wav files for demo purposes.

    In production, use real audio files from datasets like:
    - mozilla-foundation/common_voice_17_0
    - openslr/librispeech_asr
    - google/fleurs
    """
    try:
        import soundfile as sf
    except ImportError:
        raise ImportError(
            "soundfile is required for audio examples. "
            "Install with: uv pip install soundfile"
        )

    transcripts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models can understand speech.",
        "Apple Silicon makes local AI training fast.",
        "This is a test of the speech recognition system.",
        "Gemma four can transcribe audio in many languages.",
        "Fine tuning improves model accuracy on specific tasks.",
        "The weather today is sunny with clear skies.",
        "Please turn off the lights in the living room.",
        "How do I get to the nearest train station?",
        "The meeting is scheduled for three o'clock.",
    ]

    samples = []
    for i in range(num_samples):
        transcript = transcripts[i % len(transcripts)]

        # Generate a short synthetic waveform (16kHz, ~2 seconds)
        duration = 2.0
        sr = 16000
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        # Simple tone with some variation per sample
        freq = 200 + (i * 50)
        audio = 0.3 * np.sin(2 * np.pi * freq * t)
        # Add some noise for realism
        audio += 0.05 * np.random.randn(len(audio)).astype(np.float32)

        wav_path = os.path.join(output_dir, f"sample_{i:03d}.wav")
        sf.write(wav_path, audio, sr)

        samples.append({
            "audio_path": wav_path,
            "transcript": transcript,
        })

    return samples


def main():
    print("=" * 70)
    print("GEMMA 4 AUDIO ASR FINE-TUNING")
    print("=" * 70)

    # ========================================================================
    # Step 1: Load Gemma 4 model
    # ========================================================================
    print("\n[Step 1] Loading Gemma 4 E4B model...")

    model, processor = FastVisionModel.from_pretrained(
        "mlx-community/gemma-4-e4b-it-4bit",
        load_in_4bit=True,
    )

    # ========================================================================
    # Step 2: Add LoRA adapters
    # ========================================================================
    print("\n[Step 2] Adding LoRA adapters...")

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,     # No vision training
        finetune_language_layers=True,     # Train language layers for ASR
        finetune_audio_layers=False,       # Audio tower works well frozen
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        random_state=3407,
    )

    # ========================================================================
    # Step 3: Prepare ASR dataset
    # ========================================================================
    print("\n[Step 3] Preparing ASR dataset...")

    audio_dir = tempfile.mkdtemp(prefix="gemma4_audio_")
    samples = generate_synthetic_audio(audio_dir, num_samples=10)

    # Convert to message format with audio content type
    dataset = []
    for s in samples:
        dataset.append({
            "messages": [
                {"role": "user", "content": [
                    {"type": "audio", "audio": s["audio_path"]},
                    {"type": "text", "text": "Transcribe this audio."},
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": s["transcript"]},
                ]},
            ]
        })

    # Repeat for more training steps
    dataset = dataset * 5
    print(f"Dataset: {len(dataset)} samples")
    print(f"Audio files in: {audio_dir}")

    # ========================================================================
    # Step 4: Pre-training inference
    # ========================================================================
    print("\n[Step 4] Pre-training inference test...")

    FastVisionModel.for_inference(model)

    test_audio = samples[0]["audio_path"]
    try:
        response = model.generate(
            audio=test_audio,
            prompt="Transcribe this audio.",
            max_tokens=128,
            temperature=0.0,
        )
        print(f"Audio: {test_audio}")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Pre-training inference error: {e}")

    # ========================================================================
    # Step 5: Train
    # ========================================================================
    print("\n[Step 5] Training...")

    FastVisionModel.for_training(model)

    trainer = VLMSFTTrainer(
        model=model,
        tokenizer=processor,
        data_collator=UnslothVisionDataCollator(model, processor),
        train_dataset=dataset,
        args=VLMSFTConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=30,
            learning_rate=2e-4,
            logging_steps=1,
            optim="adam",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs_gemma4_audio_asr",
            report_to="none",
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=512,
        ),
    )

    trainer_stats = trainer.train()
    print(f"\nTraining metrics: {trainer_stats.metrics}")

    # ========================================================================
    # Step 6: Post-training inference
    # ========================================================================
    print("\n[Step 6] Post-training inference test...")

    FastVisionModel.for_inference(model)

    try:
        response = model.generate(
            audio=test_audio,
            prompt="Transcribe this audio.",
            max_tokens=128,
            temperature=0.0,
        )
        print(f"Audio: {test_audio}")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Post-training inference error: {e}")

    # ========================================================================
    # Step 7: Save
    # ========================================================================
    print("\n[Step 7] Saving LoRA adapters...")

    model.save_pretrained("gemma4_audio_asr_lora")
    print("Saved to gemma4_audio_asr_lora/")

    # Cleanup synthetic audio
    import shutil
    shutil.rmtree(audio_dir, ignore_errors=True)

    print("\n" + "=" * 70)
    print("Done! Gemma 4 audio ASR fine-tuning complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
