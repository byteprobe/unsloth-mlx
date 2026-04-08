"""
Example 48: Gemma 4 Audio Understanding Fine-Tuning

Fine-tune Gemma 4 E4B for audio understanding tasks like:
- Language identification
- Speaker intent classification
- Audio content description
- Emotion detection from speech

This example demonstrates finetune_audio_layers=True for domain-specific
acoustic adaptation, which applies LoRA to the 12-layer Conformer audio
tower in addition to language layers.

Supported audio models:
  - gemma-4-E2B-it  (~1GB 4-bit)
  - gemma-4-E4B-it  (~2GB 4-bit)

Usage:
    python examples/48_gemma4_audio_understanding.py
"""

import os
import tempfile
import numpy as np

from mlx_tune import FastVisionModel, UnslothVisionDataCollator, VLMSFTTrainer
from mlx_tune.vlm import VLMSFTConfig


def generate_synthetic_audio(output_dir: str, num_samples: int = 10):
    """Generate synthetic audio with varied characteristics for QA training."""
    try:
        import soundfile as sf
    except ImportError:
        raise ImportError(
            "soundfile is required for audio examples. "
            "Install with: uv pip install soundfile"
        )

    # Simulated QA pairs about audio content
    qa_pairs = [
        {
            "question": "What language is being spoken in this audio?",
            "answer": "The speaker is speaking English.",
            "freq": 250, "duration": 2.0,
        },
        {
            "question": "How many speakers are in this audio clip?",
            "answer": "There is one speaker in this audio clip.",
            "freq": 300, "duration": 1.5,
        },
        {
            "question": "What is the general topic of the speech?",
            "answer": "The speaker is discussing weather conditions.",
            "freq": 350, "duration": 2.5,
        },
        {
            "question": "Is the speaker asking a question or making a statement?",
            "answer": "The speaker is making a declarative statement.",
            "freq": 200, "duration": 2.0,
        },
        {
            "question": "Describe the audio quality of this recording.",
            "answer": "The recording has clear audio with minimal background noise.",
            "freq": 400, "duration": 1.8,
        },
        {
            "question": "What emotion does the speaker convey?",
            "answer": "The speaker sounds calm and neutral.",
            "freq": 280, "duration": 2.2,
        },
        {
            "question": "Is there any background music in this audio?",
            "answer": "No, there is no background music. Only speech is present.",
            "freq": 320, "duration": 1.6,
        },
        {
            "question": "Summarize what the speaker is saying.",
            "answer": "The speaker is providing instructions for a recipe.",
            "freq": 260, "duration": 3.0,
        },
        {
            "question": "What is the approximate duration of the speech?",
            "answer": "The speech segment is approximately two seconds long.",
            "freq": 340, "duration": 2.0,
        },
        {
            "question": "Does the speaker have an accent?",
            "answer": "The speaker appears to have a standard American English accent.",
            "freq": 290, "duration": 2.4,
        },
    ]

    samples = []
    sr = 16000

    for i in range(num_samples):
        qa = qa_pairs[i % len(qa_pairs)]

        # Generate varied synthetic waveform
        t = np.linspace(0, qa["duration"], int(sr * qa["duration"]), dtype=np.float32)
        audio = 0.3 * np.sin(2 * np.pi * qa["freq"] * t)
        # Add harmonics for more realistic audio
        audio += 0.1 * np.sin(2 * np.pi * qa["freq"] * 2 * t)
        audio += 0.05 * np.random.randn(len(audio)).astype(np.float32)

        wav_path = os.path.join(output_dir, f"audio_qa_{i:03d}.wav")
        sf.write(wav_path, audio, sr)

        samples.append({
            "audio_path": wav_path,
            "question": qa["question"],
            "answer": qa["answer"],
        })

    return samples


def main():
    print("=" * 70)
    print("GEMMA 4 AUDIO UNDERSTANDING FINE-TUNING")
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
    # Step 2: Add LoRA adapters with audio tower fine-tuning
    # ========================================================================
    print("\n[Step 2] Adding LoRA adapters (with audio tower)...")

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,     # No vision training
        finetune_language_layers=True,     # Train language layers
        finetune_audio_layers=True,        # Also adapt audio tower!
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=8,                               # Lower rank for audio tower
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        random_state=3407,
    )

    # ========================================================================
    # Step 3: Prepare audio QA dataset
    # ========================================================================
    print("\n[Step 3] Preparing audio understanding dataset...")

    audio_dir = tempfile.mkdtemp(prefix="gemma4_audio_qa_")
    samples = generate_synthetic_audio(audio_dir, num_samples=10)

    # Convert to message format
    dataset = []
    for s in samples:
        dataset.append({
            "messages": [
                {"role": "user", "content": [
                    {"type": "audio", "audio": s["audio_path"]},
                    {"type": "text", "text": s["question"]},
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": s["answer"]},
                ]},
            ]
        })

    dataset = dataset * 5
    print(f"Dataset: {len(dataset)} samples")

    # ========================================================================
    # Step 4: Pre-training inference
    # ========================================================================
    print("\n[Step 4] Pre-training inference test...")

    FastVisionModel.for_inference(model)

    test_audio = samples[0]["audio_path"]
    test_question = samples[0]["question"]
    try:
        response = model.generate(
            audio=test_audio,
            prompt=test_question,
            max_tokens=128,
            temperature=0.0,
        )
        print(f"Q: {test_question}")
        print(f"A: {response}")
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
            learning_rate=1e-4,
            logging_steps=1,
            optim="adam",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs_gemma4_audio_understanding",
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
            prompt=test_question,
            max_tokens=128,
            temperature=0.0,
        )
        print(f"Q: {test_question}")
        print(f"A: {response}")
    except Exception as e:
        print(f"Post-training inference error: {e}")

    # ========================================================================
    # Step 7: Save
    # ========================================================================
    print("\n[Step 7] Saving LoRA adapters...")

    model.save_pretrained("gemma4_audio_understanding_lora")
    print("Saved to gemma4_audio_understanding_lora/")

    # Cleanup
    import shutil
    shutil.rmtree(audio_dir, ignore_errors=True)

    print("\n" + "=" * 70)
    print("Done! Gemma 4 audio understanding fine-tuning complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
