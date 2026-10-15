#!/usr/bin/env python
"""
Tiny training test to verify speed.
"""
import os
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)


def load_tiny_dataset(data_dir: str, max_samples: int = 2):
    """Load a tiny subset of the Pattani interview dataset."""
    root = Path(data_dir)
    pairs_path = root / "pattani_dataset" / "pairs.jsonl"
    if not pairs_path.exists():
        raise FileNotFoundError(f"Could not find {pairs_path}. Run parse_transcripts.py first.")

    # Load pairs
    rows = []
    with pairs_path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line.strip()))

    # Map person/tape to audio file (wav) using simple ASCII names
    audio_map = {
        (1, 1): "person1_tape1.wav",
        (1, 2): "person1_tape2.wav",
        (2, 1): "person2_tape1.wav",
        (3, 1): "person3_tape1.wav",
    }

    # Initialize feature extractor for computing input features
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")

    samples = []
    for r in rows:
        if len(samples) >= max_samples:
            break
        person = r["person"]
        tape = r["tape"]
        key = (person, tape)
        if key not in audio_map:
            continue
        audio_filename = audio_map[key]
        audio_path = root / "wav_ascii" / audio_filename
        if not audio_path.exists():
            continue
        # Load audio
        try:
            audio_array, original_sr = sf.read(audio_path, dtype='float32')
            # Ensure mono
            if audio_array.ndim > 1:
                audio_array = np.mean(audio_array, axis=1)
            # Resample to 16kHz if needed
            target_sr = 16000
            if original_sr != target_sr:
                try:
                    import librosa
                    audio_array = librosa.resample(audio_array, orig_sr=original_sr, target_sr=target_sr)
                except ImportError:
                    # If librosa not installed, skip resampling and hope it's close enough
                    print(f"Warning: Sample rate {original_sr} != {target_sr} for {audio_path}. Librosa not installed; skipping resampling.")
                    target_sr = original_sr
            # Ensure float32
            audio_array = audio_array.astype(np.float32)
            # Truncate if too long
            max_samples_audio = int(target_sr * 30.0)
            if len(audio_array) > max_samples_audio:
                audio_array = audio_array[:max_samples_audio]
        except Exception as e:
            print(f"Failed to load audio {audio_path}: {e}")
            continue
        # Compute input features
        try:
            inputs = feature_extractor(audio_array, sampling_rate=target_sr)
            input_features = inputs.input_features[0]  # shape (80, seq_len)
        except Exception as e:
            print(f"Failed to compute features for {audio_path}: {e}")
            continue
        # Use malay_thai as label (Malay dialect in Thai script)
        label = r.get("malay_thai")
        if label is None or label.strip() == "":
            # Fallback to thai if malay_thai missing?
            label = r.get("thai")
        if label is None or label.strip() == "":
            continue
        samples.append({
            "input_features": input_features,
            "sentence": label.strip(),
        })

    if not samples:
        raise ValueError("No valid samples found. Check audio file naming and transcript content.")

    # Create dataset
    ds = Dataset.from_list(samples)
    # No need to cast; input_features is already a numpy array, sentence is string

    # Split into train and test (but for tiny dataset, we'll just use all for train and a copy for test)
    ds = ds.train_test_split(test_size=0.5, seed=42)  # 50% for test
    return DatasetDict({"train": ds["train"], "test": ds["test"]})


def main():
    import argparse
    import dataclasses
    from typing import Any
    parser = argparse.ArgumentParser(description="Tiny training test.")
    parser.add_argument("--model-name", default="openai/whisper-small", help="HF model to fine-tune")
    parser.add_argument("--output-dir", default="./tiny_checkpoint", help="Where to save the model")
    parser.add_argument("--num-train-epochs", type=float, default=1.0, help="Number of training epochs")
    parser.add_argument("--per-device-batch-size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=0, help="Warmup steps")
    parser.add_argument("--eval-steps", type=int, default=1, help="Evaluation steps")
    parser.add_argument("--save-steps", type=int, default=1, help="Save steps")
    parser.add_argument("--save-total-limit", type=int, default=1, help="Limit total checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--language", default="ms", help="Language code for transcripts (ms for Malay)")
    parser.add_argument("--task", default="transcribe", choices=["transcribe", "translate"], help="Task")
    parser.add_argument("--max-audio-seconds", type=float, default=30.0, help="Drop clips longer than this")
    args = parser.parse_args()

    # Normalize language
    language = args.language.strip().lower()
    aliases = {"msa": "ms", "malay": "ms", "tha": "th", "thai": "th"}
    language = aliases.get(language, language)

    print(f"Loading tiny dataset from interview data...")
    dataset = load_tiny_dataset(data_dir=".", max_samples=4)  # get up to 4 samples to have a split
    print(f"Dataset loaded: {dataset}")

    # Load processor, tokenizer, feature extractor, model
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_name)
    tokenizer = WhisperTokenizer.from_pretrained(
        args.model_name, language=language, task=args.task
    )
    processor = WhisperProcessor.from_pretrained(
        args.model_name, language=language, task=args.task
    )
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)

    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.generation_config.language = language
    model.generation_config.task = args.task
    model.generation_config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False  # required for gradient checkpointing
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model moved to device: {device}")

    # Since we already have input_features, we can skip the prepare_dataset step.
    # However, the trainer expects the dataset to have "input_features" and "labels".
    # We have "input_features" and "sentence". We need to map "sentence" to "labels".
    def prepare_dataset(batch):
        batch["labels"] = tokenizer(batch["sentence"]).input_ids
        return batch

    dataset = dataset.map(
        prepare_dataset,
        remove_columns=["sentence"],
        desc="Tokenizing transcripts",
    )

    @dataclasses.dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any
        decoder_start_token_id: Any

        def __call__(self, features: list[dict]):
            input_features = [
                {"input_features": feature["input_features"]} for feature in features
            ]
            batch = self.processor.feature_extractor.pad(
                input_features, return_tensors="pt"
            )
            label_features = [
                {"input_ids": feature["labels"]} for feature in features
            ]
            labels_batch = self.processor.tokenizer.pad(
                label_features, return_tensors="pt"
            )
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100
            )
            if (labels[:, 0] == self.decoder_start_token_id).all():
                labels = labels[:, 1:]
            batch["labels"] = labels
            return batch

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    def compute_metrics(pred):
        import jiwer
        pred_ids = np.argmax(pred.predictions, axis=-1)
        pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(
            pred.label_ids, skip_special_tokens=True
        )
        return {
            "wer": 100 * jiwer.wer(label_str, pred_str),
            "cer": 100 * jiwer.cer(label_str, pred_str),
        }

    use_fp16 = torch.cuda.is_available()
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        gradient_checkpointing=False,
        fp16=False,  # Disable fp16 for debugging
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        logging_steps=1,
        predict_with_generate=True,
        report_to=[],
        dataloader_num_workers=0,
        seed=args.seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"✅ Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()