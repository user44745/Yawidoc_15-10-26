import torch
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration
from datasets import Dataset, DatasetDict
import json
from pathlib import Path
import numpy as np

# Load a couple of samples to test
root = Path(".")
pairs_path = root / "pattani_dataset" / "pairs.jsonl"
rows = []
with pairs_path.open("r", encoding="utf-8") as f:
    for line in f:
        rows.append(json.loads(line.strip()))

audio_map = {
    (1, 1): "person1_tape1.wav",
    (1, 2): "person1_tape2.wav",
    (2, 1): "person2_tape1.wav",
    (3, 1): "person3_tape1.wav",
}

import soundfile as sf
try:
    import librosa
except ImportError:
    librosa = None

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="ms", task="transcribe")

samples = []
for r in rows[:2]:  # just two samples
    person = r["person"]
    tape = r["tape"]
    key = (person, tape)
    if key not in audio_map:
        continue
    audio_filename = audio_map[key]
    audio_path = root / "wav_ascii" / audio_filename
    if not audio_path.exists():
        continue
    audio_array, original_sr = sf.read(audio_path, dtype='float32')
    if audio_array.ndim > 1:
        audio_array = np.mean(audio_array, axis=1)
    target_sr = 16000
    if original_sr != target_sr:
        if librosa is not None:
            audio_array = librosa.resample(audio_array, orig_sr=original_sr, target_sr=target_sr)
        else:
            print(f"Warning: Sample rate {original_sr} != {target_sr} for {audio_path}. Librosa not installed; skipping resampling.")
            target_sr = original_sr
    audio_array = audio_array.astype(np.float32)
    max_samples = int(target_sr * 30.0)
    if len(audio_array) > max_samples:
        audio_array = audio_array[:max_samples]
    inputs = feature_extractor(audio_array, sampling_rate=target_sr)
    input_features = inputs.input_features[0]  # shape (80, seq_len)
    label = r.get("malay_thai")
    if label is None or label.strip() == "":
        label = r.get("thai")
    if label is None or label.strip() == "":
        continue
    samples.append({
        "input_features": input_features,
        "sentence": label.strip(),
    })

print(f"Number of samples: {len(samples)}")
for i, s in enumerate(samples):
    print(f"Sample {i}: input_features shape: {s['input_features'].shape}, sentence: {s['sentence'][:50]}")

# Create dataset
ds = Dataset.from_list(samples)
# Tokenize
def prepare_dataset(batch):
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

ds = ds.map(prepare_dataset, remove_columns=["sentence"])
print(f"Dataset after tokenization: {ds}")

# Data collator
from dataclasses import dataclass
from typing import Any, List, Dict

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: Any

    def __call__(self, features: List[Dict]):
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )
        label_features = [{"input_ids": feature["labels"]} for feature in features]
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

processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="ms", task="transcribe")
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=tokenizer.bos_token_id,
)

# Get tokenized examples from the dataset
tokenized_examples = [ds[i] for i in range(len(ds))]
batch = data_collator(tokenized_examples)
print(f"Batch keys: {batch.keys()}")
print(f"Batch input_features shape: {batch['input_features'].shape}")
print(f"Batch labels shape: {batch['labels'].shape}")

# Move to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch = {k: v.to(device) for k, v in batch.items()}

# Load model
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.generation_config.language = "ms"
model.generation_config.task = "transcribe"
model.config.suppress_tokens = []
model.config.use_cache = False
model.to(device)
print(f"Model device: {next(model.parameters()).device}")

# Forward pass
with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
    outputs = model(**batch)
print(f"Loss: {outputs.loss}")
print(f"Loss shape: {outputs.loss.shape}")

# Backward pass
outputs.loss.backward()
print("Backward successful")