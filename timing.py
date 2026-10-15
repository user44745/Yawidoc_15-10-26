import time
import torch
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf
import numpy as np
import json
from pathlib import Path

# Load one sample to get the shape
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

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="ms", task="transcribe")

# Load audio for first sample
r = rows[0]
person = r["person"]
tape = r["tape"]
key = (person, tape)
audio_filename = audio_map[key]
audio_path = root / "wav_ascii" / audio_filename
audio_array, original_sr = sf.read(audio_path, dtype='float32')
if audio_array.ndim > 1:
    audio_array = np.mean(audio_array, axis=1)
target_sr = 16000
if original_sr != target_sr:
    try:
        import librosa
        audio_array = librosa.resample(audio_array, orig_sr=original_sr, target_sr=target_sr)
    except ImportError:
        target_sr = original_sr
audio_array = audio_array.astype(np.float32)
max_samples = int(target_sr * 30.0)
if len(audio_array) > max_samples:
    audio_array = audio_array[:max_samples]
inputs = feature_extractor(audio_array, sampling_rate=target_sr)
input_features = inputs.input_features[0]  # shape (80, seq_len)
print(f"Input features shape: {input_features.shape}")

# Prepare a batch of size 2 by repeating the same sample
batch_size = 2
input_features_batch = np.stack([input_features] * batch_size, axis=0)  # (batch, 80, 3000)
# Convert to tensor
input_features_tensor = torch.from_numpy(input_features_batch)  # on CPU

# Labels: we need to create dummy labels, but for timing we can skip the loss computation?
# Let's create dummy labels of appropriate length.
# We'll tokenize a dummy sentence.
dummy_sentence = "ทดสอบ"
labels = tokenizer(dummy_sentence).input_ids
# Pad to max length? We'll just create a tensor of shape (batch_size, seq_len) with padding token.
# For simplicity, we'll create a tensor filled with pad token ID.
pad_token_id = tokenizer.pad_token_id
labels_batch = torch.full((batch_size, len(labels)), pad_token_id, dtype=torch.long)
# But we need to shift the labels for decoder? Actually, the model expects labels as input_ids for the decoder.
# We'll just use the same dummy labels for both sequences in the batch.
labels_batch = torch.tensor([labels] * batch_size, dtype=torch.long)

# Move to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_features_tensor = input_features_tensor.to(device)
labels_batch = labels_batch.to(device)

# Load model
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.generation_config.language = "ms"
model.generation_config.task = "transcribe"
model.config.suppress_tokens = []
model.config.use_cache = False
model.to(device)
model.train()  # set to train mode

print(f"Model device: {next(model.parameters()).device}")

# Time the forward pass
torch.cuda.synchronize() if torch.cuda.is_available() else None
start = time.time()
outputs = model(input_features=input_features_tensor, labels=labels_batch)
torch.cuda.synchronize() if torch.cuda.is_available() else None
end = time.time()
print(f"Forward pass time: {end - start:.3f} seconds")
print(f"Loss: {outputs.loss.item()}")

# Time the backward pass
torch.cuda.synchronize() if torch.cuda.is_available() else None
start = time.time()
outputs.loss.backward()
torch.cuda.synchronize() if torch.cuda.is_available() else None
end = time.time()
print(f"Backward pass time: {end - start:.3f} seconds")