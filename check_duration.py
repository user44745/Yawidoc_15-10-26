import soundfile as sf
from pathlib import Path

wav_dir = Path("wav_ascii")
for wav_file in wav_dir.glob("*.wav"):
    audio, sr = sf.read(wav_file)
    duration = len(audio) / sr
    print(f"{wav_file.name}: duration={duration:.2f} seconds, samples={len(audio)}, sr={sr}")