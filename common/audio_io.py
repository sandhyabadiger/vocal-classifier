"""Audio loading + mic recording helpers, shared across scripts."""
import numpy as np
import librosa


def load_audio(path, sr=16000):
    wav, sr = librosa.load(path, sr=sr, mono=True)
    return np.asarray(wav, dtype=np.float32), sr


def record_from_mic(seconds, sr=16000):
    import sounddevice as sd
    print(f"recording {seconds}s - go ahead and make a sound")
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    print("got it")
    return audio.flatten()
