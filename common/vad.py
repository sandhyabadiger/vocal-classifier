"""Silero VAD wrapper, shared across eval/live/ensemble scripts."""
import torch
from silero_vad import load_silero_vad, get_speech_timestamps

_vad_model = None


def get_vad_model():
    global _vad_model
    if _vad_model is None:
        _vad_model = load_silero_vad()
    return _vad_model


def has_speech(audio, sr=16000):
    vad = get_vad_model()
    timestamps = get_speech_timestamps(torch.tensor(audio), vad, sampling_rate=sr)
    return len(timestamps) > 0
