#!/usr/bin/env python3
"""
Speech vs non-speech inference, PANNs only. 

Checks PANNs first for laughter/crying/cough/etc since those mostly get
mistaken for speech by VAD (both are voiced, mouth activity)
Only falls back to VAD's "speech" call if nothing in the human-sound
vocab matches confidently

PANNs runs at 32kHz but Silero VAD wants 16kHz, so the clip gets
resampled down just for the VAD check.

usage:
    python inference_panns.py --record --duration 3
    python inference_panns.py --file some_clip.wav
"""
import argparse
import librosa
from common.vad import has_speech
from common.audio_io import load_audio, record_from_mic
from common.buckets import best_bucket
from models.panns_model import PANNsModel

CONFIDENCE_THRESHOLD = 0.15 #adjust accordingly
VAD_SR = 16000


def classify(model, audio):
    sound_type, sound_score = best_bucket(model.top_predictions(audio))

    if sound_score >= CONFIDENCE_THRESHOLD:
        return {"label": "nonspeech", "type": sound_type, "score": round(sound_score, 3)}

    audio_16k = librosa.resample(audio, orig_sr=model.sample_rate, target_sr=VAD_SR)
    if has_speech(audio_16k, sr=VAD_SR):
        return {"label": "speech"}

    return {"label": "nonspeech", "type": "not_classified", "score": round(sound_score, 3)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--file", type=str)
    args = parser.parse_args()

    print("loading PANNs model, give it a sec...")
    model = PANNsModel()
    print("ready\n")

    if args.record:
        clip = record_from_mic(args.duration, sr=model.sample_rate)
    elif args.file:
        clip, _ = load_audio(args.file, sr=model.sample_rate)
    else:
        print("need either --record or --file <path>")
        raise SystemExit

    result = classify(model, clip)

    print()
    for key, val in result.items():
        print(f"{key}: {val}")


if __name__ == "__main__":
    main()
