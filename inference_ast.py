#!/usr/bin/env python3
"""
Speech vs non-speech inference, AST only. 

Checks AST first for laughter/crying/cough/etc since those mostly get
mistaken for speech by VAD(both are voiced,mouth activity).
Only falls back to VAD's "speech" call if nothing in the human-sound
vocab matches confidently

usage:
    python inference_ast.py --record --duration 3
    python inference_ast.py --file some_clip.wav
"""
import argparse
from common.vad import has_speech
from common.audio_io import load_audio, record_from_mic
from common.buckets import best_bucket
from models.ast_model import ASTModel

CONFIDENCE_THRESHOLD = 0.15


def classify(model, audio, sr):
    sound_type, sound_score = best_bucket(model.top_predictions(audio))

    if sound_score >= CONFIDENCE_THRESHOLD:
        return {"label": "nonspeech", "type": sound_type, "score": round(sound_score, 3)}

    if has_speech(audio, sr=sr):
        return {"label": "speech"}

    return {"label": "nonspeech", "type": "not_classified", "score": round(sound_score, 3)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--file", type=str)
    args = parser.parse_args()

    print("loading AST model")
    model = ASTModel()
    print("ready\n")

    if args.record:
        clip = record_from_mic(args.duration, sr=model.sample_rate)
    elif args.file:
        clip, _ = load_audio(args.file, sr=model.sample_rate)
    else:
        print("need either --record or --file <path>")
        raise SystemExit

    result = classify(model, clip, model.sample_rate)

    print()
    for key, val in result.items():
        print(f"{key}: {val}")


if __name__ == "__main__":
    main()
