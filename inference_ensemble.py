#!/usr/bin/env python3
"""
Speech vs non-speech inference, AST + PANNs ensemble. Inference only

If both models agree on the label, then trust. If only one is confident,
go with that one and flag it. If they disagree, returns as "uncertain"
instead of guessing.

AST and Silero VAD both run at 16kHz, PANNs runs at 32kHz, so the clip
gets loaded at both rates.

usage:
    python inference_ensemble.py --record --duration 3
    python inference_ensemble.py --file some_clip.wav
"""
import argparse
import librosa
from common.vad import has_speech
from common.audio_io import load_audio, record_from_mic
from common.buckets import best_bucket
from models.ast_model import ASTModel
from models.panns_model import PANNsModel

#adjust if needed
CONFIDENCE_THRESHOLD = 0.15
VAD_SR = 16000
AST_SR = 16000
PANNS_SR = 32000


def classify(ast_model, panns_model, audio_16k, audio_32k):
    ast_type, ast_score = best_bucket(ast_model.top_predictions(audio_16k))
    panns_type, panns_score = best_bucket(panns_model.top_predictions(audio_32k))

    ast_confident = ast_score >= CONFIDENCE_THRESHOLD
    panns_confident = panns_score >= CONFIDENCE_THRESHOLD

    ast_guess = f"{ast_type} ({ast_score:.3f})"
    panns_guess = f"{panns_type} ({panns_score:.3f})"

    if ast_confident and panns_confident and ast_type == panns_type:
        return {"label": "nonspeech", "type": ast_type, "agreement": "both models agree",
                "ast_guess": ast_guess, "panns_guess": panns_guess}

    if ast_confident and panns_confident and ast_type != panns_type:
        return {"label": "uncertain", "reason": "models disagree",
                "ast_guess": ast_guess, "panns_guess": panns_guess}

    if ast_confident and not panns_confident:
        return {"label": "nonspeech", "type": ast_type, "agreement": "only AST confident",
                "ast_guess": ast_guess, "panns_guess": panns_guess}

    if panns_confident and not ast_confident:
        return {"label": "nonspeech", "type": panns_type, "agreement": "only PANNs confident",
                "ast_guess": ast_guess, "panns_guess": panns_guess}

    if has_speech(audio_16k, sr=VAD_SR):
        return {"label": "speech"}

    return {"label": "nonspeech", "type": "not_classified",
            "ast_guess": ast_guess, "panns_guess": panns_guess}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--file", type=str)
    args = parser.parse_args()

    print("loading AST + PANNs models")
    ast_model = ASTModel()
    panns_model = PANNsModel()
    print("ready\n")

    if args.record:
        audio_32k = record_from_mic(args.duration, sr=PANNS_SR)
        audio_16k = librosa.resample(audio_32k, orig_sr=PANNS_SR, target_sr=VAD_SR)
    elif args.file:
        audio_16k, _ = load_audio(args.file, sr=VAD_SR)
        audio_32k, _ = load_audio(args.file, sr=PANNS_SR)
    else:
        print("need either --record or --file <path>")
        raise SystemExit

    result = classify(ast_model, panns_model, audio_16k, audio_32k)

    print()
    for key, val in result.items():
        print(f"{key}: {val}")


if __name__ == "__main__":
    main()
