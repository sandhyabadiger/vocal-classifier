#!/usr/bin/env python3
"""
Zero-shot speech vs non-speech batch evaluation, PANNs only.

Runs Silero VAD for speech detection+PANNs (CNN14) for
non-speech labeling, against a labeled manifest (CommonVoice + VIVAE
chunks). checks the predictions against the known ground-truth label.

PANNs runs at 32kHz but Silero VAD wants 16kHz, so each clip gets loaded
twice at the two different rates.

usage:
    python eval_panns.py --manifest manifests/combined_manifest.csv --sample_n 300
    python eval_panns.py --manifest manifests/combined_manifest.csv --sample_n 300 --balanced
"""
import argparse
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

from common.vad import has_speech
from common.audio_io import load_audio
from common.buckets import best_bucket
from common.sampling import stratified_sample
from models.panns_model import PANNsModel

CONFIDENCE_THRESHOLD = 0.15
VAD_SR = 16000


def predict_binary_label(model, filepath):
    audio_panns, _ = load_audio(filepath, sr=model.sample_rate)
    fine_type, fine_score = best_bucket(model.top_predictions(audio_panns))

    if fine_score >= CONFIDENCE_THRESHOLD:
        return "nonspeech", fine_type
    audio_vad, _ = load_audio(filepath, sr=VAD_SR)
    if has_speech(audio_vad, sr=VAD_SR):
        return "speech", None
    return "nonspeech", "not_classified"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--sample_n", type=int, default=None,
                     help="Evaluate on a subset instead of everything")
    ap.add_argument("--balanced", action="store_true",
                     help="With --sample_n, split evenly across labels and sources instead of plain random sampling")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("Loading PANNs model...")
    model = PANNsModel()
    print("Model loaded")

    df = pd.read_csv(args.manifest)
    if args.sample_n:
        if args.balanced:
            df = stratified_sample(df, args.sample_n, seed=args.seed)
        else:
            df = df.sample(n=min(args.sample_n, len(df)), random_state=args.seed)

    print(f"Evaluating on {len(df)} clips")
    print(f"  by label: {df['label'].value_counts().to_dict()}")
    print(f"  by source: {df['source'].value_counts().to_dict()}")

    y_true, y_pred, fine_grained_preds, sources, filenames = [], [], [], [], []

    for i, row in df.iterrows():
        try:
            pred_label, fine = predict_binary_label(model, row["filepath"])
        except Exception as e:
            print(f"  [error] {row['filename']}: {e}")
            continue

        y_true.append(row["label"])
        y_pred.append(pred_label)
        fine_grained_preds.append(fine)
        sources.append(row["source"])
        filenames.append(row["filename"])

        if len(y_true) % 50 == 0:
            print(f"  {len(y_true)}/{len(df)}")

    print("\nZERO-SHOT VAD+PANNS RESULTS")
    print(classification_report(y_true, y_pred))
    print("Confusion matrix (rows=true, cols=pred), labels=[nonspeech, speech]:")
    print(confusion_matrix(y_true, y_pred, labels=["nonspeech", "speech"]))

    result_df = pd.DataFrame({
        "filename": filenames, "true": y_true, "pred": y_pred,
        "fine_grained": fine_grained_preds, "source": sources
    })
    print("\nAccuracy by source:")
    for src in result_df["source"].unique():
        sub = result_df[result_df["source"] == src]
        acc = (sub["true"] == sub["pred"]).mean()
        print(f"  {src}: {acc:.3f} ({len(sub)} clips)")

    result_df.to_csv("eval_panns_results.csv", index=False)
    print("\nSaved: eval_panns_results.csv")


if __name__ == "__main__":
    main()