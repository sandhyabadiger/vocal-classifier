"""Balanced sampling for eval scripts, so a small --sample_n still gets a
fair mix of speech/nonspeech and both source datasets instead of whatever
a plain random draw happens to grab."""
import pandas as pd


def stratified_sample(df, n, seed=42):
    labels = sorted(df["label"].unique())
    per_label = n // len(labels)

    parts = []
    for label in labels:
        label_df = df[df["label"] == label]
        sources = sorted(label_df["source"].unique())
        per_source = per_label // len(sources)

        for source in sources:
            pool = label_df[label_df["source"] == source]
            take = min(per_source, len(pool))
            parts.append(pool.sample(n=take, random_state=seed))

    sampled = pd.concat(parts)
    return sampled.sample(frac=1, random_state=seed).reset_index(drop=True)