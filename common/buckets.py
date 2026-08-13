# Human-sound bucket mapping, shared across AST and YAMNet since they
# both use AudioSet-style labels, so the same keyword mapping works for
# either one.

HUMAN_SOUND_KEYWORDS = {
    "affect_burst": ["gasp", "sigh", "groan", "wail", "moan", "whimper",
                      "scream", "cry", "sob", "yell", "bellow", "whoop"],
    "laughter": ["laugh", "giggle", "chuckle", "snicker", "belly laugh"],
    "cough": ["cough"],
    "breathing": ["breath", "wheeze", "snore", "pant", "sniff"],
    "hiccup": ["hiccup"],
}


def bucket_scores_from_predictions(preds):
    # preds is a list of (label, score) tuples, returns a dict of bucket to max score
    scores = {bucket: 0.0 for bucket in HUMAN_SOUND_KEYWORDS}
    for label, score in preds:
        label_lower = label.lower()
        for bucket, keywords in HUMAN_SOUND_KEYWORDS.items():
            if any(k in label_lower for k in keywords):
                scores[bucket] = max(scores[bucket], score)
    return scores


def best_bucket(preds):
    # returns the winning bucket name and its score
    scores = bucket_scores_from_predictions(preds)
    winner = max(scores, key=scores.get)
    return winner, scores[winner]
