import numpy as np
from .base import SoundTagger


class ASTModel(SoundTagger):
    name = "ast"
    sample_rate = 16000

    def __init__(self):
        from transformers import pipeline
        self._classifier = pipeline(
            "audio-classification",
            model="MIT/ast-finetuned-audioset-10-10-0.4593",
        )

    def top_predictions(self, audio, top_k=5):
        audio = np.asarray(audio, dtype=np.float32)
        preds = self._classifier({"array": audio, "sampling_rate": self.sample_rate})
        return [(p["label"], p["score"]) for p in preds[:top_k]]
