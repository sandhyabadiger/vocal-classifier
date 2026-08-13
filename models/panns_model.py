import numpy as np
from .base import SoundTagger


class PANNsModel(SoundTagger):
    name = "panns"
    sample_rate = 32000  # PANNs expects 32kHz

    def __init__(self):
        from panns_inference import AudioTagging
        self._model = AudioTagging(checkpoint_path=None, device="cpu")

    def top_predictions(self, audio, top_k=5):
        audio = np.asarray(audio, dtype=np.float32)
        clip = audio[np.newaxis, :]  # panns_inference wants shape (batch, samples)
        clipwise_output, _embedding = self._model.inference(clip)
        scores = clipwise_output[0]
        top_indices = scores.argsort()[-top_k:][::-1]
        labels = self._model.labels
        return [(labels[i], float(scores[i])) for i in top_indices]
