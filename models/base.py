from abc import ABC, abstractmethod


class SoundTagger(ABC):
    #common interface both AST and PANNS implement

    name: str
    sample_rate: int

    @abstractmethod
    def top_predictions(self, audio, top_k=5):
        # audio is a mono float32 numpy array at self.sample_rate
        # returns a list of (label, score) tuples- sorted descending
        raise NotImplementedError
