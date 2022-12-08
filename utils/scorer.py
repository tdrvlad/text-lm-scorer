from pydantic import BaseModel
from typing import List


class ScorerInterface:

    class TokenScore(BaseModel):
        token_id: int
        string: str
        prob: float = None
        suggested_strings: List[str] = None
        n_words_ahead: int = 0

    # TODO: fix this - doesn't work with list of strings yet, but with one string
    def score(self, strings_batch: List[str]) -> List[TokenScore]:
        """Extract token scores from list of strings."""
        pass

    @staticmethod
    def high_threshold(prob: float) -> bool:
        pass

    @staticmethod
    def med_threshold(prob: float) -> bool:
        pass

    # TODO: not used yet - need to use this to preprocess text
    @staticmethod
    def replace_cedilla_letters(string):
        return string.replace("ţ", "ț").replace("ş", "ș").replace("Ţ", "Ț").replace("Ş", "Ș")
