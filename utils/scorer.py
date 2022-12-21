from typing import List, Any
from pydantic import BaseModel


class TokenProb(BaseModel):
    token_id: int
    prob: float = None
    suggested_tokens: List[Any] = None  # type List[TokenProb]


class WordScore(BaseModel):
    string: str
    score: float = None
    suggested_strings: List[str] = None

    def score_sentence(self, sentence: str) -> List[Any]:
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
