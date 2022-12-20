import torch.nn.functional
from pydantic import BaseModel
from typing import List
import torch

def compute_score(token_prob, top_k_probs):

    rel_probs = top_k_probs - token_prob
    abs_probs = torch.abs(rel_probs)
    # norm_rel_top_k_probs = torch.nn.functional.normalize(abs_probs, dim=-1)
    score = 1 - min(abs_probs)
    return score


class TokenProb(BaseModel):
    token_id: int
    prob: float = None


class TokenScore(BaseModel):
    token_id: int
    score: float = None
    suggested_tokens: List[TokenProb] = None


class WordScore(BaseModel):
    string: str
    score: float = None
    suggested_strings: List[str] = None


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
