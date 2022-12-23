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
