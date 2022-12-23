from typing import List, Any
from pydantic import BaseModel


class TokenScore(BaseModel):
    token_id: int
    prob: float = None
    suggested_tokens: List[Any] = None  # type List[TokenProb]
