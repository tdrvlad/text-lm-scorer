from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn import Softmax
import torch
from typing import List


class TokenScore:
    def __init__(self, token_id: int, string: str, prob: float = None, suggested_strings: List[str] = None, n_words_ahead: int = 0):
        self.token_id = token_id
        self.string = string
        self.prob = prob
        self.suggested_strings = suggested_strings
        self.n_words_ahead = n_words_ahead


class LMScorer:
    def __init__(self, model_id='gpt2'):
        self.model_id = model_id
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)


    def __call__(self, strings_batch: List[str]):

        tokenized_strings_batch = self.tokenizer(strings_batch, return_tensors="pt")
        input_ids_batch = tokenized_strings_batch.input_ids
        outputs_batch = self.model.forward(input_ids_batch)
        logits_batch = outputs_batch.logits
        logits_probs_batch = torch.nn.functional.softmax(logits_batch, dim=2)
        tokens_scores_batch = []
        for input_ids, logits_probs in zip(input_ids_batch, logits_probs_batch):
            token_strings = [self.tokenizer.decode(input_id) for input_id in input_ids]
            token_scores = []
            # Probability for the first token cannot be computed since it is not proceeded by any token
            first_token_score = TokenScore(
                token_id=input_ids[0],
                string=token_strings[0]
            )
            token_scores.append(first_token_score)
            for i, (token_id, string, prob) in enumerate(zip(input_ids[1:], token_strings[1:], logits_probs[:-1])):
                suggested_tokens = torch.topk(prob, 5).indices
                suggested_strings = [self.tokenizer.decode(suggested_token) for suggested_token in suggested_tokens]
                token_score = TokenScore(
                    token_id=token_id,
                    string=string,
                    prob=prob[token_id],
                    suggested_strings=suggested_strings,
                    n_words_ahead=i+1
                )
                token_scores.append(token_score)
            tokens_scores_batch.append(token_scores)
        return tokens_scores_batch


if __name__ == '__main__':
    lm_scorer = LMScorer()
    texts = [
        'Today is a beautiful day, what shall we do?'
    ]
    scored_texts = lm_scorer(texts)
    for text, scored_text in zip(texts, scored_texts):
        print(f'\n{text}')
        for token_score in scored_text:
            if token_score.prob is None:
                print(f'{token_score.string.strip()}')
            else:
                print(f'{token_score.string.strip()}: {token_score.prob * 100 / token_score.n_words_ahead}% ({token_score.suggested_strings})')