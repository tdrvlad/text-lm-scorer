from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List
from utils.scorer import TokenProb


class Thresholds:
    HIGH = 1e-06
    MED = 1e-05


# Not used for now
class GPTScorer:
    def __init__(self):
        self.model_id = 'readerbench/RoGPT2-medium'
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

    def score(self, strings_batch: List[str]):
        tokenized_strings_batch = self.tokenizer(strings_batch, return_tensors="pt")

        input_ids_batch = tokenized_strings_batch.input_ids
        outputs_batch = self.model.forward(input_ids_batch)
        logits_batch = outputs_batch.logits
        probs_batch = torch.nn.functional.softmax(logits_batch, dim=2)

        tokens_scores_batch = []
        for token_ids, token_probs in zip(input_ids_batch, probs_batch):
            token_strings = [self.tokenizer.decode(input_id) for input_id in token_ids]
            token_scores = []

            # Probability for the first token cannot be computed since it is not proceeded by any token
            first_token_score = TokenProb(
                token_id=token_ids[0],
                string=token_strings[0]
            )
            token_scores.append(first_token_score)

            for i, (token_id, string, prob) in enumerate(zip(token_ids[1:], token_strings[1:], token_probs[:-1])):
                suggested_tokens = torch.topk(prob, 5).indices
                suggested_strings = [self.tokenizer.decode(suggested_token) for suggested_token in suggested_tokens]

                token_scores.append(TokenProb(
                    token_id=token_id,
                    string=string,
                    prob=prob[token_id],
                    suggested_strings=suggested_strings,
                    n_words_ahead=i+1
                ))

            tokens_scores_batch.append(token_scores)

        return tokens_scores_batch

    @staticmethod
    def high_threshold(prob):
        return prob <= Thresholds.HIGH

    @staticmethod
    def med_threshold(prob):
        return Thresholds.HIGH < prob <= Thresholds.MED
