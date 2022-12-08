from transformers import BertForMaskedLM, BertTokenizer
import torch
from typing import List
from utils.scorer import ScorerInterface


class Thresholds:
    HIGH = 0.8
    MED = 0.9


class BERTScorer(ScorerInterface):
    def __init__(self):
        self.model_id = 'dumitrescustefan/bert-base-romanian-cased-v1'
        self.model = BertForMaskedLM.from_pretrained(self.model_id, return_dict=True)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_id)

    def score(self, strings_batch: List[str]):
        tokenized_strings_batch = self.tokenizer(strings_batch, return_tensors="pt")
        input_ids_batch = tokenized_strings_batch.input_ids

        outputs_batch = self.model.forward(input_ids_batch)
        logits_batch = outputs_batch[0]
        probs_batch = torch.nn.functional.softmax(logits_batch, dim=-1)

        tokens_scores_batch = []
        for token_ids, token_probs in zip(input_ids_batch, probs_batch):
            token_strings = [self.tokenizer.decode(input_id) for input_id in token_ids]
            token_scores = []

            # remove [CLS] and [SEP] tokens
            token_ids = token_ids[1:-1]
            token_strings = token_strings[1:-1]
            token_probs = token_probs[1:-1]

            for i, (token_id, string, prob) in enumerate(zip(token_ids, token_strings, token_probs)):
                suggested_tokens = torch.topk(prob, 5).indices
                suggested_strings = [self.tokenizer.decode(suggested_token) for suggested_token in suggested_tokens]

                token_scores.append(ScorerInterface.TokenScore(
                    token_id=int(token_id),
                    string=string.replace(" ", "").replace('##', ''),  # join letters & remove ## in split words
                    prob=float(prob[token_id]),
                    suggested_strings=[string.replace(" ", "") for string in suggested_strings],
                    n_words_ahead=i+1
                ))

            tokens_scores_batch.append(token_scores)

        return tokens_scores_batch

    @staticmethod
    def high_threshold(prob):  # p < 0.8
        return prob <= Thresholds.HIGH

    @staticmethod
    def med_threshold(prob):  # 0.8 < p < 0.9
        return Thresholds.HIGH < prob <= Thresholds.MED
