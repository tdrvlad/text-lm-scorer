from transformers import BertForMaskedLM, BertTokenizer
import torch
from typing import List
from utils.scorer import ScorerInterface
import itertools

class Thresholds:
    HIGH = 0.8
    MED = 0.95


class BERTScorer(ScorerInterface):
    def __init__(self, model_id='dumitrescustefan/bert-base-romanian-cased-v1'):

        self.model_id = model_id
        self.model = BertForMaskedLM.from_pretrained(self.model_id, return_dict=True)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_id)

    def score_text(self, text: str):
        input_ids = self.tokenizer([text], return_tensors="pt").input_ids[0]

        input_ids_batch = input_ids.repeat(len(input_ids), 1)
        tokens_mask_batch = 1 - torch.eye(len(input_ids))

        with torch.no_grad():
            logits_batch_list = []
            for i in range(len(input_ids_batch)):
                logits_batch_list.append(self.model(input_ids_batch[i][None, :], attention_mask=tokens_mask_batch[i][None, :]).logits[0])
            # logits_batch = self.model(input_ids_batch, attention_mask=tokens_mask_batch).logits

        logits_batch = torch.stack(logits_batch_list)
        tokens_logits = torch.diagonal(logits_batch).swapaxes(0,1)

        token_scores = []
        for input_id, token_logits in zip(input_ids, tokens_logits):
            probabilities = torch.nn.functional.softmax(token_logits, dim=-1)
            token_probability = probabilities[input_id]

            suggested_input_ids = torch.topk(probabilities, 5).indices
            suggested_tokens = [self.tokenizer.decode(inp_id) for inp_id in suggested_input_ids]

            token_score = ScorerInterface.TokenScore(
                token_id=int(input_id),
                prob=float(token_probability),
                suggested_strings=[suggested_token.replace(" ", "") for suggested_token in suggested_tokens]
            )
            token_scores.append(token_score)

        word_scores = self.token_scores_to_word_scores(token_scores)
        return word_scores

    def token_scores_to_word_scores(self, token_scores):
        word_scores = []
        token_scores = token_scores[1:-1]  # Ignore BOS and EOS tokens
        text = self.tokenizer.decode([tkn_score.token_id for tkn_score in token_scores])
        words = [w for w in text.split(' ') if len(w)]
        current_index = 0
        for word in words:
            tokenized_word = self.tokenizer.encode(word)[1:-1] # Ignore BOS and EOS tokens
            n_word_tokens = len(tokenized_word)
            current_word_token_scores = token_scores[current_index: current_index + n_word_tokens]
            word_score = ScorerInterface.WordScore(
                string=word,
                prob=sum([ts.prob for ts in current_word_token_scores]) / len(current_word_token_scores),
                suggested_strings=list(itertools.chain.from_iterable([ts.suggested_strings for ts in current_word_token_scores]))
            )
            word_scores.append(word_score)
            current_index += n_word_tokens
        return word_scores


    @staticmethod
    def high_threshold(prob):  # p < 0.8
        return prob <= Thresholds.HIGH

    @staticmethod
    def med_threshold(prob):  # 0.8 < p < 0.9
        return Thresholds.HIGH < prob <= Thresholds.MED

if __name__ == '__main__':
    text = 'Primul Război Mondial a început în anul 1980 și a durat 40 de ani.'
    bert_scorer = BERTScorer()
    token_scores = bert_scorer.score_text(text)
    bert_scorer.token_scores_to_word_scores(token_scores)