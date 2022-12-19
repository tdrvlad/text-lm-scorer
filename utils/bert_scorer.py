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
        text_input_ids = self.tokenizer([text], return_tensors="pt").input_ids[0]

        input_ids_batch = text_input_ids.repeat(len(text_input_ids), 1)

        # Apply mask to each individual token
        input_ids_batch.diagonal().fill_(self.tokenizer.mask_token_id)

        with torch.no_grad():
            # Process them individually for not causing OOM
            # logits_batch = self.model(input_ids_batch).logits if we had the memory
            logits_batch = [self.model(torch.unsqueeze(input_ids, 0)).logits[0] for input_ids in input_ids_batch]

        logits_batch = torch.stack(logits_batch)
        tokens_logits = torch.diagonal(logits_batch).swapaxes(0, 1)

        token_scores = []
        for input_id, token_logits in zip(text_input_ids, tokens_logits):
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
                # suggested_strings=list(itertools.chain.from_iterable([ts.suggested_strings for ts in current_word_token_scores]))
                suggested_strings=current_word_token_scores[0].suggested_strings
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


def test_bert():
    model_id = 'dumitrescustefan/bert-base-romanian-cased-v1'

    model = BertForMaskedLM.from_pretrained(model_id, return_dict=True)
    tokenizer = BertTokenizer.from_pretrained(model_id)

    text = 'Primul Război Mondial a început în anul 1980 și a durat 40 de ani.'
    tokenized_input = tokenizer([text, 'Ana are mere'], return_tensors="pt")


def test_bert_scorer():
    bert_scorer = BERTScorer()
    text = 'Primul Război Mondial a început în anul 1980 și a durat 40 de ani.'

    word_scores = bert_scorer.score_text(text)
    print(word_scores)

if __name__ == '__main__':
    test_bert_scorer()