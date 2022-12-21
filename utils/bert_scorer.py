import torch
from transformers import BertForMaskedLM, BertTokenizer

from utils.scorer import TokenScore, TokenProb, WordScore


class Thresholds:
    HIGH = 0.1
    MED = 0.4


top_k = 10


class BERTScorer:
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
            # probabilities = torch.nn.functional.normalize(token_logits, dim=-1)

            token_string = self.tokenizer.decode([input_id])

            token_probability = probabilities[input_id]
            sorted_probs, sorted_ids = torch.sort(probabilities, descending=True)

            sorted_strings = self.tokenizer.decode(sorted_ids[:top_k])

            score = token_probability

            token_index = (sorted_ids == input_id).nonzero().item()
            probs_before = sorted_probs[:token_index]

            if len(probs_before):
                mean_diff = torch.mean(probs_before - token_probability)
                score = (score + (1 - mean_diff)) / 2

            token_index = (sorted_ids == input_id).nonzero().item()

            # top_k_strings = self.tokenizer.decode(sorted_ids[:5])
            #
            # top_k_ids = torch.topk(probabilities, top_k).indices
            # top_k_probs = torch.topk(probabilities, top_k).values
            # top_k_strings = [self.tokenizer.decode(tk_id).replace(" ", "") for tk_id in top_k_ids]
            #
            suggested_tokens = [TokenProb(token_id=tk_id, prob=prob) for tk_id, prob in
                                zip(sorted_ids[:top_k], sorted_probs[:top_k])]
            #
            # rel_probs = top_k_probs - token_probability
            # abs_probs = torch.abs(rel_probs)
            # # norm_rel_top_k_probs = torch.nn.functional.normalize(abs_probs, dim=-1)
            # score = 1 - min(abs_probs)

            token_score = TokenScore(
                token_id=int(input_id),
                score=score,
                suggested_tokens=suggested_tokens
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
            tokenized_word = self.tokenizer.encode(word)[1:-1]  # Ignore BOS and EOS tokens
            n_word_tokens = len(tokenized_word)
            current_word_token_scores = token_scores[current_index: current_index + n_word_tokens]
            word_score = WordScore(
                string=word,
                score=sum([ts.score for ts in current_word_token_scores]) / len(current_word_token_scores),
                suggested_strings=[self.tokenizer.decode([tp.token_id]) for tp in
                                   current_word_token_scores[0].suggested_tokens]
            )
            word_scores.append(word_score)
            current_index += n_word_tokens
        return word_scores

    @staticmethod
    def high_threshold(prob):  # p <= 0.8
        return prob <= Thresholds.HIGH

    @staticmethod
    def med_threshold(prob):  # 0.8 < p <= 0.9
        return Thresholds.HIGH < prob <= Thresholds.MED

    @staticmethod
    def low_threshold(prob):  # 0.9 < p
        return Thresholds.MED < prob


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
