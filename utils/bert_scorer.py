import torch
from transformers import BertForMaskedLM, BertTokenizer
from utils.scorer import TokenProb, WordScore


class Thresholds:
    HIGH = 0.1
    MED = 0.4


TOP_K_WORDS = 10


class BERTScorer:
    def __init__(self):
        self.model_id = 'dumitrescustefan/bert-base-romanian-cased-v1'
        self.model = BertForMaskedLM.from_pretrained(self.model_id, return_dict=True)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_id)

    def score_sentence(self, sentence: str):
        text_input_ids = self.tokenizer([sentence], return_tensors="pt").input_ids[0]

        input_ids_batch = text_input_ids.repeat(len(text_input_ids), 1)
        input_ids_batch.diagonal().fill_(self.tokenizer.mask_token_id)  # apply mask to each individual token

        with torch.no_grad():
            # Process them individually for not causing OOM
            # if we had the memory - logits_batch = self.model(input_ids_batch).logits
            logits_batch = [self.model(torch.unsqueeze(input_ids, 0)).logits[0] for input_ids in input_ids_batch]

        logits_batch = torch.stack(logits_batch)
        tokens_logits = torch.diagonal(logits_batch).swapaxes(0, 1)

        token_probabilities = []
        for input_id, token_logits in zip(text_input_ids, tokens_logits):
            # retrieve sorted probabilities along with token probability
            probabilities = torch.nn.functional.softmax(token_logits, dim=-1)
            sorted_probs, sorted_ids = torch.sort(probabilities, descending=True)
            original_token_probability = probabilities[input_id]

            # compute token score - based on better probabilities
            token_index = (sorted_ids == input_id).nonzero().item()
            probs_before = sorted_probs[:token_index]
            updated_token_probability = original_token_probability
            if len(probs_before):
                mean_diff = torch.mean(probs_before - original_token_probability)
                updated_token_probability = (updated_token_probability + (1 - mean_diff)) / 2

            # retrieve top k tokens & probabilities
            suggested_tokens = [TokenProb(token_id=tk_id, prob=prob)
                                for tk_id, prob in zip(sorted_ids[:TOP_K_WORDS], sorted_probs[:TOP_K_WORDS])]

            # create and append token probability
            token_probability = TokenProb(
                token_id=int(input_id),
                prob=updated_token_probability,
                suggested_tokens=suggested_tokens
            )
            token_probabilities.append(token_probability)

        # match token probabilities to words
        token_scores = self.token_scores_to_word_scores(token_probabilities)

        return token_scores

    def token_scores_to_word_scores(self, token_scores):
        sentence_scores = []

        token_scores = token_scores[1:-1]  # Ignore BOS and EOS tokens
        text = self.tokenizer.decode([tkn_score.token_id for tkn_score in token_scores])
        # split them (?) - still taking the token text - not the pdf one - shouldn't we have the pdf sentence here?
        words = [w for w in text.split(' ') if len(w)]

        current_index = 0
        for word in words:
            tokenized_word = self.tokenizer.encode(word)[1:-1]  # Ignore BOS and EOS tokens
            n_word_tokens = len(tokenized_word)

            current_word_token_scores = token_scores[current_index: current_index + n_word_tokens]
            word_score = WordScore(
                string=word,
                score=sum([ts.prob for ts in current_word_token_scores]) / len(current_word_token_scores),  # mean
                suggested_strings=[self.tokenizer.decode([tp.token_id]) for tp in
                                   current_word_token_scores[0].suggested_tokens]  # first word suggestions only
            )
            sentence_scores.append(word_score)
            current_index += n_word_tokens

        return sentence_scores

    @staticmethod
    def high_threshold(prob):  # p <= 0.8
        return prob <= Thresholds.HIGH

    @staticmethod
    def med_threshold(prob):  # 0.8 < p <= 0.9
        return Thresholds.HIGH < prob <= Thresholds.MED

    @staticmethod
    def low_threshold(prob):  # 0.9 < p
        return Thresholds.MED < prob
