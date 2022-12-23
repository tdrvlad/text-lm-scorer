import torch
from transformers import BertForMaskedLM, BertTokenizer
from utils.scorer import TokenScore


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

        token_scores = []
        for input_id, token_logits in zip(text_input_ids, tokens_logits):
            # retrieve sorted probabilities along with token probability
            probs = torch.nn.functional.softmax(token_logits, dim=-1)
            sorted_probs, sorted_ids = torch.sort(probs, descending=True)
            original_token_prob = probs[input_id]

            # compute token score - based on better probabilities
            token_index = (sorted_ids == input_id).nonzero().item()
            probs_before = sorted_probs[:token_index]
            updated_token_prob = original_token_prob
            if len(probs_before):
                mean_diff = torch.mean(probs_before - original_token_prob)
                updated_token_prob = (updated_token_prob + (1 - mean_diff)) / 2

            # retrieve top k tokens & scores
            suggested_tokens = [TokenScore(token_id=tk_id, prob=prob)
                                for tk_id, prob in zip(sorted_ids[:TOP_K_WORDS], sorted_probs[:TOP_K_WORDS])]

            # create and append token score
            token_score = TokenScore(
                token_id=int(input_id),
                prob=updated_token_prob,
                suggested_tokens=suggested_tokens
            )
            token_scores.append(token_score)

        token_scores = token_scores[1:-1]  # Ignore BOS and EOS tokens
        return token_scores

    def encode(self, string):
        return self.tokenizer.encode(string)[1:-1]  # Ignore BOS and EOS tokens

    def decode(self, token_id):
        return self.tokenizer.decode([token_id])  # why in []?

    @staticmethod
    def high_threshold(prob):  # p <= 0.8
        return prob <= Thresholds.HIGH

    @staticmethod
    def med_threshold(prob):  # 0.8 < p <= 0.9
        return Thresholds.HIGH < prob <= Thresholds.MED

    @staticmethod
    def low_threshold(prob):  # 0.9 < p
        return Thresholds.MED < prob
