from transformers import BertForMaskedLM, BertTokenizer
import torch
from typing import List
from utils.scorer import ScorerInterface


class Thresholds:
    HIGH = 0.8
    MED = 0.9


class BERTScorer(ScorerInterface):
    def __init__(self, model_id='dumitrescustefan/bert-base-romanian-cased-v1'):

        self.model_id = model_id
        self.model = BertForMaskedLM.from_pretrained(self.model_id, return_dict=True)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_id)

    def _score_text(self, text: str):
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids[0]

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
            token = self.tokenizer.decode(input_id)
            token_probability = probabilities[input_id]

            suggested_input_ids = torch.topk(probabilities, 5).indices
            suggested_tokens = [self.tokenizer.decode(inp_id) for inp_id in suggested_input_ids]

            token_score = ScorerInterface.TokenScore(
                token_id=int(input_id),
                string=token.replace(" ", "").replace('##', ''),  # join letters & remove ## in split words
                prob=float(token_probability),
                suggested_strings=[suggested_token.replace(" ", "") for suggested_token in suggested_tokens],
                n_words_ahead=len(token_scores) + 1
            )
            token_scores.append(token_score)
        return token_scores

    def score(self, texts_batch: List[str]):
        token_scores_batch = [self._score_text(text) for text in texts_batch]
        return token_scores_batch

        # # To REDO
        # tokenized_strings_batch = self.tokenizer(strings_batch, return_tensors="pt")
        # input_ids_batch = tokenized_strings_batch.input_ids
        #
        # outputs_batch = self.model.forward(input_ids_batch)
        # logits_batch = outputs_batch[0]
        # probs_batch = torch.nn.functional.softmax(logits_batch, dim=2)
        #
        # tokens_scores_batch = []
        # for token_ids, token_probs in zip(input_ids_batch, probs_batch):
        #     token_strings = [self.tokenizer.decode(input_id) for input_id in token_ids]
        #     token_scores = []
        #
        #     # remove [CLS] and [SEP] tokens
        #     token_ids = token_ids[1:-1]
        #     token_strings = token_strings[1:-1]
        #     token_probs = token_probs[1:-1]
        #
        #     for i, (token_id, string, prob) in enumerate(zip(token_ids, token_strings, token_probs)):
        #         suggested_tokens = torch.topk(prob, 5).indices
        #         suggested_strings = [self.tokenizer.decode(suggested_token) for suggested_token in suggested_tokens]
        #
        #         token_scores.append(ScorerInterface.TokenScore(
        #             token_id=int(token_id),
        #             string=string.replace(" ", "").replace('##', ''),  # join letters & remove ## in split words
        #             prob=float(prob[token_id]),
        #             suggested_strings=[string.replace(" ", "") for string in suggested_strings],
        #             n_words_ahead=i+1
        #         ))
        #
        #     tokens_scores_batch.append(token_scores)
        #
        # return tokens_scores_batch

    @staticmethod
    def high_threshold(prob):  # p < 0.8
        return prob <= Thresholds.HIGH

    @staticmethod
    def med_threshold(prob):  # 0.8 < p < 0.9
        return Thresholds.HIGH < prob <= Thresholds.MED
