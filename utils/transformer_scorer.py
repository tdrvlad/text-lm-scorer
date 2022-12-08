import inspect
import numpy as np
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer



class TransformerScorer:
    def __init__(self, model_id, max_seq_length=256):

        self.model_id = model_id
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        self.max_seq_length = max_seq_length

        if self.tokenizer.pad_token_id is not None:
            self.pad_id = self.tokenizer.pad_token_id

        elif self.tokenizer.eos_token_id is not None:
            self.pad_id = self.tokenizer.eos_token_id
        else:
            logging.warning("Using 0 as pad_id as the tokenizer has no pad_id or eos_id.")
            self.pad_id = 0

        self.tokenizer.padding_side = "right"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if "attention_mask" in inspect.getfullargspec(self.model.forward).args:
            self.support_att_mask = True
        else:
            self.support_att_mask = False

    def __call__(self, input_text):
        if isinstance(input_text, str):
            input_text = [input_text]

        return self._tokenize_and_predict(input_text)


    def _tokenize_and_predict(self, text_batch):
        assert not isinstance(text_batch, str), 'Provide a batch of texts for prediction.'

        with torch.no_grad():
            tokens = self.tokenizer(text_batch, max_length=self.max_seq_length)
            input_ids = tokens['input_ids']
            input_masks = tokens['attention_mask']

        # input_ids, input_masks = self._pad_input(input_ids, input_masks)

        input_ids = torch.tensor(input_ids)
        input_masks = torch.tensor(input_masks)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=input_masks)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        self._decode_probabilities(input_ids, probabilities)

    def _decode_probabilities(self, input_ids_batch, probabilities_batch):
        for input_ids, probabilities in zip(input_ids_batch, probabilities_batch):
            for input_id, prob in zip(input_ids, probabilities):
                print(f'{self.tokenizer.decode(input_id)}: {prob[input_id].cpu().detach().numpy()}')


    def _pad_input(self, input_ids, input_masks):

        max_length = 0
        for i in range(len(input_ids)):
            if self.tokenizer.bos_token_id is not None:
                input_ids[i] = [self.tokenizer.bos_token_id] + input_ids[i]
                input_masks[i] = [1] + input_masks[i]
            if self.tokenizer.eos_token_id is not None:
                input_ids[i] += [self.tokenizer.eos_token_id]
                input_masks[i] += [1]
            max_length = max(max_length, len(input_ids[i]))

        for i in range(len(input_ids)):
            input_ids[i] = np.pad(
                input_ids[i],
                pad_width=(0, max_length - len(input_ids[i])),
                constant_values=0,
            )
            input_masks[i] = np.pad(
                input_masks[i],
                pad_width=(0, max_length - len(input_masks[i])),
                constant_values=0,
            )

        input_ids = torch.tensor(input_ids)
        input_masks = torch.tensor(input_masks)

        return input_ids, input_masks


    #
    #
    #
    # def score_sentence(self, text):
    #     if isinstance(text, str):
    #         text = [text]
    #     with torch.no_grad():
    #         tokens = self.tokenizer(text, max_length=self.max_seq_length)
    #         input_ids = tokens['input_ids']
    #         input_masks = tokens['attention_mask']
    #
    #         max_length = 0
    #         for i in range(len(input_ids)):
    #             if self.tokenizer.bos_token_id is not None:
    #                 input_ids[i] = [self.tokenizer.bos_token_id] + input_ids[i]
    #                 input_masks[i] = [1] + input_masks[i]
    #             if self.tokenizer.eos_token_id is not None:
    #                 input_ids[i] += [self.tokenizer.eos_token_id]
    #                 input_masks[i] += [1]
    #             max_length = max(max_length, len(input_ids[i]))
    #
    #         for i in range(len(input_ids)):
    #             input_ids[i] = np.pad(
    #                 input_ids[i],
    #                 pad_width=(0, max_length - len(input_ids[i])),
    #                 constant_values=0,
    #             )
    #             input_masks[i] = np.pad(
    #                 input_masks[i],
    #                 pad_width=(0, max_length - len(input_masks[i])),
    #                 constant_values=0,
    #             )
    #
    #         input_masks = torch.tensor(input_masks)
    #         input_ids = torch.tensor(input_ids)
    #
    #         with torch.no_grad():
    #             outputs = self.model(input_ids, attention_mask=input_masks)
    #         probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    #         tokens_probs = (
    #             probs[:, :-1].gather(2, input_ids[:, 1:].unsqueeze(2)).squeeze(2)
    #         )
    #         for token_
    #
    #         neural_lm_score = torch.sum(tokens_probs * input_masks[:, 1:], dim=-1)
    #         neural_lm_score_val = neural_lm_score.cpu().detach().numpy()
    #
    #         return neural_lm_score_val

def test_scorer():

    # scorer = TransformerScorer('readerbench/RoGPT2-medium')
    scorer = TransformerScorer('dumitrescustefan/bert-base-romanian-cased-v1')
    texts = [
        "Primul Război Mondial a început în anul 1932 .",
    ]

    for text in texts:
        print("\n{}: {}".format(scorer(text), text))

if __name__ == "__main__":
    test_scorer()


