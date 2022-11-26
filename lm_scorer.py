from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn import Softmax
import torch


class LMScorer:
    def __init__(self, model_id='gpt2'):
        self.model_id = model_id
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        self.index_to_token_dict = {v: k for k, v in self.tokenizer.get_vocab().items()}

    def get_vocab_token(self, index):
        return self.index_to_token_dict[index]

    def __call__(self, input_string):
        torch_softmax = Softmax(dim=2)

        inputs = self.tokenizer(input_string, return_tensors="pt")
        outputs = self.model(**inputs, labels=inputs["input_ids"])

        sentence_length = len(inputs["input_ids"].flatten())
        tokens_prob = torch_softmax(outputs.logits)

        actual_token_index = inputs["input_ids"].flatten()[1:].tolist()
        actual_token_p = tokens_prob[0, range(0, sentence_length - 1), actual_token_index].tolist()
        best_token_index = torch.max(tokens_prob, 2).indices.flatten().tolist()
        best_token_p = torch.max(tokens_prob, 2).values.flatten().tolist()

        return zip(
            [self.get_vocab_token(index) for index in actual_token_index], actual_token_p,
            [self.get_vocab_token(index) for index in best_token_index], best_token_p
        )
