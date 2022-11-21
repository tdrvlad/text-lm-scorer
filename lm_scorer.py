from transformers import AutoModelForCausalLM, AutoTokenizer


class LMScorer:
    def __init__(self, model_id='gpt2'):
        self.model_id = model_id
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

    def __call__(self, input_string):
        #TODO
        pass

