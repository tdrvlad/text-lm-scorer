from utils.gpt_scorer import GPTScorer
from utils.bert_scorer import BERTScorer


def pretty_print(texts, scored_texts):

    for text, scored_text in zip(texts, scored_texts):

        print(f'Sentence: {text}\n')
        for token_score in scored_text:
            if token_score.prob is None:
                print(f'{token_score.string.strip()}')
            else:
                print(f'{token_score.string.strip()}: '
                      # f'{token_score.prob}% '
                      f'{token_score.prob * 100 / token_score.n_words_ahead:.2f}% '
                      f'({token_score.suggested_strings})')


if __name__ == '__main__':
    text = ['Primul razboi mondial a inceput in anul 2000.']

    print('GPT Scores')
    gpt_socrer = GPTScorer()
    scored_texts = gpt_socrer.score(text)
    pretty_print(text, scored_texts)

    print('BERT Scores')
    bert_scorer = BERTScorer()
    scored_texts = bert_scorer.score(text)
    pretty_print(text, scored_texts)
