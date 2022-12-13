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
                      f'{token_score.prob * 100:.2f}% '
                      f'({token_score.suggested_strings})')


if __name__ == '__main__':
    text = ['Primul Război Mondial a început în anul 1923.']


    print('BERT Scores')
    bert_scorer = BERTScorer()

    scored_texts = bert_scorer.score(text)
    pretty_print(text, scored_texts)
