from utils.lm_scorer import LMScorer


def pretty_print(texts, scored_texts):

    for text, scored_text in zip(texts, scored_texts):
        print(f'\n{text}\n')
        for token_score in scored_text:
            if token_score.prob is None:
                print(f'{token_score.string.strip()}')
            else:
                print(f'{token_score.string.strip()}: '
                      f'{token_score.prob * 100 / token_score.n_words_ahead:.2f}% '
                      f'({token_score.suggested_strings})')


if __name__ == '__main__':
    lm_scorer = LMScorer()

    text = ['Today is a beautiful day, what shall we do?']
    scored_texts = lm_scorer(text)
    pretty_print(text, scored_texts)
