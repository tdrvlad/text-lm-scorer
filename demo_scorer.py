from utils.bert_scorer import BERTScorer


def pretty_print(text, word_scores):
    print(f'\n\tSentence: {text.split(".")[0]}.\n\t\t\t{text.split(".")[1]}.\n')
    for word_score in word_scores:
        print(f'\t{word_score.string.strip()}: '
              f'{word_score.score * 100:.1f}%  '
              f'Suggestions: {", ".join(word_score.suggested_strings)}')


if __name__ == '__main__':

    # text = 'Neil Armstrong (n. 5 august 1930, Wapakoneta, Ohio, SUA – d. 25 august 2012, Cincinnati, Ohio, SUA) ' \
    #        'a fost un astronaut american, pilot de încercare și pilot naval, cunoscut ca fiind primul om care a ' \
    #        'pășit pe Lună. Primul său zbor spațial a avut loc în 1966. În aceasta misiune el a executat prima ' \
    #        'andocare a două nave spațiale, împreună cu pilotul David Scott.'
    # text = 'Primul Război Mondial a început în anul 1980. Soldații care merg pe front au condiții grele.'
    # text = 'Ana are trei mere.'

    text = 'Primul Război Mondial a început în anul 1980 și a durat 40 de ani.'

    bert_scorer = BERTScorer()
    word_scores = bert_scorer.score_sentence(text)

    pretty_print(text, word_scores)
