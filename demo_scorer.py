from utils.gpt_scorer import GPTScorer
from utils.bert_scorer import BERTScorer


def pretty_print(text, word_scores):

    # print(f'\n\tSentence: {text}\n')
    print(f'\n\tSentence: {text.split(".")[0]}.\n\t\t\t{text.split(".")[1]}.\n')
    # for word_score in word_scores:
    #     print(f'{word_score.string.strip()}: '
    #           f'{word_score.prob * 100:.2f}% '
    #           f'({word_score.suggested_strings})')
    for word_score in word_scores:
        print(f'\t{word_score.string.strip()}: '
              f'{word_score.prob * 100:.0f}%  '
              f'Suggestions: {", ".join(word_score.suggested_strings)}')



if __name__ == '__main__':
    # text = 'Neil Armstrong (n. 5 august 1930, Wapakoneta, Ohio, SUA – d. 25 august 2012, Cincinnati, Ohio, SUA) a fost un astronaut american, pilot de încercare și pilot naval, cunoscut ca fiind primul om care a pășit pe Lună. Primul său zbor spațial a avut loc în 1966. În aceasta misiune el a executat prima andocare a două nave spațiale, împreună cu pilotul David Scott.'
    text = 'Primul război Mondial a început în anul 1980. soldații care merge pe front au condiți grele.'

    bert_scorer = BERTScorer()

    print('BERT Scores')
    word_scores = bert_scorer.score_text(text)
    pretty_print(text, word_scores)
