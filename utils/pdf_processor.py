from pydantic import BaseModel
import fitz
from utils.scorer import ScorerInterface
from utils.gpt_scorer import GPTScorer
from utils.bert_scorer import BERTScorer
import itertools
from typing import Union, Tuple, List


class Color:
    RED = (1, 0, 0)
    YELLOW = (1, 1, 0)


class WordObject(BaseModel):
    string: str

    quads: Tuple[float, float, float, float]
    page_in_doc: int
    paragraph_in_page: int
    sentence_in_paragraph: int
    word_in_sentence: int

    word_score: ScorerInterface.WordScore = None



class ScorerType:
    BERT = 'BERT'
    GPT = 'GPT'


class PDFProcessor:
    filepath: str
    words: List[WordObject]
    scorer: Union[GPTScorer, BERTScorer]

    def __init__(self, filepath: str, scorer: ScorerType = ScorerType.BERT):
        self.filepath = filepath
        self.doc = fitz.open(filepath)
        self.words = self.retrieve_words_data()
        self.scorer = BERTScorer()

    def get_words(self):
        return self.words

    def retrieve_words_data(self):
        doc_words = []
        pages = self.doc.pages(start=None, stop=None)  # note: can't store pages inside the object

        for page_index, page in enumerate(pages):
            raw_words = page.get_text("words")
            processed_words = []

            for raw_word in raw_words:
                processed_words.append(WordObject(
                    string=raw_word[4],
                    quads=raw_word[:4],
                    page_in_doc=page_index,
                    paragraph_in_page=raw_word[5],
                    sentence_in_paragraph=raw_word[6],
                    word_in_sentence=raw_word[7]
                ))

            doc_words.extend(processed_words)

        return doc_words

    def get_wait_time(self):  # TODO: save this as a class attribute to save time
        return len(self.get_paragraphs()) - 1

    def score_paragraphs(self, with_yield=False):
        paragraphs = self.get_paragraphs()

        all_word_scores = []
        for index, paragraph in enumerate(paragraphs):
            paragraph_scores = self.scorer.score_text(paragraph)
            all_word_scores.extend(paragraph_scores)

            if with_yield:
                yield index

        assert len(self.words) == len(all_word_scores), 'Different number of words and word scores'
        for w, ws in zip(self.words, all_word_scores):
            w.word_score = ws

    # TODO: this can be done a lot more efficient
    #   * note: this is currently based on the fact that the word list is in order
    def get_paragraphs(self):
        current_paragraph = self.words[0].paragraph_in_page
        paragraph_text = ''
        paragraphs = []

        for word in self.words:
            if word.paragraph_in_page == current_paragraph:
                paragraph_text += f' {word.string}'
            else:
                paragraphs.append(paragraph_text)
                paragraph_text = word.string
                current_paragraph = word.paragraph_in_page

        paragraphs.append(paragraph_text)  # append last paragraph

        return paragraphs


    @staticmethod
    def prefix_in_word_buffered_uncased(word_buffer, prefix):
        return word_buffer.lower().startswith(prefix.lower())

    def get_word_quads_with_probabilities(self, page, check_threshold):
        return [word.quads for word in self.words
                if word.page_in_doc == page and
                word.word_score is not None and
                check_threshold(word.word_score.prob)]  # this only takes first token prob into consideration

    def highlight_mistakes(self):

        pages = self.doc.pages(start=None, stop=None)  # note: can't store pages inside the object

        for page in pages:
            high_mistake_words_quads = self.get_word_quads_with_probabilities(page.number, self.scorer.high_threshold)
            highlight = page.add_highlight_annot(high_mistake_words_quads)
            highlight.set_colors({"stroke": Color.RED})
            highlight.update()

            med_mistake_words_quads = self.get_word_quads_with_probabilities(page.number, self.scorer.med_threshold)
            highlight = page.add_highlight_annot(med_mistake_words_quads)
            highlight.set_colors({"stroke": Color.YELLOW})
            highlight.update()

    def get_scorer(self):
        return self.scorer

    def save(self, filename):
        self.doc.save(filename)
