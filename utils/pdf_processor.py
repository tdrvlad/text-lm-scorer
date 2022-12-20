from typing import Tuple, List

import fitz
from pydantic import BaseModel

from utils.bert_scorer import BERTScorer
from utils.scorer import WordScore


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

    word_score: WordScore = None


PUNCTUATION_MARKS = ['.', '...', '!', '?']


def word_ends_in_punctuation(word: WordObject):
    for punctuation_mark in PUNCTUATION_MARKS:
        if word.string.endswith(punctuation_mark):
            return True
    return False


def word_starts_with_capital_letter(word: WordObject):
    return word.string[0].isupper()


def words_on_separate_lines(word_1: WordObject, word_2: WordObject):
    return word_1.paragraph_in_page != word_2.paragraph_in_page


def sentence_split_words_condition(word_obj_1: WordObject, word_obj_2: WordObject):
    """
    Heuristic method for deciding whether there is a sentence split between word_obj_1 and word_obj_2.
    """
    # Case: word ends in punctuation AND following word starts with capital letter
    if word_ends_in_punctuation(word_obj_1) and word_starts_with_capital_letter(word_obj_2):
        return True

    # Case: words are on separate lines and second word starts with capital letter
    if words_on_separate_lines(word_obj_1, word_obj_2) and word_starts_with_capital_letter(word_obj_2):
        return True

    # TODO Add other heuristics TODO

    return False


class PDFProcessor:
    filepath: str
    words: List[WordObject]
    scorer: BERTScorer

    def __init__(self, filepath: str):
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

    # def get_wait_time(self):  # TODO: save this as a class attribute to save time
    #     return len(self.get_paragraphs()) - 1

    # def score_paragraphs(self, with_yield=False):
    #     paragraphs = self.get_paragraphs()
    #
    #     all_word_scores = []
    #     for index, paragraph in enumerate(paragraphs):
    #         paragraph_scores = self.scorer.score_text(paragraph)
    #         all_word_scores.extend(paragraph_scores)
    #
    #         if with_yield:
    #             yield index
    #
    #     assert len(self.words) == len(all_word_scores), f'Different number of words and word scores: {len(self.words)} {len(all_word_scores)}'
    #     for w, ws in zip(self.words, all_word_scores):
    #         w.word_score = ws

    def get_sentences(self):
        """ Returns the pdf sentences as a list of lists of WordObjects."""
        sentences = []
        current_sentence = []
        for word_1, word_2 in zip(self.words[:-1], self.words[1:]):
            current_sentence.append(word_1)
            if sentence_split_words_condition(word_1, word_2):
                sentences.append(current_sentence)
                current_sentence = []
        current_sentence.append(self.words[-1])
        sentences.append(current_sentence)
        return sentences

    def get_sentence_string(self, words: List[WordObject]):
        """ Concatenate the strings in the WordObjects to form the sentence as string."""
        return ' '.join([w.string for w in words])

    def score_sentences(self):
        sentences = self.get_sentences()
        for sentence in sentences:
            sentence_string = self.get_sentence_string(sentence)
            sentence_scores = self.scorer.score_text(sentence_string)
            for w, ws in zip(sentence, sentence_scores):
                w.word_score = ws

    # def get_paragraphs(self):
    #     current_paragraph = self.words[0].paragraph_in_page
    #     paragraph_text = ''
    #     paragraphs = []
    #
    #     for word in self.words:
    #         if word.paragraph_in_page == current_paragraph:
    #             paragraph_text += f' {word.string}'
    #         else:
    #             paragraphs.append(paragraph_text)
    #             paragraph_text = word.string
    #             current_paragraph = word.paragraph_in_page
    #
    #     paragraphs.append(paragraph_text)  # append last paragraph
    #
    #     return paragraphs

    @staticmethod
    def prefix_in_word_buffered_uncased(word_buffer, prefix):
        return word_buffer.lower().startswith(prefix.lower())

    def get_word_quads_with_probabilities(self, page, check_threshold):
        return [word.quads for word in self.words
                if word.page_in_doc == page and
                word.word_score is not None and
                check_threshold(word.word_score.score)]  # this only takes first token prob into consideration

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
