from typing import Tuple, List

import fitz
from pydantic import BaseModel

from utils.bert_scorer import BERTScorer
from utils.scorer import TokenScore


class Color:
    RED = (1, 0, 0)
    YELLOW = (1, 1, 0)
    GREEN = (0, 1, 0)


class WordObject(BaseModel):
    string: str

    quads: Tuple[float, float, float, float]
    page_in_doc: int
    paragraph_in_page: int
    line_in_paragraph: int
    word_in_line: int

    word_score: float = None
    suggested_strings: List[str] = None
    tokens: List[TokenScore] = None


PUNCTUATION_MARKS = ['.', '...', '!', '?']


def ends_in_punctuation(word: WordObject):
    for punctuation_mark in PUNCTUATION_MARKS:
        if word.string.endswith(punctuation_mark):
            return True
    return False


def starts_with_capital_letter(word: WordObject):
    return word.string[0].isupper()


def in_different_paragraphs(current_word: WordObject, next_word: WordObject):
    return current_word.paragraph_in_page != next_word.paragraph_in_page


def words_in_different_sentences(current_word: WordObject, next_word: WordObject):
    """ Heuristic method for deciding whether there word1 and word2 belong to different sentences."""

    # Case: current word ends in punctuation and following word starts with capital letter
    if ends_in_punctuation(current_word) and starts_with_capital_letter(next_word):
        return True

    # TODO: here you are not checking for separate lines but for separate paragraphs ()
    #  which one do you want? was called on_separate_lines before
    # Case: words are on separate lines and second word starts with capital letter
    if in_different_paragraphs(current_word, next_word) and starts_with_capital_letter(next_word):
        return True

    # TODO Add other heuristics

    return False


def get_sentence_string(words: List[WordObject]):
    """ Concatenate the strings in the list of WordObjects to form the sentence as string."""
    return ' '.join([w.string for w in words])


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
                    line_in_paragraph=raw_word[6],
                    word_in_line=raw_word[7]
                ))

            doc_words.extend(processed_words)

        return doc_words

    def score_sentences(self):
        sentences = self.generate_sentences_as_list_of_words()

        for sentence_words in sentences:
            sentence_string = get_sentence_string(sentence_words)
            token_scores = self.scorer.score_sentence(sentence_string)
            self.match_token_scores_against_words(token_scores, sentence_words)

        return sentences

    def match_token_scores_against_words(self, token_scores: List[TokenScore], sentence_words: List[WordObject]):
        current_token_index = 0

        for word_object in sentence_words:
            tokenized_word = self.scorer.encode(word_object.string)

            n_tokens_in_word = len(tokenized_word)
            current_word_token_scores = token_scores[current_token_index: current_token_index + n_tokens_in_word]

            # assign tokens score and suggestions to word our objects
            word_object.word_score = sum([ts.prob for ts in current_word_token_scores]) / len(current_word_token_scores)
            word_object.suggested_strings = [self.scorer.decode(tp.token_id) for tp in
                                             current_word_token_scores[0].suggested_tokens]
            word_object.tokens = current_word_token_scores

            current_token_index += n_tokens_in_word

    def generate_sentences_as_list_of_words(self):
        """ Returns the pdf sentences as a list of lists of WordObjects."""

        sentences = []
        current_sentence = []

        for current_word, next_word in zip(self.words[:-1], self.words[1:]):
            current_sentence.append(current_word)
            if words_in_different_sentences(current_word, next_word):
                sentences.append(current_sentence)
                current_sentence = []

        current_sentence.append(self.words[-1])
        sentences.append(current_sentence)

        return sentences

    def get_word_quads_with_probabilities(self, page, check_threshold):
        return [word.quads for word in self.words
                if word.page_in_doc == page and
                word.word_score is not None and
                check_threshold(word.word_score)]

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

            # this is added just to highlight all else in green
            low_mistake_words_quads = self.get_word_quads_with_probabilities(page.number, self.scorer.low_threshold)
            highlight = page.add_highlight_annot(low_mistake_words_quads)
            highlight.set_colors({"stroke": Color.GREEN})
            highlight.update()

    def get_scorer(self):
        return self.scorer

    def save(self, filename):
        self.doc.save(filename)
