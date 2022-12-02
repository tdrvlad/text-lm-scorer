from pydantic import BaseModel
import fitz
from utils.lm_scorer import TokenScore
from utils.lm_scorer import LMScorer
import itertools
from tqdm import tqdm


class Color:
    RED = (1, 0, 0)
    YELLOW = (1, 1, 0)


class Thresholds:
    HIGH = 1e-06
    MED = 1e-05


# TODO: move TokenScore here?
class WordObject(BaseModel):
    string: str

    quads: tuple[float, float, float, float]
    page_in_doc: int
    paragraph_in_page: int
    sentence_in_paragraph: int
    word_in_sentence: int

    token_scores: list[TokenScore] = list()


class PDFProcessor:
    __filepath: str
    __words: list[WordObject]
    __lm_scorer: LMScorer

    def __init__(self, filepath: str):
        self.__filepath = filepath
        self.__doc = fitz.open(filepath)
        self.__words = self.retrieve_words_data()
        self.__lm_scorer = LMScorer()

    def get_words(self):
        return self.__words

    def retrieve_words_data(self):
        doc_words = []
        pages = self.__doc.pages(start=None, stop=None)  # note: can't store pages inside the object

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

    def score_paragraphs(self):
        paragraphs = self.get_paragraphs()

        all_token_scores = []
        for paragraph in tqdm(paragraphs):
            paragraph_scores = self.__lm_scorer(paragraph)
            token_scores = list(itertools.chain(*paragraph_scores))  # flatten list
            all_token_scores.extend(token_scores)

        self.match_token_scores_to_words(all_token_scores)

    # TODO: this can be done a lot more efficient
    #   * note: this is currently based on the fact that the word list is in order
    def get_paragraphs(self):
        current_paragraph = self.__words[0].paragraph_in_page
        paragraph_text = ''
        paragraphs = []

        for word in self.__words:
            if word.paragraph_in_page == current_paragraph:
                paragraph_text += f' {word.string}'
            else:
                paragraphs.append(paragraph_text)
                paragraph_text = word.string
                current_paragraph = word.paragraph_in_page

        paragraphs.append(paragraph_text)  # append last paragraph

        return paragraphs

    # TODO: take into consideration the issues below
    #  * strip is very important for word_buffer & prefix here since we're doing exact matching for now
    #  * it doesn't handle ANY noise
    #  * token scores MUST only contain what we have in words, same order, no random characters/spaces
    #  * always takes it from the FIRST word until the LAST
    #  * if it can't find a match in the word it is stuck, doesn't move to the next word: can potentially fix this
    def match_token_scores_to_words(self, token_scores):
        current_token = token_scores[0]
        token_index = 0
        no_of_token_scores = len(token_scores)

        for word in self.__words:
            word_buffer = word.string.strip()
            while word_buffer:
                prefix = current_token.string.strip()
                if word_buffer.startswith(prefix):
                    word.token_scores.append(current_token)
                    word_buffer = word_buffer[len(prefix):]

                    token_index += 1
                    if token_index >= no_of_token_scores:
                        return
                    current_token = token_scores[token_index]

    # this only takes first token into consideration for now
    def get_word_quads_from_page_with_probability(self, page, check_threshold):
        return [word.quads for word in self.__words
                if word.page_in_doc == page and
                word.token_scores and
                word.token_scores[0].prob and  # this check is because 1st word prob is None
                check_threshold(word.token_scores[0].prob)]

    def highlight_mistakes(self):

        pages = self.__doc.pages(start=None, stop=None)  # note: can't store pages inside the object

        for page in pages:
            high_mistake_words_quads = self.get_word_quads_from_page_with_probability(page.number, PDFProcessor.high_threshold)
            highlight = page.add_highlight_annot(high_mistake_words_quads)
            highlight.set_colors({"stroke": Color.RED})
            highlight.update()

            med_mistake_words_quads = self.get_word_quads_from_page_with_probability(page.number, PDFProcessor.med_threshold)
            highlight = page.add_highlight_annot(med_mistake_words_quads)
            highlight.set_colors({"stroke": Color.YELLOW})
            highlight.update()

    @staticmethod
    def high_threshold(prob):
        return prob <= Thresholds.HIGH

    @staticmethod
    def med_threshold(prob):
        return Thresholds.HIGH < prob <= Thresholds.MED

    def save(self, filename):
        self.__doc.save(filename)
