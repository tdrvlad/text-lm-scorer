from typing import List, Tuple

import fitz  # install with 'pip install pymupdf'
from collections import Counter
import re

LETTERS_REGEX = '[a-zA-ZăĂâÂîÎșȘțȚ)]'


def parse_highlight(annot: fitz.Annot, wordlist: List[Tuple[float, float, float, float, str, int, int, int]]) -> str:
    points = annot.vertices
    quad_count = int(len(points) / 4)
    sentences = []
    for i in range(quad_count):
        # where the highlighted part is
        r = fitz.Quad(points[i * 4 : i * 4 + 4]).rect

        words = [w for w in wordlist if fitz.Rect(w[:4]).intersects(r)]
        sentences.append(" ".join(w[4] for w in words))
    sentence = " ".join(sentences)
    string_split = sentence.split(",")
    sent = ",".join(sorted(set(string_split), key=string_split.index))
    # output = re.sub(r'\b(\d+(?:\.\d+)?)\b', r'\1,', sent)
    return sent


def parse_text_line(text_string):
    """
    Remove whitespace after line when the line connects two adjacent sub-words.
    :param text_string:
    :return:
    """
    text_string = re.sub(f"(?<={LETTERS_REGEX})- (?={LETTERS_REGEX})", '-', text_string)
    return text_string


def get_page_annotations(page: fitz.Page) -> Tuple[List[str], List[str]]:
    """
    Returns two lists containing the highlighted original text and the corresponding annotation content.
    :param page: the current document page
    :return: highlights, annotations
    """
    highlights = []
    annotations = []
    page_words = page.get_text("words")
    for annot in page.annots():
        annotations.append(annot.info['content'])
        highlights.append(parse_highlight(annot, page_words))
    return highlights, annotations


def get_doc_annotations(doc, start=None, stop=None):
    pages = doc.pages(start=start, stop=stop)
    highlights = []
    annotations = []
    highlights_pages = []
    for i, page in enumerate(pages):
        page_highlights, page_annotations = get_page_annotations(page)
        highlights.extend(page_highlights)
        annotations.extend(page_annotations)
        highlights_pages.extend([i] * len(highlights))
    return highlights, annotations, highlights_pages



def process_word_tuple(w):
    return tuple([int(w[0]), int(w[1]), int(w[2]), int(w[3]), w[4]])


def add_page_delta(w, page_number):
    if w[4].isnumeric():
        return tuple([w[0], w[1], w[2], w[3], w[4], int(w[4]) - page_number])
    else:
        return w

def remove_page_delta(w):
    return tuple([w[0], w[1], w[2], w[3], w[4]])


def get_doc_words(doc, start=None, stop=None):
    doc_words = []
    pages = doc.pages(start=start, stop=stop)
    for page in pages:
        page_words_raw = page.get_text("words")
        doc_words.extend(page_words_raw)
    return doc_words


def get_heading_and_footer_words(doc: fitz.Document, page_ratio_threshold=0.4, start=None, stop=None):
    """
    Considerations:
    - headers & footers: look for identical words appearing at the same position across multiple pages
    - page numbers: loom for numerical words that follow the same value difference to the page number and appear at similar positions in the document.
    :param doc:
    :return:
    """
    pages = doc.pages(start=start, stop=stop)
    word_counter = Counter()
    numeric_word_counter = Counter()
    numeric_word_dict = {}

    doc_words = []

    for page in pages:
        page_words_raw = page.get_text("words")
        doc_words.extend(page_words_raw)

        page_words = [process_word_tuple(w) for w in page_words_raw]
        word_counter.update(page_words)

        page_words_numeric = [w for w in page_words if (w[4]).isnumeric()]
        page_words_numeric = [add_page_delta(w, page.number) for w in page_words_numeric]

        numeric_word_counter.update([w[5] for w in page_words_numeric])

        for w in page_words_numeric:
            if w[5] not in numeric_word_dict:
                numeric_word_dict[w[5]] = []
            numeric_word_dict[w[5]].append(w)

    heading_footer_words_candidates = [item[0] for item in word_counter.items() if item[1] > doc.page_count * page_ratio_threshold]
    page_numbering_candidates = numeric_word_dict[numeric_word_counter.most_common(1)[0][0]]
    page_numbering_candidates = [remove_page_delta(w) for w in page_numbering_candidates]

    words_to_remove = page_numbering_candidates + heading_footer_words_candidates
    return words_to_remove


def reconstruct_text(doc_words, respect_newline=True):
    """
    w[5] gives off the document line index. If two words have equal w[5] they are separated by
        :param doc_words:
    :return:
    """
    last_doc_word = None
    text = ''
    for w in doc_words:
        if last_doc_word is None:
            text += w[4]
        else:
            if w[5] == last_doc_word[5] or not respect_newline:
                text += f' {w[4]}'
            else:
                text += f'\n{w[4]}'
        last_doc_word = w
    return text


def reconstruct_sentences(doc_words):
    sentences = []
    last_sentence_words = []


def get_doc_text(doc: fitz.Document, start=None, stop=None, remove_header_and_footer=True, header_footer_words_count_page_ratio=0.4):

    doc_words = get_doc_words(doc, start=start, stop=stop)
    if remove_header_and_footer:
        words_to_remove = get_heading_and_footer_words(doc, start=start, stop=stop, page_ratio_threshold=header_footer_words_count_page_ratio)
        doc_words = [w for w in doc_words if process_word_tuple(w) not in words_to_remove]

    text = reconstruct_text(doc_words)
    return text


def get_text_from_page_crop(page):
    page_words = page.get_text("words")
    return parse_highlight(
        annot=fitz.Annot(),
        wordlist=page_words
    )

def get_doc_perplexity(doc):
    pass


def main(filepath: str):
    doc = fitz.open(filepath)
    text = get_doc_text(doc)
    paragraphs = text.split('\n')
    for p in paragraphs:
        print('\n', p)



if __name__ == "__main__":
    main('demo/sample.pdf')
