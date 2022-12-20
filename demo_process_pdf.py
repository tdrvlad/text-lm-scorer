from utils.pdf_processor import PDFProcessor

# TODO:
#  * words are attached to punctuation marks - look into splitting them
#  * error if a paragraph is split on 2 pages currently
#  * figure out how you want to structure it - words/sentences/paragraphs

if __name__ == '__main__':
    pdf_processor = PDFProcessor('demo_data/demo.pdf')
    sentences = pdf_processor.score_sentences()
    pdf_processor.highlight_mistakes()
    pdf_processor.save('demo_data/processed-sample.pdf')

    # TODO: remove
    # sentences_strings = [pdf_processor.get_sentence_string(s) for s in sentences]
    # print(sentences_strings)
