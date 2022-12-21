from utils.pdf_processor import PDFProcessor


if __name__ == '__main__':
    # pdf_processor = PDFProcessor('demo_data/demo.pdf')
    # sentences = pdf_processor.score_sentences()
    # pdf_processor.highlight_mistakes()
    # pdf_processor.save('demo_data/processed-sample.pdf')

    # TODO: !!! take care of cedilla letters

    # TODO: remove
    pdf_processor = PDFProcessor('demo_data/demo-long.pdf')
    sentences = pdf_processor.generate_sentences_as_list_of_words()
    for sentence_words in sentences:
        print('word, word in line, line in paragraph, paragraph in page')
        for word in sentence_words:
            print(f'{word.string} :: '
                  f'{word.word_in_line}, '
                  f'{word.line_in_paragraph}, '
                  f'{word.paragraph_in_page}')

        print()
