from utils.pdf_processor import PDFProcessor


# TODO: !!! take care of cedilla letters
def replace_cedilla_letters(string):
    return string.replace("ţ", "ț").replace("ş", "ș").replace("Ţ", "Ț").replace("Ş", "Ș")


if __name__ == '__main__':
    pdf_processor = PDFProcessor('demo_data/demo.pdf')
    sentences = pdf_processor.score_sentences()

    pdf_processor.highlight_mistakes()
    pdf_processor.save('demo_data/processed-sample.pdf')

    for sentence in sentences[:3]:
        print('word, word in line, line in paragraph, paragraph in page')
        for word in sentence:
            print(f'{word.string} :: '
                  f'{word.word_score}, '
                  f'{word.suggested_strings}')

    print()