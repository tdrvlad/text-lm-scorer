from utils.pdf_processor import PDFProcessor, ScorerType

# TODO:
#  * words are attached to punctuation marks - look into splitting them
#  * error if a paragraph is split on 2 pages currently
#  * figure out how you want to structure it - words/sentences/paragraphs

if __name__ == "__main__":
    sample_pdf = PDFProcessor('demo_data/demo.pdf', ScorerType.BERT)
    for percent_complete in sample_pdf.score_paragraphs(with_yield=True):
        pass
    sample_pdf.highlight_mistakes()
    sample_pdf.save('demo_data/bert-processed-sample.pdf')

    # sample_pdf = PDFProcessor('demo_data/sample.pdf', ScorerType.GPT)
    # sample_pdf.score_paragraphs()
    # sample_pdf.highlight_mistakes()
    # sample_pdf.save('demo_data/gpt-processed-sample.pdf')
