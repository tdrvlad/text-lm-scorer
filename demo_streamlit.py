import streamlit as st
import os
import base64
from utils.pdf_processor import PDFProcessor
import shutil

st.title('PDF Analysis')

uploaded_file = st.file_uploader("Please select the PDF file you want to analyse.", type=['pdf'])

if uploaded_file:
    # 0. remove and create tmp dir
    temp_path = 'tmp'
    if os.path.exists(temp_path) and os.path.isdir(temp_path):
        shutil.rmtree(temp_path)
    os.mkdir(temp_path)

    # 1. save file under tmp and get file path - to give to scorer
    tmp_file_path = os.path.join(temp_path, uploaded_file.name)
    with open(tmp_file_path, "wb") as pdf_file:
        pdf_file.write(uploaded_file.getbuffer())

    # 2. create processed file path
    processed_file_path = os.path.join(temp_path, f'processed-{uploaded_file.name}')

    # 3. process file and save
    with st.spinner('Wait for it...'):
        sample_pdf = PDFProcessor(tmp_file_path)
        sample_pdf.score_sentences()
        sample_pdf.highlight_mistakes()
        sample_pdf.save(processed_file_path)

    # 4. display Processed File when ready
    with open(processed_file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)
