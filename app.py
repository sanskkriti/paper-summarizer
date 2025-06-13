import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import streamlit as st
import fitz  # PyMuPDF
from transformers import pipeline

# Load the summarization pipeline (uses PyTorch backend)
summarizer = pipeline(
    "summarization", 
    model="facebook/bart-large-cnn", 
    framework="pt", 
    device=-1
)

def extract_text_from_pdf(file):
    """Extract text from each page of the uploaded PDF."""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        return "".join(page.get_text() for page in doc)

def get_summary(text):
    """Generate a more detailed summary from extracted text."""
    input_text = text[:3000]  # Still truncate to avoid model token limits
    result = summarizer(input_text, max_length=400, min_length=200, do_sample=False)
    return result[0]["summary_text"]

# Streamlit UI
st.set_page_config(page_title="Paper Summarizer", page_icon="🧠")
st.title("📄 Research Paper Auto-Summarizer by Sanskriti")
st.markdown("Upload a PDF research paper and generate a quick AI-based summary.")

pdf_file = st.file_uploader("Upload your PDF file", type="pdf")

if pdf_file is not None:
    with st.spinner("Reading your PDF..."):
        paper_text = extract_text_from_pdf(pdf_file)
    st.success("Text successfully extracted!")

    if st.button("Generate Summary"):
        with st.spinner("Summarizing..."):
            summary_output = get_summary(paper_text)
        st.subheader("🔍 Summary")
        st.write(summary_output)

