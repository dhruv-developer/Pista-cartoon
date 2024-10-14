import os
import streamlit as st
import pdfplumber
from dotenv import load_dotenv
import re
from rag_model import generate_summary  # Assuming your RAG model exists here
from cartoon_script_generator import generate_cartoon_script

# Load the API Key from .env file
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Function to clean and split text into chunks for better summary generation
def split_text_into_chunks(text, chunk_size=1000):
    """
    Cleans and splits the input text into manageable chunks for processing.
    """
    # Clean the text by removing unnecessary whitespace/newlines
    text = re.sub(r'\s+', ' ', text).strip()

    # Split the text into sentences (assumes sentences end with '. ')
    sentences = re.split(r'(?<=\.) ', text)
    
    # Combine sentences into chunks of approximately chunk_size
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # If adding the sentence exceeds chunk_size, save the current chunk
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def call_llama_groq_api(text_chunk, api_key):
    """
    Calls the real RAG model to generate a summary for the provided text chunk.
    
    Args:
        text_chunk (str): The chunk of text to be summarized.
        api_key (str): The API key if needed for external APIs (if not needed, can be ignored).
    
    Returns:
        str: The generated summary for the text chunk.
    """
    # Remove the mock warning
    st.info("Generating summary with the RAG model...")

    # Call the real RAG-based summary generation function
    try:
        # Use the real generate_summary function from your rag_model module
        summary = generate_summary(text_chunk)
        
        # Return the generated summary
        return summary
    
    except Exception as e:
        # Handle any exceptions (like API issues or model errors)
        st.error(f"Error during summary generation: {e}")
        return "Error generating summary. Please try again."
    
# Function to generate a detailed summary by processing PDF content in chunks
def generate_detailed_summary(text, api_key):
    """
    Splits the text into chunks and generates a summary for each chunk using an API.
    """
    chunks = split_text_into_chunks(text)
    detailed_summary = ""

    # For each chunk, call the external summary API
    for chunk in chunks:
        summary_chunk = call_llama_groq_api(chunk, api_key)
        detailed_summary += summary_chunk + "\n\n"

    return detailed_summary.strip()

# Streamlit App Interface
st.title("AI-Powered PDF Summary and Cartoon Script Generator")
st.write("Upload a PDF file and let AI generate a detailed summary along with a cartoon script.")

# PDF Upload
uploaded_pdf = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_pdf:
    # Extract text from the PDF using pdfplumber
    with pdfplumber.open(uploaded_pdf) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""  # Extract text or return empty if not available

    # Ensure text was extracted
    if not text.strip():
        st.error("Failed to extract text from the PDF. Please check if it's scanned or contains valid text.")
    else:
        # Generate detailed summary from the extracted text
        st.write("Generating detailed summary...")
        detailed_summary = generate_detailed_summary(text, GROQ_API_KEY)

        # Check if summary generation was successful
        if detailed_summary:
            st.success("Detailed summary generated successfully!")

            # Display Detailed Summary
            st.subheader("Detailed Summary:")
            st.write(detailed_summary)

            # Generate Cartoon Script from the detailed summary
            st.write("Generating cartoon script from the summary...")
            cartoon_script = generate_cartoon_script(detailed_summary)

            # Display Cartoon Script
            st.subheader("Cartoon Script:")
            st.json(cartoon_script)

            # Option to download the cartoon script as a JSON file
            st.download_button(
                label="Download Cartoon Script as JSON",
                data=str(cartoon_script),
                file_name="cartoon_script.json",
                mime="application/json"
            )
        else:
            st.error("Failed to generate a detailed summary. Please try again later.")
