import streamlit as st
from comparison import compare_responses_via_api
from lib import *
from brain import get_index_for_pdf
from document_handler import initialize_session_state, handle_file_uploads
import pandas as pd
import os
from openai import OpenAI

# Set the API key as an environment variable
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Initialize the OpenAI client
client = OpenAI()

# Set the title for the Streamlit app
st.title("RAG-OCR Enhanced Chatbot")

prompt_template = """
    You are a helpful Assistant who answers to users questions based on multiple contexts given to you.

    Keep answer correct and to the point. Try to answer from context first.

    Try answering in proper order and proper indentation do not output in paragraph much, use bullet points and all.
    
    If you did not get anything related to query Print "Did not get any Related Information", 
    
    The evidence are the context of the pdf extract with metadata. 
    
    Only give response and do not mention source or page or filename. If user asks for it, then tell.
        
    The PDF content is:
    {pdf_extract}
"""

def initialize_prompt():
    if 'prompt' not in st.session_state:
        st.session_state.prompt = [{"role": "system", "content": "none"}]

def display_chat_history():
    for message in st.session_state.prompt:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.write(message["content"])

def perform_similarity_search(vectordbs, question):
    pdf_extracts = []
    for vectordb in vectordbs:
        search_results = vectordb.similarity_search(question, k=10)
        pdf_extracts.append("\n".join([result.page_content for result in search_results]))
    return pdf_extracts

def generate_initial_responses(pdf_extracts, question, document_names):
    combined_responses = []
    for extract, doc_name in zip(pdf_extracts, document_names):
        individual_prompt = prompt_template.format(pdf_extract=extract)
        try:
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": individual_prompt},
                    {"role": "user", "content": question}
                ],
                temperature=0.2
            )
            response = completion.choices[0].message.content
            combined_responses.append((doc_name, response.strip()))
        except Exception as e:
            st.error(f"An error occurred: {e}")
    return combined_responses

def refine_combined_response(combined_response_text, question):
    formatted_prompt = f"""
    I have gathered the following information from multiple sources in response to the question: "{question}"

    {combined_response_text}

    Please refine and improve the answer by making it more coherent and comprehensive and please do not repeat anything and output it in proper order.
    """
    try:
        refinement = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": formatted_prompt}],
            temperature=0
        )
        final_response = refinement.choices[0].message.content
    except Exception as e:
        st.error(f"An error occurred during refinement: {e}")
        final_response = ""
    return final_response.strip()

def handle_user_input(question, display_mode):
    vectordbs = st.session_state.get("vectordbs", None)
    document_names = st.session_state.get("document_names", [])
    if "responses" not in st.session_state:
        st.session_state["responses"] = {}

    if not vectordbs:
        with st.chat_message("assistant"):
            st.write("You need to provide a PDF")
            st.stop()

    pdf_extracts = perform_similarity_search(vectordbs, question)
    new_responses = generate_initial_responses(pdf_extracts, question, document_names)

    combined_response_text = "\n\n".join([response for _, response in new_responses])
    final_result = refine_combined_response(combined_response_text, question)

    # Store individual and combined responses in session state
    response_entry = {
        "question": question,
        "individual_responses": new_responses,
        "combined_response": final_result,
        "display_mode": display_mode
    }
    st.session_state["responses"][question] = response_entry

    # Display responses based on their associated display mode
    for q, entry in st.session_state["responses"].items():
        with st.chat_message("user"):
            st.write(q)

        if entry["display_mode"] == "Document-wise":
            for doc_name, response in entry["individual_responses"]:
                with st.chat_message("assistant"):
                    st.write(f"Response from the Document \"{doc_name}\" :")
                    st.write(response)
        elif entry["display_mode"] == "Summarized":
            with st.chat_message("assistant"):
                st.write(entry["combined_response"])
        elif entry["display_mode"] == "Comparison":
            comparison_result = compare_responses_via_api(question, vectordbs, document_names)
            with st.chat_message("assistant"):
                pd.set_option('display.max_colwidth', None)
                st.table(comparison_result)


# Initialize session state variables if not already done
if 'prompt' not in st.session_state:
    st.session_state.prompt = []

if 'combined_responses' not in st.session_state:
    st.session_state.combined_responses = []

if 'display_mode' not in st.session_state:
    st.session_state.display_mode = "Document-wise"

def main():
    initialize_session_state()
    initialize_prompt()
    handle_file_uploads()

    # Add radio buttons for display mode selection
    display_mode = st.radio("Select Display Mode:", ["Document-wise", "Summarized", "Comparison"], index=0)
    st.session_state.display_mode = display_mode

    display_chat_history()

    question = st.chat_input("Ask anything")
    if question:
        handle_user_input(question, display_mode)

if __name__ == "__main__":
    main()

