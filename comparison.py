from openai import OpenAI
import pandas as pd
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed

# Initialize the OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def get_completion(prompt, model="gpt-3.5-turbo", temperature=0):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    return response.choices[0].message.content.strip()

def identify_key_points(question):
    prompt = f"""
    Based on the question "{question}", generate a list of 5-7 key financial metrics or points that would be most relevant for comparison across financial documents.
    Focus on important financial indicators, ratios, or performance measures.
    Provide the list in a comma-separated format.
    For example, if comparing financials, key points might include:
    Revenue, Net Income, EBITDA, Profit Margin, Return on Equity (ROE), Debt-to-Equity Ratio, Free Cash Flow
    Ensure the key points are:
    1. Relevant to the specific question asked
    2. Commonly used in financial analysis
    3. Likely to provide meaningful insights when compared across documents
    4. Diverse enough to cover different aspects of financial performance
    List of key points:
    """
    response = get_completion(prompt, temperature=0.3)
    return [point.strip() for point in response.split(',')]

def perform_similarity_search(vectordbs, question, key_points):
    pdf_extracts = []
    for vectordb in vectordbs:
        search_results = vectordb.similarity_search(question, k=10)
        pdf_extract = "\n".join([result.page_content for result in search_results])
        
        # Add key points to the pdf_extract
        pdf_extract += f"\n\nKey Points for Analysis: {', '.join(key_points)}"
        
        pdf_extracts.append(pdf_extract)
    return pdf_extracts

def extract_key_point_info(response, key_point):
    prompt = f"""
    Extract and summarize information related to "{key_point}" from the following text:
    {response}
    Provide a concise summary that:
    1. Focuses on relevant facts, figures, and brief details
    2. Is no longer than 2-3 sentences
    3. Highlights any quantitative data if available
    4. Maintains objectivity and accuracy
    If no relevant information is found, respond with "No specific information available."
    """
    return get_completion(prompt, temperature=0.1)

def compare_responses_via_api(question, vectordbs, document_names):
    key_points = identify_key_points(question)
    pdf_extracts = perform_similarity_search(vectordbs, question, key_points)
    
    comparison_data = {"Document": document_names}
    
    def process_key_point(key_point):
        return [extract_key_point_info(extract, key_point) for extract in pdf_extracts]
    
    with ThreadPoolExecutor() as executor:
        futures = {key_point: executor.submit(process_key_point, key_point) for key_point in key_points}
        for key_point, future in futures.items():
            comparison_data[key_point] = future.result()
    
    df = pd.DataFrame(comparison_data)
    
    return df
