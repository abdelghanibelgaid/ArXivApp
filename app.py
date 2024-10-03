# Streamlit app script: `arxiv_summarizer_app.py`

import streamlit as st
import pandas as pd
import openai
import arxivscraper
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ===========================
# Streamlit App Layout
# ===========================
st.title("ArXiv Paper Summarizer and Trends Analyzer")

# Option for OpenAI API key or Open-Source LLM
llm_choice = st.radio("Choose the LLM for summarization:", ["OpenAI API", "Open-source LLM"])

if llm_choice == "OpenAI API":
    openai_api_key = st.text_input("Enter your OpenAI API key:", type="password")
else:
    st.write("You have selected an open-source LLM (e.g., GPT-J or GPT-Neo). This feature will run locally.")

# Define category options
categories = {
    'Computer Science': 'cs',
    'Economics': 'econ',
    'Electrical Engineering and Systems Science': 'eess',
    'Mathematics': 'math',
    'Physics': 'physics',
    'Astrophysics': 'physics:astro-ph',
    'Condensed Matter': 'physics:cond-mat',
    'General Relativity and Quantum Cosmology': 'physics:gr-qc',
    'High Energy Physics - Experiment': 'physics:hep-ex',
    'High Energy Physics - Lattice': 'physics:hep-lat',
    'High Energy Physics - Phenomenology': 'physics:hep-ph',
    'High Energy Physics - Theory': 'physics:hep-th',
    'Mathematical Physics': 'physics:math-ph',
    'Nonlinear Sciences': 'physics:nlin',
    'Nuclear Experiment': 'physics:nucl-ex',
    'Nuclear Theory': 'physics:nucl-th',
    'Quantum Physics': 'physics:quant-ph',
    'Quantitative Biology': 'q-bio',
    'Quantitative Finance': 'q-fin',
    'Statistics': 'stat'
}

# Create input fields
category = st.selectbox("Select a Category", list(categories.keys()))
date_from = st.date_input("Date From", datetime(2024, 1, 1))
date_to = st.date_input("Date To", datetime(2024, 10, 1))

# Scrape ArXiv papers when the user clicks the button
if st.button("Scrape and Summarize Papers"):
    with st.spinner("Scraping ArXiv papers..."):
        scraper = arxivscraper.Scraper(category=categories[category], date_from=date_from.strftime('%Y-%m-%d'), date_until=date_to.strftime('%Y-%m-%d'))
        papers = scraper.scrape()
        df = pd.DataFrame(papers, columns=['id', 'title', 'categories', 'authors', 'abstract', 'pdf_url', 'doi', 'url', 'created', 'updated'])

    st.success(f"Scraped {len(df)} papers from ArXiv.")

    if not df.empty:
        st.write(df[['title', 'abstract']])

        # ===========================
        # Generate Summaries
        # ===========================

        if llm_choice == "OpenAI API":
            # Check if API key is provided
            if openai_api_key:
                openai.api_key = openai_api_key

                def generate_summary(text):
                    try:
                        response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            messages=[{"role": "user", "content": text}],
                            max_tokens=200,
                            temperature=0.5,
                        )
                        return response.choices[0].message["content"].strip()
                    except Exception as e:
                        return f"Error: {str(e)}"
            else:
                st.error("Please enter your OpenAI API key.")

        else:
            # Use open-source LLM like GPT-J/Neo (dummy function for now)
            def generate_summary(text):
                return "Open-source LLM (GPT-J/Neo) will summarize this text."

        df['summary'] = df['abstract'].apply(lambda abstract: generate_summary(f"Summarize this abstract: {abstract}"))

        # Display the summaries
        st.write(df[['title', 'summary']])

        # ===========================
        # Analyze Sub-Field Trends
        # ===========================
        st.subheader("Trends in Sub-Fields")
        
        df['sub_fields'] = df['categories'].apply(lambda x: x.split())
        exploded_df = df.explode('sub_fields')

        # Plot top sub-fields by frequency
        top_sub_fields = exploded_df['sub_fields'].value_counts().head(10)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_sub_fields.values, y=top_sub_fields.index, palette='coolwarm')
        plt.title("Top 10 Sub-Fields by Frequency")
        plt.xlabel("Frequency")
        plt.ylabel("Sub-Field")
        st.pyplot(plt)

        # Analyze the trends in sub-fields
        st.subheader("Emerging Fields and Trends")
        emerging_fields = exploded_df.groupby('sub_fields')['title'].count().reset_index().sort_values(by='title', ascending=False)
        st.write(emerging_fields.head(10))

# Run this with `streamlit run arxiv_summarizer_app.py`
