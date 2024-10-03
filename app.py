import streamlit as st
import pandas as pd
import arxivscraper
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from transformers import pipeline
import openai

# ===========================
# Streamlit App Layout
# ===========================
st.title("ArXiv Paper Summarizer and Trends Analyzer")

# Option for OpenAI API key or Open-Source LLM
llm_choice = st.radio("Choose the LLM for summarization:", ["OpenAI API", "Open-source LLM"])

# Initialize Open-Source LLM if selected
if llm_choice == "Open-source LLM":
    # Load the summarization pipeline from Hugging Face's transformers (e.g., GPT-J or GPT-Neo)
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    st.write("Using an open-source LLM (DistilBART model from Hugging Face).")

# If OpenAI is selected, the user must provide the API key
if llm_choice == "OpenAI API":
    openai_api_key = st.text_input("Enter your OpenAI API key:", type="password")

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
        # Generate Summaries per Sub-Field
        # ===========================

        def generate_summary_openai(abstracts):
            """Use OpenAI API to generate a summary for a list of abstracts."""
            combined_abstracts = "\n\n".join(abstracts)
            prompt = f"Summarize the following research abstracts into a brief summary focusing on the latest trends and advancements:\n\n{combined_abstracts}\n\nSummary:"
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=400,
                    temperature=0.5,
                )
                return response.choices[0].message["content"].strip()
            except Exception as e:
                return f"Error: {str(e)}"

        def generate_summary_opensource(abstracts):
            """Use open-source LLM (e.g., GPT-J or GPT-Neo) to generate a summary."""
            combined_abstracts = " ".join(abstracts)
            try:
                summary = summarizer(combined_abstracts, max_length=200, min_length=30, do_sample=False)
                return summary[0]["summary_text"]
            except Exception as e:
                return f"Error: {str(e)}"

        # Group papers by sub-categories and generate summaries
        df['sub_fields'] = df['categories'].apply(lambda x: x.split())
        exploded_df = df.explode('sub_fields')

        summaries = {}

        for sub_field in exploded_df['sub_fields'].unique():
            abstracts = exploded_df[exploded_df['sub_fields'] == sub_field]['abstract'].tolist()
            
            if llm_choice == "OpenAI API" and openai_api_key:
                openai.api_key = openai_api_key
                summaries[sub_field] = generate_summary_openai(abstracts)
            elif llm_choice == "Open-source LLM":
                summaries[sub_field] = generate_summary_opensource(abstracts)
        
        # Display the summaries for each sub-category
        for sub_field, summary in summaries.items():
            st.subheader(f"Summary for {sub_field}")
            st.write(summary)

        # ===========================
        # Analyze Sub-Field Trends
        # ===========================
        st.subheader("Trends in Sub-Fields")
        
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
