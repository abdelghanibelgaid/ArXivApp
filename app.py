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
st.title("UM6P ArXiv Research Trends Analyzer")

st.markdown("""
Welcome to the **UM6P ArXiv Research Trends Analyzer**! Easily explore the latest advancements in your field by summarizing research papers directly from ArXiv. 
Choose your category of interest, define the date range, and receive concise summaries along with insights into emerging trends.
""")

# Set up OpenAI API key input
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
category = st.selectbox("Select a Research Category", list(categories.keys()))
date_from = st.date_input("Start Date", datetime(2024, 1, 1))
date_to = st.date_input("End Date", datetime(2024, 10, 1))

# Scrape ArXiv papers when the user clicks the button
if st.button("Analyze Trends"):
    if openai_api_key:
        # Scraping ArXiv papers
        with st.spinner("Fetching the latest research papers..."):
            scraper = arxivscraper.Scraper(category=categories[category], date_from=date_from.strftime('%Y-%m-%d'), date_until=date_to.strftime('%Y-%m-%d'))
            papers = scraper.scrape()
            df = pd.DataFrame(papers, columns=['id', 'title', 'categories', 'authors', 'abstract', 'pdf_url', 'doi', 'url', 'created', 'updated'])

        st.success(f"Successfully retrieved {len(df)} papers!")

        if not df.empty:
            st.write("Here are the titles and abstracts of the papers found:")
            st.write(df[['title', 'abstract']])

            # ===========================
            # Generate Summaries per Sub-Category
            # ===========================

            openai.api_key = openai_api_key

            def generate_summary(abstracts):
                """Use OpenAI API to generate a single summary for a list of abstracts."""
                combined_abstracts = "\n\n".join(abstracts)
                prompt = f"Summarize the following research abstracts and highlight the key trends and advancements in the field:\n\n{combined_abstracts}\n\nSummary:"
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

            # Group papers by sub-categories and generate summaries
            df['sub_fields'] = df['categories'].apply(lambda x: x.split())
            exploded_df = df.explode('sub_fields')

            st.subheader(f"Summarized Insights for {category}")
            summaries = {}

            for sub_field in exploded_df['sub_fields'].unique():
                abstracts = exploded_df[exploded_df['sub_fields'] == sub_field]['abstract'].tolist()
                summaries[sub_field] = generate_summary(abstracts)

            # Display the summaries for each sub-category
            for sub_field, summary in summaries.items():
                st.subheader(f"Key Findings for {sub_field}")
                st.write(summary)

            # ===========================
            # Analyze Sub-Field Trends
            # ===========================
            st.subheader("Research Trends Across Sub-Fields")
            
            # Plot top sub-fields by frequency
            top_sub_fields = exploded_df['sub_fields'].value_counts().head(10)
            plt.figure(figsize=(10, 6))
            sns.barplot(x=top_sub_fields.values, y=top_sub_fields.index, palette='Blues_r')
            plt.title("Top 10 Sub-Fields by Frequency of Papers")
            plt.xlabel("Number of Papers")
            plt.ylabel("Sub-Field")
            st.pyplot(plt)

            # Display emerging trends and fields
            st.subheader("Emerging Fields and Opportunities")
            st.markdown("Below are the emerging fields based on the number of recent publications:")
            emerging_fields = exploded_df.groupby('sub_fields')['title'].count().reset_index().sort_values(by='title', ascending=False)
            st.write(emerging_fields.head(10))
    else:
        st.error("Please provide your OpenAI API key to proceed.")
