# NLP Business Case: Automated Customer Reviews

Build an end-to-end Natural Language Processing (NLP) pipeline to extract actionable marketing insights from over 41,000 Amazon product reviews.

[Task Descriptions and Project Instructions](https://github.com/ironhack-labs/project-nlp-business-case-automated-customers-reviews-v2)

## Project Results
In this project, we processed a raw dataset of 41,291 Amazon reviews and built a comprehensive AI pipeline. We explored and integrated three different NLP approaches:
- **Model 1 (Classification):** Used `distilbert-base-uncased-finetuned-sst-2-english` to perform Sentiment Analysis. We created a "Credibility Filter" to ensure our business insights are only based on reviews where human ratings match the AI sentiment.
- **Model 2 (Clustering):** Applied `TfidfVectorizer` (with n-grams) and `K-Means` to group over 100 messy product names into 5 clean, distinct Meta-Categories (e.g., Tablets, Smart Home, Batteries).
- **Model 3 (Generative AI Summarization):** Leveraged the `facebook/bart-large-cnn` transformer to mathematically extract the Top 3 products, main complaints, and worst products per category, generating executive summaries.

We deployed an interactive local web application using **Streamlit** to showcase the final AI-generated summaries and filtered insights for the Marketing Department.

## Repository Folders and Files

Here is a short description of the folders and files available in this repository.

### Documents
**Group / Individual NLP Project - Presentation Slides** Final Slides presentation (PPT) detailing the business case and technical choices.

### Notebooks & Scripts
- **main.ipynb**: The core Jupyter Notebook containing the full Data Science lifecycle: data cleaning, EDA, DistilBERT classification, K-Means clustering, and BART text generation.
- **app.py**: The Python script for the interactive Streamlit Web Dashboard, allowing users to filter categories and read AI summaries.
  
### Data
- **final_dashboard_data.csv**: The clean, processed, and credible dataset exported from the main notebook, used to power the Streamlit dashboard.

## Installation & Execution
Use **requirements.txt** to install the required packages to run the notebook and the web application. It is advised to use a virtual environment.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the interactive dashboard locally
streamlit run app.py