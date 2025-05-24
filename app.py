import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time
import urllib.parse

# --- Selenium scraper for karkidi.com ---
def scrape_karkidi_jobs_selenium(job_title):
    base_url = "https://www.karkidi.com/Find-Jobs"
    params = {'keyword': job_title}
    search_url = f"{base_url}?{urllib.parse.urlencode(params)}"

    options = Options()
    options.headless = True
    # Optional: to prevent detection & speed up
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(options=options)

    driver.get(search_url)
    time.sleep(5)  # wait for page and JS to load

    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')
    driver.quit()

    job_listings = []

    # The jobs are inside <a class="job-listing-link"> tags (update if class changes)
    job_cards = soup.find_all('a', class_='job-listing-link')

    for card in job_cards:
        title_tag = card.find('h2', class_='job-title')
        company_tag = card.find('div', class_='company-name')
        skills_tag = card.find('div', class_='job-tags')  # Could hold skills/tags

        title = title_tag.get_text(strip=True) if title_tag else 'N/A'
        company = company_tag.get_text(strip=True) if company_tag else 'N/A'
        if skills_tag:
            skills_raw = skills_tag.get_text(separator=',', strip=True)
            skills_list = [skill.strip() for skill in skills_raw.split(',')]
        else:
            skills_list = []

        job_listings.append({'title': title, 'company': company, 'skills': skills_list})

    return pd.DataFrame(job_listings)


# --- Load models and data ---
try:
    hierarchical_model = joblib.load('hierarchical_model.pkl')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    original_train_tfidf_matrix = scipy.sparse.load_npz('original_train_tfidf_matrix.npz')
    initial_training_df = pd.read_csv('karkidi_jobs_initial_processed.csv')
    initial_training_df['skills'] = initial_training_df['skills'].apply(eval)
    st.sidebar.success("Models and data loaded successfully.")
except Exception as e:
    st.sidebar.error(f"Model loading error: {e}")
    hierarchical_model = None
    tfidf_vectorizer = None
    original_train_tfidf_matrix = None
    initial_training_df = pd.DataFrame()

# --- Streamlit Interface ---
st.title("Karkidi Job Categorizer with Selenium Scraper")
st.markdown("Powered by Hierarchical Clustering & TF-IDF")

job_input = st.text_input("Enter job title to search:", "")

if hierarchical_model and tfidf_vectorizer and original_train_tfidf_matrix is not None:
    if st.button("Scrape & Classify Jobs"):
        if not job_input.strip():
            st.warning("Please enter a job title to search.")
        else:
            with st.spinner("Scraping and classifying..."):
                new_jobs = scrape_karkidi_jobs_selenium(job_input.strip())
                if not new_jobs.empty:
                    classified = classify_new_jobs(new_jobs, tfidf_vectorizer, hierarchical_model, original_train_tfidf_matrix)
                    st.session_state['classified_jobs'] = classified
                    st.success(f"Classified {len(classified)} new jobs for '{job_input.strip()}'.")
                    st.dataframe(classified[['title', 'company', 'skills', 'cluster']])
                else:
                    st.warning(f"No jobs found for '{job_input.strip()}'. Please try another keyword.")

    if 'classified_jobs' in st.session_state:
        st.subheader("Explore Classified Jobs")
        st.dataframe(st.session_state['classified_jobs'][['title', 'company', 'skills', 'cluster']])

        st.subheader("Cluster Insights")
        if not initial_training_df.empty:
            cluster_info = get_top_skills_per_cluster(hierarchical_model, tfidf_vectorizer, initial_training_df)
            for cluster_id, info in cluster_info.items():
                st.write(f"**Cluster {cluster_id}**")
                st.markdown(f"Sample Jobs: {', '.join(info['sample_jobs'])}")
                st.markdown(f"Top Skills: {', '.join(info['top_skills'])}")

else:
    st.warning("Model or data not loaded. Train the model and save files: `hierarchical_model.pkl`, `tfidf_vectorizer.pkl`, `original_train_tfidf_matrix.npz`, and `karkidi_jobs_initial_processed.csv`.")
