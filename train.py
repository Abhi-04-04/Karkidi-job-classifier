import pandas as pd
import numpy as np
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
import requests
from bs4 import BeautifulSoup
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import scipy.sparse  Needed for saving/loading sparse matrices
from sklearn.metrics import pairwise_distances 

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    def preprocess_text(text):
        doc = nlp(text)
        tokens = [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha]
        return ' '.join(tokens)
except OSError:
    print("SpaCy model 'en_core_web_sm' not found. Using simple preprocessing instead. Run: python -m spacy download en_core_web_sm")

    def simple_preprocess_text(text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return " ".join([word for word in text.split() if word not in ['a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were']]) # very basic stop word removal

    def preprocess_text(text):
        return simple_preprocess_text(text)


    # Fallback to a simpler text processing if spaCy isn't available
    def simple_preprocess_text(text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return " ".join([word for word in text.split() if word not in ['a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were']]) # very basic stop word removal

    def preprocess_text(text):
        return simple_preprocess_text(text)


# --- Utility Functions ---

def scrape_karkidi_jobs():
    """
    Scrapes job listings from karkidi.com or provides sample data if scraping fails.
    YOU MUST UPDATE THE base_url AND BeautifulSoup SELECTORS.
    """
    base_url = "https://www.karkidi.com/"
    job_listings = []

    try:
        response = requests.get(base_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

    
        job_cards = soup.find_all('div', class_='job-card') # Example: <div class="job-card">

        if not job_cards:
            print(f"Warning: No job cards found with current selectors on {base_url}. Using expanded sample data.")
            job_listings.extend([
                {'title': 'Sample Data Scientist', 'company': 'AI Labs', 'skills': ['python', 'machine learning', 'tensorflow', 'pytorch', 'data analysis'], 'job_description': 'Work with data to build ML models.'},
                {'title': 'Sample Web Developer', 'company': 'Code Innovators', 'skills': ['javascript', 'react', 'node.js', 'html', 'css', 'frontend'], 'job_description': 'Develop cutting-edge web applications.'},
                {'title': 'Sample HR Manager', 'company': 'People Solutions', 'skills': ['recruitment', 'hr operations', 'employee relations', 'talent acquisition', 'payroll'], 'job_description': 'Manage human resources operations.'},
                {'title': 'Sample Marketing Specialist', 'company': 'Brand Builders', 'skills': ['digital marketing', 'seo', 'content creation', 'social media', 'google analytics'], 'job_description': 'Plan and execute marketing campaigns.'},
                {'title': 'Sample Sales Executive', 'company': 'Growth Inc', 'skills': ['sales', 'crm', 'negotiation', 'client management', 'lead generation'], 'job_description': 'Drive sales and manage client relationships.'},
                {'title': 'Cloud Engineer', 'company': 'CloudPro', 'skills': ['aws', 'azure', 'devops', 'kubernetes', 'cloud security', 'terraform'], 'job_description': 'Design and implement cloud infrastructure.'},
                {'title': 'Cyber Security Analyst', 'company': 'SecureNet', 'skills': ['cybersecurity', 'splunk', 'firewall', 'linux', 'network security', 'incident response'], 'job_description': 'Protect systems from cyber threats.'},
                {'title': 'Business Analyst', 'company': 'Consulting Inc', 'skills': ['business analysis', 'sql', 'tableau', 'requirements gathering', 'data visualization', 'agile'], 'job_description': 'Analyze business needs and propose solutions.'},
                {'title': 'Mobile App Developer', 'company': 'AppCreators', 'skills': ['kotlin', 'swift', 'android', 'ios', 'mobile development', 'ui/ux'], 'job_description': 'Develop mobile applications for Android and iOS.'},
                {'title': 'DevOps Engineer', 'company': 'Automation Hub', 'skills': ['jenkins', 'ansible', 'docker', 'ci/cd', 'git', 'scripting'], 'job_description': 'Automate software delivery processes.'},
                {'title': 'Senior Data Analyst', 'company': 'Quant Insights', 'skills': ['sql', 'power bi', 'excel', 'data modeling', 'statistics', 'dashboarding'], 'job_description': 'Perform complex data analysis and reporting.'},
                {'title': 'Backend Developer', 'company': 'API Builders', 'skills': ['java', 'spring boot', 'rest api', 'microservices', 'database design', 'kotlin'], 'job_description': 'Build robust backend services and APIs.'},
                {'title': 'Frontend Developer', 'company': 'UI Masters', 'skills': ['html', 'css', 'javascript', 'vue.js', 'responsive design', 'web design'], 'job_description': 'Create intuitive and engaging user interfaces.'},
                {'title': 'Project Manager', 'company': 'Delivery Solutions', 'skills': ['project management', 'agile', 'scrum', 'leadership', 'stakeholder management', 'risk management'], 'job_description': 'Lead and manage software development projects.'},
                {'title': 'Content Writer', 'company': 'Wordsmiths', 'skills': ['content writing', 'copywriting', 'blogging', 'seo', 'editing', 'grammar'], 'job_description': 'Create engaging content for various platforms.'},
                {'title': 'Financial Analyst', 'company': 'Wealth Mgmt', 'skills': ['finance', 'forecasting', 'excel', 'valuation', 'financial modeling', 'investment analysis'], 'job_description': 'Analyze financial data and prepare reports.'},
                {'title': 'Network Administrator', 'company': 'NetServe', 'skills': ['networking', 'cisco', 'windows server', 'active directory', 'troubleshooting', 'router configuration'], 'job_description': 'Manage and maintain network infrastructure.'},
                {'title': 'Machine Learning Engineer', 'company': 'AI Innovate', 'skills': ['pytorch', 'nlp', 'computer vision', 'data science', 'deep learning', 'model deployment'], 'job_description': 'Build and deploy machine learning models.'},
                {'title': 'Product Manager', 'company': 'InnovateTech', 'skills': ['product management', 'roadmap', 'market research', 'ux', 'agile product management', 'strategy'], 'job_description': 'Define product vision and strategy.'},
                {'title': 'Full Stack Java Dev', 'company': 'Enterprise Solutions', 'skills': ['java', 'springboot', 'angular', 'rest', 'hibernate', 'typescript'], 'job_description': 'Develop end-to-end Java applications.'},
                {'title': 'IT Support Specialist', 'company': 'HelpDesk Pro', 'skills': ['troubleshooting', 'hardware', 'software', 'customer service', 'windows', 'microsoft office'], 'job_description': 'Provide technical support to users.'},
                {'title': 'Graphic Designer', 'company': 'Creative Edge', 'skills': ['adobe photoshop', 'illustrator', 'indesign', 'branding', 'layout design'], 'job_description': 'Create visual concepts for various media.'},
                {'title': 'Quality Assurance Engineer', 'company': 'Quality First', 'skills': ['qa testing', 'manual testing', 'automation testing', 'selenium', 'jira'], 'job_description': 'Ensure software quality through testing.'},
                {'title': 'Biotech Scientist', 'company': 'BioTech Labs', 'skills': ['biology', 'chemistry', 'research', 'lab techniques', 'data analysis', 'genomics'], 'job_description': 'Conduct scientific research in biotechnology.'},
                {'title': 'Legal Counsel', 'company': 'Law Associates', 'skills': ['legal research', 'contract drafting', 'litigation', 'corporate law', 'compliance'], 'job_description': 'Provide legal advice and representation.'},
            ])
        else:
            for card in job_cards:
                title_tag = card.find('h2', class_='job-title') # Example class
                company_tag = card.find('p', class_='company-name') # Example class
                skills_tag = card.find('div', class_='job-skills') # Example class, assuming skills are grouped
                desc_tag = card.find('div', class_='job-description') # Example class for description

                title = title_tag.get_text(strip=True) if title_tag else 'N/A'
                company = company_tag.get_text(strip=True) if company_tag else 'N/A'
                skills_raw = skills_tag.get_text(separator=',', strip=True) if skills_tag else ''
                job_description = desc_tag.get_text(strip=True) if desc_tag else 'N/A'

                skills_list = [skill.strip() for skill in skills_raw.split(',')]

                job_listings.append({
                    'title': title,
                    'company': company,
                    'skills': skills_list,
                    'job_description': job_description
                })
    except requests.exceptions.RequestException as e:
        print(f"Error scraping {base_url}: {e}. Falling back to sample data.")
        job_listings.extend([
            {'title': 'Network Error Job 1', 'company': 'Offline Corp', 'skills': ['network', 'troubleshooting', 'connectivity'], 'job_description': 'Handle network issues.'},
            {'title': 'Network Error Job 2', 'company': 'Disconnected LLC', 'skills': ['security', 'VPN', 'firewall'], 'job_description': 'Ensure system security.'},
            {'title': 'Network Error Job 3', 'company': 'Lost Signal Inc', 'skills': ['routers', 'switches', 'WAN'], 'job_description': 'Manage network devices.'},
            {'title': 'Network Error Job 4', 'company': 'Bad Connection Co', 'skills': ['dns', 'dhcp', 'tcp/ip'], 'job_description': 'Configure network protocols.'}
        ])
    except Exception as e:
        print(f"An unexpected error occurred during scraping: {e}. Falling back to sample data.")
        job_listings.extend([
            {'title': 'Unexpected Error Job 1', 'company': 'Buggy Systems', 'skills': ['debug', 'testing', 'software qa'], 'job_description': 'Test and debug software.'},
            {'title': 'Unexpected Error Job 2', 'company': 'Glitch Solutions', 'skills': ['analysis', 'system audit', 'process improvement'], 'job_description': 'Analyze system performance.'},
            {'title': 'Unexpected Error Job 3', 'company': 'Code Failures Ltd', 'skills': ['error handling', 'logging', 'bug reporting'], 'job_description': 'Report and fix code errors.'},
            {'title': 'Unexpected Error Job 4', 'company': 'Crash Corp', 'skills': ['stability testing', 'performance testing', 'fault tolerance'], 'job_description': 'Ensure system stability.'}
        ])

    return pd.DataFrame(job_listings)

def preprocess_skills(df, skills_column='skills'):
    """
    Cleans and converts list of skills into a single string for TF-IDF vectorization.
    Combines with job_description and title for richer features.
    """
    df['skills_text'] = df[skills_column].apply(lambda x: ' '.join(x).lower() if isinstance(x, list) else str(x).lower())
    df['skills_text'] = df['skills_text'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))
    df['skills_text'] = df['skills_text'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())

    # Combine with job description and title for comprehensive processing
    df['combined_processed_text'] = df['job_description'].fillna('') + " " + \
                                    df['skills_text'].fillna('') + " " + \
                                    df['title'].fillna('')

    df['combined_processed_text'] = df['combined_processed_text'].apply(preprocess_text)
    return df

def vectorize_skills(df, column='combined_processed_text'): # Changed to use combined_processed_text
    """
    Converts skill text (or combined text) into TF-IDF vectors.
    """
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', min_df=5, max_df=0.85)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df[column])
    return tfidf_matrix, tfidf_vectorizer

def train_hierarchical_clustering_model(tfidf_matrix, n_clusters=None, distance_threshold=None, linkage_method='ward'):
    """
    Trains an AgglomerativeClustering (Hierarchical Clustering) model.
    """
    if n_clusters is None and distance_threshold is None:
        raise ValueError("Either n_clusters or distance_threshold must be provided.")

    if n_clusters is not None and distance_threshold is not None:
        print("Warning: Both n_clusters and distance_threshold provided. n_clusters will be used.")
        distance_threshold = None

    if n_clusters:
        print(f"Training Hierarchical Clustering with {n_clusters} clusters...")
        hierarchical_model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    else:
        print(f"Training Hierarchical Clustering with distance_threshold={distance_threshold}...")
        hierarchical_model = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold, linkage=linkage_method)

    if n_clusters and tfidf_matrix.shape[0] < n_clusters:
        print(f"Warning: Cannot create {n_clusters} clusters with only {tfidf_matrix.shape[0]} samples. Adjusting n_clusters to {tfidf_matrix.shape[0]-1 if tfidf_matrix.shape[0] > 1 else 1}.")
        hierarchical_model = AgglomerativeClustering(n_clusters=tfidf_matrix.shape[0]-1 if tfidf_matrix.shape[0] > 1 else 1, linkage=linkage_method)


    hierarchical_model.fit(tfidf_matrix.toarray())
    return hierarchical_model



def get_top_skills_per_cluster_hierarchical(hierarchical_model, tfidf_vectorizer, df, num_words=10):
    """
    Analyzes and prints the top skills for each cluster identified by Hierarchical Clustering.
    """
    print("\nTop skills per cluster:")
    labels = hierarchical_model.labels_
    n_clusters = hierarchical_model.n_clusters_
    terms = tfidf_vectorizer.get_feature_names_out()

    for i in range(n_clusters):
        print(f"Cluster {i}:")
        cluster_indices = np.where(labels == i)[0]
        if len(cluster_indices) == 0:
            print("   No jobs in this cluster.")
            continue

        cluster_jobs_df = df.iloc[cluster_indices]
        print(f"   Sample Jobs: {cluster_jobs_df['title'].head().tolist()}")

        temp_tfidf_matrix_for_centroid = tfidf_vectorizer.transform(cluster_jobs_df['combined_processed_text'])

        if temp_tfidf_matrix_for_centroid.shape[0] > 0:
            cluster_centroid = temp_tfidf_matrix_for_centroid.mean(axis=0)
            top_skill_indices = np.asarray(cluster_centroid).flatten().argsort()[::-1][:num_words]
            print("   Top skills:")
            for ind in top_skill_indices:
                print(f'   - {terms[ind]}')
        else:
            print("   No dominant skills (cluster empty or issue).")
        print('\n')

def save_model(model, vectorizer, original_train_tfidf_matrix, model_path='hierarchical_model.pkl', vectorizer_path='tfidf_vectorizer.pkl', matrix_path='original_train_tfidf_matrix.npz'):
    """Saves the trained model, TF-IDF vectorizer, and the original training TF-IDF matrix."""
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    scipy.sparse.save_npz(matrix_path, original_train_tfidf_matrix) # Save the training matrix
    print(f"Model saved to {model_path}")
    print(f"Vectorizer saved to {vectorizer_path}")
    print(f"Original training TF-IDF matrix saved to {matrix_path}")

def load_model(model_path='hierarchical_model.pkl', vectorizer_path='tfidf_vectorizer.pkl', matrix_path='original_train_tfidf_matrix.npz'):
    """Loads a trained model, TF-IDF vectorizer, and the original training TF-IDF matrix."""
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    original_train_tfidf_matrix = scipy.sparse.load_npz(matrix_path) # Load the training matrix
    print(f"Model loaded from {model_path}")
    print(f"Vectorizer loaded from {vectorizer_path}")
    print(f"Original training TF-IDF matrix loaded from {matrix_path}")
    return model, vectorizer, original_train_tfidf_matrix

def classify_new_jobs(new_job_df, vectorizer, model, original_train_tfidf_matrix):
    """
    Classifies new job postings into existing clusters using centroid-based assignment.
    """
    print("Classifying new jobs...")
    new_job_df_processed = preprocess_skills(new_job_df.copy())
    new_skills_tfidf_matrix = vectorizer.transform(new_job_df_processed['combined_processed_text'])
    new_skills_tfidf_dense = new_skills_tfidf_matrix.toarray()

    # Get the cluster labels from the fitted model (these are for the data the model was trained on)
    original_training_labels = model.labels_

    # Calculate cluster centroids from the original training TF-IDF matrix
    unique_clusters = np.unique(original_training_labels)
    cluster_centroids = []
    for cluster_id in unique_clusters:
        indices_in_cluster = np.where(original_training_labels == cluster_id)[0]
        if scipy.sparse.issparse(original_train_tfidf_matrix):
            centroid = original_train_tfidf_matrix[indices_in_cluster].mean(axis=0)
            centroid = np.asarray(centroid).flatten() # Ensure 1D array
        else:
            centroid = np.mean(original_train_tfidf_matrix[indices_in_cluster], axis=0)
        cluster_centroids.append(centroid)

    cluster_centroids = np.array(cluster_centroids)
    print(f"Calculated {len(cluster_centroids)} cluster centroids from training data.")

    # Calculate distances from each new job to all cluster centroids
    distances = pairwise_distances(new_skills_tfidf_dense, cluster_centroids, metric='cosine')

    # Assign new jobs to the cluster with the closest centroid
    assigned_clusters = np.argmin(distances, axis=1)

    new_job_df_processed['cluster'] = assigned_clusters
    print("New jobs classified.")
    return new_job_df_processed

# --- User Preference and Notification Functions ---
user_preferences = {
    'user1': {'preferred_clusters': [0, 2], 'email': 'user1@example.com'}, # Replace with actual cluster IDs after training
    'user2': {'preferred_clusters': [1], 'email': 'user2@example.com'}
}

def send_email_notification(recipient_email, job_title, company, cluster_id):
    """
    Sends an email notification.
    IMPORTANT: Configure your email sender_email and sender_password securely.
    For Gmail, you will need to generate an App Password.
    """
    sender_email = "your_email@example.com" # <--- CHANGE THIS to your actual sending email
    sender_password = "your_email_password" # <--- CHANGE THIS to your actual email password / app password

    subject = f"New Job Alert: {job_title} at {company} (Category: {cluster_id})"
    body = f"""
    Hello,

    A new job matching your interests has been posted!

    Job Title: {job_title}
    Company: {company}
    Category: {cluster_id}

    Check it out on karkidi.com!

    Best regards,
    Your Job Monitoring System
    """

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587) # Adjust for your email provider
        server.starttls() # Enable TLS
        server.login(sender_email, sender_password)
        text = msg.as_string()
        server.sendmail(sender_email, recipient_email, text)
        print(f"Notification sent to {recipient_email} for job: {job_title}")
    except Exception as e:
        print(f"Failed to send email to {recipient_email}: {e}. Check email credentials and SMTP settings.")
    finally:
        try:
            if 'server' in locals() and server is not None:
                server.quit()
        except Exception as e:
            print(f"Error quitting SMTP server: {e}")


def check_and_notify(new_jobs_df, user_preferences):
    """
    Checks if new jobs match user preferences and sends notifications.
    """
    if new_jobs_df.empty:
        print("No new jobs to check for notifications.")
        return

    for index, job in new_jobs_df.iterrows():
        job_cluster = job['cluster']
        for user_id, prefs in user_preferences.items():
            if job_cluster in prefs['preferred_clusters']:
                print(f"Match found for {user_id}: Job '{job['title']}' in cluster {job_cluster}. Sending notification...")
                send_email_notification(prefs['email'], job['title'], job['company'], job_cluster)


# --- Main Execution Block ---
if __name__ == '__main__':
    print("--- Initial Data Collection & Model Training (Hierarchical Clustering) ---")
    initial_job_df = scrape_karkidi_jobs()

    if not initial_job_df.empty:
        # Convert skills column back to list if it was read from CSV as string representation of list
        if 'skills' in initial_job_df.columns and isinstance(initial_job_df['skills'].iloc[0], str):
            try:
                initial_job_df['skills'] = initial_job_df['skills'].apply(eval)
            except Exception as e:
                print(f"Could not convert 'skills' column from string to list: {e}")
                pass # Continue even if conversion fails, skills_text will be based on string

        initial_job_df_processed = preprocess_skills(initial_job_df.copy())
        tfidf_matrix, tfidf_vectorizer = vectorize_skills(initial_job_df_processed)

     
        try:
            n_clusters_input = input("Enter the desired number of clusters (press Enter for default 5): ")
            if n_clusters_input:
                n_clusters = int(n_clusters_input)
                if n_clusters <= 0:
                    raise ValueError("Number of clusters must be positive.")
                if n_clusters > tfidf_matrix.shape[0]:
                    print(f"Warning: Desired clusters ({n_clusters}) exceed available samples ({tfidf_matrix.shape[0]}). Setting n_clusters to number of samples minus 1 (or 1 if only one sample).")
                    n_clusters = tfidf_matrix.shape[0] - 1 if tfidf_matrix.shape[0] > 1 else 1
            else:
                n_clusters = 5 # Default value if no input
                print(f"Using default number of clusters: {n_clusters}")
        except ValueError as e:
            print(f"Invalid input for number of clusters: {e}. Using a default of 5 for sample data or max samples.")
            n_clusters = min(5, tfidf_matrix.shape[0] - 1 if tfidf_matrix.shape[0] > 1 else 1) # Default to 5 or max_samples - 1

        if n_clusters < 1:
            n_clusters = 1
        
        # Ensure we have at least 2 samples for clustering if n_clusters > 1
        if tfidf_matrix.shape[0] < 2 and n_clusters > 1:
            print("Warning: Only one sample available, forcing n_clusters to 1.")
            n_clusters = 1

        hierarchical_model = train_hierarchical_clustering_model(tfidf_matrix, n_clusters=n_clusters)
        initial_job_df_processed['cluster'] = hierarchical_model.labels_

        get_top_skills_per_cluster_hierarchical(hierarchical_model, tfidf_vectorizer, initial_job_df_processed)

        # Save the trained model, vectorizer, AND the original training TF-IDF matrix
        save_model(hierarchical_model, tfidf_vectorizer, tfidf_matrix) # Passing tfidf_matrix here
        # Save processed data for Streamlit's cluster insights
        initial_job_df_processed.to_csv('karkidi_jobs_initial_processed.csv', index=False)
        print("Processed initial job data saved to 'karkidi_jobs_initial_processed.csv'.")
    else:
        print("No initial jobs scraped (or sample data generated). Cannot train model.")

    # --- Load the model and vectorizer for daily runs / Streamlit ---
    print("\n--- Loading Trained Model and Vectorizer for Classification ---")
    loaded_hierarchical_model = None
    loaded_tfidf_vectorizer = None
    loaded_original_train_tfidf_matrix = None # New variable to store the loaded matrix

    try:
        loaded_hierarchical_model, loaded_tfidf_vectorizer, loaded_original_train_tfidf_matrix = load_model()
        print("Loaded TF-IDF vectorizer, AgglomerativeClustering model, and original training TF-IDF matrix.")
    except FileNotFoundError:
        print("Model, vectorizer, or original training TF-IDF matrix files not found. Please run the initial training phase first to create them.")

    if loaded_hierarchical_model and loaded_tfidf_vectorizer and loaded_original_train_tfidf_matrix is not None:
        # Example of classifying a new job (for testing)
        print("\n--- Classifying a Sample New Job ---")
        sample_new_job_data = [{'title': 'Full Stack Developer', 'company': 'Web Innovations', 'skills': ['python', 'javascript', 'react', 'node.js', 'sql'], 'job_description': 'Build amazing web apps.'}]
        sample_new_job_df = pd.DataFrame(sample_new_job_data)
        
        classified_sample_new_jobs = classify_new_jobs(
            sample_new_job_df,
            loaded_tfidf_vectorizer,
            loaded_hierarchical_model,
            loaded_original_train_tfidf_matrix # <--- This is the key argument now
        )
        print(classified_sample_new_jobs[['title', 'cluster']])

        # Example of simulating a daily run and notification (will be called by cron/scheduler in real deployment)
        print("\n--- Simulating Daily Job Monitoring and Notification (Conceptual) ---")
       
    else:
        print("Skipping classification and notification examples as model/vectorizer/original training TF-IDF matrix were not loaded.")
