from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def compute_similarity_matrix(resumes, jobs):
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)

    # Combine resumes and jobs for fitting the vectorizer
    all_docs = resumes + jobs
    tfidf_matrix = vectorizer.fit_transform(all_docs)

    # Split the TF-IDF matrix back into resumes and jobs
    resume_matrix = tfidf_matrix[: len(resumes)]
    job_matrix = tfidf_matrix[len(resumes) :]

    # Compute cosine similarity between resumes and jobs
    similarity_matrix = cosine_similarity(resume_matrix, job_matrix)

    return similarity_matrix
