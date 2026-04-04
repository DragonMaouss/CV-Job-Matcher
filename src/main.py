import os
import kagglehub
import pandas as pd
from preprocess import preprocess_text
from matcher import compute_similarity_matrix


DATASET_NAME = "surendra365/recruitement-dataset"
DATASET_FILE = "job_applicant_dataset.csv"
RESUME_COLUMN = "Resume"
JOB_COLUMN = "Job Description"


def load_dataset():
    # Download the dataset from Kaggle
    path = kagglehub.dataset_download(DATASET_NAME)
    # Load the dataset into a pandas DataFrame
    return pd.read_csv(f"{path}/{DATASET_FILE}")


def build_matching_results(similarity_matrix):
    results = []
    for i in range(similarity_matrix.shape[0]):
        # Find the best matching job for each resume
        best_job_idx = similarity_matrix[i].argmax()
        best_score = similarity_matrix[i][best_job_idx]
        results.append(
            {"cv_id": i, "job_id": int(best_job_idx), "score": float(best_score)}
        )
    return results


def main():
    df = load_dataset()
    # Extract the "Resume" and "Job_Description" columns
    resumes_df = df.loc[:, RESUME_COLUMN]
    jobs_df = df.loc[:, JOB_COLUMN]

    # Save the extracted columns as separate CSV files
    if not os.path.exists("../data"):
        os.makedirs("../data", exist_ok=True)

    resumes_df.to_csv("../data/resumes.csv", index=False)
    jobs_df.to_csv("../data/job.csv", index=False)

    # Preprocess all texts
    clean_resumes = [preprocess_text(r) for r in resumes_df]
    clean_jobs = [preprocess_text(j) for j in jobs_df]

    similarity_matrix = compute_similarity_matrix(clean_resumes, clean_jobs)

    # Sort by best matches and save
    results_df = pd.DataFrame(build_matching_results(similarity_matrix))
    results_df = results_df.sort_values(by="score", ascending=False)
    results_df.to_csv("../data/matching_results.csv", index=False)

    print(results_df.head(10))


if __name__ == "__main__":
    main()
